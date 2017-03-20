#from __future__ import division
import wave_transform as mw
import numpy as np
import matplotlib.pyplot as plt 
import pyfits as pf
import matplotlib.cm as cm
from scipy import signal as scp
import scipy.ndimage.filters as med
from numpy import linalg as LA
import Lens
import warnings
warnings.simplefilter("ignore")

##SLIT: Sparse Lens Inversion Technique

def SLIT(img, Fkappa, kmax, niter, size, PSF,  PSFconj, levels = [0], mask = [0]):
    ##DESCRIPTION:
    ##    Function that estimates the source light profile from an image of a lensed source given the mass density profile.
    ##
    ##INPUTS:
    ##  -img: a 2-D image of a lensed source given as nt1xnt2 numpy array.
    ##  -Fkappa: an array giving the mapping between lens and source. This array is calculated from the lens mass density 
    ##          using tools from SLIT.Lens
    ##  -kmax: the detection threshold in units of noise levels. We usualy set this value to 5 to get a 5 sigma 
    ##          detection threshold.
    ##  -niter: maximal number of iterations of the algorithm.
    ##  -size: resoluution factor between lens and source grids such thathe size of the output source
    ##          will be nt1sizexnt2size
    ##  -PSF: the point spread function of the observation provided as a 2D array.
    ##  -PSFconj: The conjugate of the PSF. Usually computed via np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
    ##          butthe user has to make sure that the conjugate is well centered.
    ##
    ##OPTIONS:
    ##  -levels: an array that contains the noise levels at each band of the wavelet decomposition of the source.
    ##          If not provided, the routine will compute the levels and save them in a fits file 'Noise_levels.fits'
    ##          so that they can be used at a later time. This option allows to save time when running the same
    ##          experiment several times.
    ##  -mask: an array of zeros and one with size nb1xnb2. The zeros will stand for masked data.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile
    ##
    ##EXAMPLE:
    ##  S,FS = SLIT(img, Fkappa, 5, 100, 1, PSF,  PSFconj)

    
    nt1,nt2 = np.shape(img)
    
    #Size of the source
    nb1,nb2 = nt1*size, nt2*size
    #Number of starlet scales in source plane
    lvl = np.int(np.log2(nb2))
    
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((nt1,nt2))
    img = img*mask

    #Noise in image plane
    sigma0 = MAD(img)
    #Mapping of an all-at-one image to source plane
    lensed = lens_one(Fkappa, nt1,nt2, size)
    #estimation of the frame of the image in source plane
    supp = np.zeros(lensed.shape)
    supp[lensed/lensed ==1] =1

    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        levels = simulate_noise(nt1,nt2, size, Fkappa, lensed, PSFconj)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels.fits', clobber=True)

##Compute spectral norms
    def Finv_apply(I):
        return Lens.image_to_source(I, size, Fkappa, lensed = lensed)
    def F_apply(Si):
        return Lens.source_to_image(Si, nt1, nt2,Fkappa)
    def PSF_apply(i):
        return scp.fftconvolve(i,PSF,mode = 'same')
    def PSFT_apply(ii):
        return scp.fftconvolve(ii,PSFconj,mode = 'same')
    def star(x):
        return mw.wave_transform(x, lvl, newwave = 1)
    def star_inv(x):
        return mw.iuwt(x)
    def PSF_proj(x):
        return Finv_apply(PSFT_apply(x))
    def PSF_deproj(x):
        return PSF_apply(F_apply(x))
    F_norm = spectralNorm(nb1,nb2,20,1e-10,F_apply,Finv_apply)
    Star_norm_s = spectralNorm(nb1,nb2,20,1e-10,star,star_inv)
    PSF_norm = spectralNorm(nb1,nb2,20,0.00000001, PSF_deproj, PSF_proj)
    mu = 1./(Star_norm_s*PSF_norm*F_norm)**2

    #Initialisation
    R=0
    S=np.random.randn(nb1,nb2)*sigma0
    FS = 0
    alpha = np.zeros((lvl,nb1,nb2))
    csi = np.copy(alpha)
    t =1.
    Res1 = []
    i=0
    while i < niter:
        print(i)
        #Gradient step
        R = mu*scp.fftconvolve((img-FS),(PSFconj),mode = 'same')
        Rlens = Lens.image_to_source(R, size, Fkappa, lensed=lensed)

        #Thresholding in starlet space
        alpha_new  = csi+mw.wave_transform(Rlens, lvl, newwave = 1)
        alpha_new= ST(alpha_new, kmax, levels, sigma0)*supp

        #Inertial step
        t_new = (1.+np.sqrt(1.+4.*t**2))/2.
        csi = alpha_new+((t-1)/t_new)*(alpha_new-alpha)
        alpha = alpha_new

        #Reconstruction of the source for next step
        S = mw.iuwt(csi)
        FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
        FS = scp.fftconvolve(FS,PSF, mode = 'same')*mask

        t = np.copy(t_new)
        #Convergence condition
        Res1.append((np.std(img-FS)**2)/sigma0**2)
        if i >20:
            if np.abs(Res1[i]-Res1[i-10])<0.001 and Res1[i]<kmax:
                print('Converged')
                break
        i = i+1
    
    #Final reconstruction of the source
    S = mw.iuwt(alpha)
    S[np.where(S<0)] = 0
    FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
    FS = scp.fftconvolve(FS,PSF, mode = 'same')*mask
    return S, FS


#############################SLIT MCA for blended lenses############################


def SLIT_MCA(img, Fkappa, kmax, niter, riter, size,PSF, PSFconj, levels = [0], mask = [0,0], Ginit=0):
    ##DESCRIPTION:
    ##    Function that estimates the source and lens light profiles from an image of a
    ##          strong lens system
    ##
    ##INPUTS:
    ##  -img: a 2-D image of a lensed source given as nt1xnt2 numpy array.
    ##  -Fkappa: an array giving the mapping between lens and source. This array is calculated from the lens mass density 
    ##          using tools from SLIT.Lens
    ##  -kmax: the detection threshold in units of noise levels. We usualy set this value to 5 to get a 5 sigma 
    ##          detection threshold.
    ##  -niter: maximal number of iterations in the main loop over G.
    ##  -riter: maximal number of iterations in the inner loop over S.
    ##  -size: resoluution factor between lens and source grids such thathe size of the output source
    ##          will be nt1sizexnt2size
    ##  -PSF: the point spread function of the observation provided as a 2D array.
    ##  -PSFconj: The conjugate of the PSF. Usually computed via np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
    ##          butthe user has to make sure that the conjugate is well centered.
    ##
    ##OPTIONS:
    ##  -levels: an array that contains the noise levels at each band of the wavelet decomposition of the source.
    ##          If not provided, the routine will compute the levels and save them in a fits file 'Noise_levels.fits'
    ##          so that they can be used at a later time. This option allows to save time when running the same
    ##          experiment several times.
    ##  -mask: an array of zeros and one with size nb1xnb2. The zeros will stand for masked data.
    ##  -Ginit: Educated guedd for the lens galaxy light profile. if set to a 2D numpy array, the array will be used as
    ##          as an initialisation for G.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -G: the convolved lens light profile
    ##  -FS: the lensed version of the estimated source light profile
    ##
    ##EXAMPLE:
    ##  S,FS = SLIT(img, Fkappa, 5, 100, 1, PSF,  PSFconj)

    #Shape of the image
    nt1,nt2 = np.shape(img)
    #Initialisation of the source
    nb1= nt1*size
    nb2 = nt2*size
    #Number of starlet scales in source and image planes
    lvlg = np.int(np.log2(nt1))
    lvls = np.int(np.log2(nb1))
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((nt1,nt2))
    img = img*mask
    #Noise standard deviation in image plane
    sigma0 = MAD(img)
    #Mapping of an all-at-one image
    lensed = lens_one(Fkappa, nt1,nt2, size)
    #Limits of the image plane in source plane
    bound = lensed/lensed
    bound[lensed == 0] = 0
    #Noise levels in image plane  in starlet space
    levelg = level(nt1,nt1)
    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        levels = simulate_noise(nt1,nt2, size, Fkappa, lensed, PSFconj)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels.fits', clobber=True)



##
##Compute spectral norms
    def Finv_apply(I):
        return Lens.image_to_source(I, size, Fkappa, lensed = lensed)
    def F_apply(Si):
        return Lens.source_to_image(Si, nt1, nt2,Fkappa)
    def PSF_apply(i):
        return scp.fftconvolve(i,PSF,mode = 'same')
    def PSFT_apply(ii):
        return scp.fftconvolve(ii,PSFconj,mode = 'same')
    def star(x):
        return mw.wave_transform(x, lvlg, newwave = 1)
    def star_inv(x):
        return mw.iuwt(x)

    #Computationt of spectral norms
    F_norm = spectralNorm(nb1,nb2,20,1e-10,F_apply,Finv_apply)
    Star_norm_im = spectralNorm(nt1,nt2,20,1e-10,star,star_inv)
    Star_norm_s = spectralNorm(nb1,nb2,20,1e-10,star,star_inv)
    PSF_norm = spectralNorm(nt1,nt2,20,0.0000001,PSF_apply,PSFT_apply)
    muG = 1./(Star_norm_im**2)
    muS = 1./(Star_norm_s*F_norm*PSF_norm)**2


    #Initialisations
    if np.sum(Ginit)==0:
        G = np.random.randn(nt1,nt2)*sigma0
    else:
        G = Ginit
    S = np.random.randn(nb1,nb2)*sigma0
    FS = np.random.randn(nt1,nt2)*sigma0
    alphas = np.zeros((lvls,nb1,nb2))
    alphag = np.zeros((lvlg,nt1,nt2))
    csis = np.copy(alphas)
    csig = np.copy(alphag)
    K_s = np.zeros((niter))
    tg =1
    i=0
    #Beginning of main loop
    while i < niter:
        print(i)
        #The image to approximate by solving the problem in S
        DS = img-G
        #Reinitialisation
        j = 0
        ts = 1
        tr = np.zeros(riter)
        #Beginning of inner loop
        while j < riter:
            #Gradient step
            RS = muS*(DS-FS)
            RS = scp.fftconvolve(RS,(PSFconj),mode = 'same')
            RS = Lens.image_to_source(RS, size, Fkappa, lensed=lensed)
            alphas_new  = csis+mw.wave_transform(RS, lvls, newwave = 1)

            #Thresholding of the source
            alphas_new = ST(alphas_new, kmax, levels, sigma0)*bound

            #Inertial step
            ts_new = (1.+np.sqrt(1.+4.*ts**2))/2.
            csis = alphas_new+((ts-1)/ts_new)*(alphas_new-alphas)
            alphas = alphas_new
            ts = np.copy(ts_new)
            #Estimation of S for next loop
            S = mw.iuwt(csis)
            FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
            FS = scp.fftconvolve(FS,PSF,mode = 'same')

            #Residuals amplitude
            tr[j] = (np.std(img-FS-G)**2)/sigma0**2
            j+=1
        #Estimate of S at the end of inner loop
        S = mw.iuwt(alphas)
        FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
        FS = scp.fftconvolve(FS,PSF,mode = 'same')

        #Image to approximate by solving the problem in G
        DG = img-FS
        #Gradient step
        RG = muG*(DG-G)
        alphag_new  = csig+mw.wave_transform(RG, lvlg, newwave = 1)

        #Thresholding of the lens
        alphag_new = ST(alphag_new, kmax, levelg, sigma0)

        #inertial step
        tg_new = (1.+np.sqrt(1.+4.*tg**2))/2.
        csig = alphag_new+((tg-1)/tg_new)*(alphag_new-alphag)
        alphag = alphag_new
        tg = np.copy(tg_new)
        #Reconstruction of G for next iteration
        G = mw.iuwt(csig)  
        #residuals amplitude
        newres = (np.std(img-FS-G)**2)/sigma0**2
        K_s[i] = newres
        res = np.copy(newres)

        i +=1
    #Final reconstructions
    G = mw.iuwt(alphag)
    S[np.where(S<0)] = 0
    FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
    FS = scp.fftconvolve(FS,PSF,mode = 'same')
    
    return S, FS,G






################################### TOOLS ###################################

def plot_cube(cube):
    ##DESCRIPTION:
    ##    Plotting device that displays layers of a cube in different subplot panels.
    ##
    ##INPUTS:
    ##  -cube: Cube for which to plot the layers with shape (n,n1,n2) with n, the number of layers and n1xn2, the number of pixels.
    ##
    ##OUTPUTS:
    ##  -None
    n,n1,n2 = np.shape(cube)
    i = n/2
    if i == n/2.+0.5:
        i+=1
    j = 2
    for k in range(n):
        plt.subplot(i,j,k)
        plt.imshow(cube[k,:,:]); plt.colorbar()

    return None


def level(nt1,nt2):
    ##DESCRIPTION:
    ##    Estimates the noise levels in starlet space in image plane.
    ##
    ##INPUTS:
    ##  -nt1,nt2: shape of the image for which to get noise levels
    ##
    ##OUTPUTS:
    ##  -levels: units of noise levels at each scale and location of a starlet transform
    dirac = np.zeros((nt1,nt2))
    lvl = np.int(np.log2(nt1))
    dirac[nt1/2,nt2/2] = 1
    wave_dirac = mw.wave_transform(dirac,lvl, newwave = 0)
    
    wave_sum = np.sqrt(np.sum(np.sum(wave_dirac**2,1),1))
    levels = np.multiply(np.ones((lvl,nt1,nt2)).T,wave_sum).T
    
    return levels


def spectralNorm(nx,ny,Niter,tol,f,finv):
    ##DESCRIPTION:
    ##    Function that estimates the source light profile from an image of a lensed source given the mass density profile.
    ##
    ##INPUTS:
    ##    -nx,ny: shape of the input
    ##    -nz: number of decomposition scales (if the operator tis a multiscale decomposition for instance)
    ##    -Niter: number of iterations
    ##    -tol: tolerance error as a stopping criteria
    ##    -f: operator
    ##    -finv: inverse operator
    ##
    ##OUTPUTS:
    ##  -SspNorm: The spectral norm of the operator

    #### Initilize array with random numbers ###
    matA = np.random.randn(nx,ny)
    ### Normalize the input ###
    spNorm = LA.norm(matA)
    matA /= spNorm
    matA = np.array(matA)
    it = 0
    err = abs(tol)
    while it < Niter and err >= tol:
        ### Apply operator ###
        wt = f(matA)
        ### Apply joint operator ###
        matA = finv(wt)  
        ### Compute norm ###
        spNorm_new = LA.norm(matA)
        matA /= spNorm_new
        err = abs(spNorm_new - spNorm)/spNorm_new
        spNorm = spNorm_new
        it += 1       
    return spNorm



def lens_one(Fkappa, nt1,nt2,size):
    ##DESCRIPTION:
    ##    Function that maps an all at one image to source plane.
    ##
    ##INPUTS:
    ##  -Fkappa: the mapping between source and image planes
    ##  -nt1,nt2: the shape of the image.
    ##  -size: the factor that scales the shape of the source relative to the shape of the image
    ##
    ##OUTPUTS:
    ##  -lensed: the projection to source plane of an all at aone image.
    dirac = np.ones((nt1,nt2))
    lensed = Lens.image_to_source(dirac, size,Fkappa,lensed = [0])
    return lensed


def MAD(x,n=3):
    ##DESCRIPTION:
    ##  Estimates the noise standard deviation from Median Absolute Deviation
    ##
    ##INPUTS:
    ##  -x: a 2D image for which we look for the noise levels.
    ##
    ##OPTIONS:
    ##  -n: size of the median filter. Default is 3.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile
    meda = med.median_filter(x,size = (n,n))
    medfil = np.abs(x-meda)
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma


def ST(alpha, k, levels, sigma):
    ##DESCRIPTION:
    ##  Soft thresholding operator.
    ##
    ##INPUTS:
    ##  -alpha: the starlet decomposition to be thresholded.
    ##  -k: the threshold in units of noise levels (usually 5).
    ##  -levels: the noise levels at each scale and location of the starlet decomposition.
    ##  -sigma: the noise standard deviation.
    ##
    ##OUTPUTS:
    ##  -alpha: The thresholded coefficients.
    lvl, n1,n2 = np.shape(alpha)
    th = np.ones((lvl,n1,n2))*k
    th[0,:,:] = th[0,:,:]+1
    th[-1,:,:] = 0

    
    alpha0 = np.copy(alpha)
    th = th*levels*sigma

    alpha= np.sign(alpha0)*(np.abs(alpha0)-th)
    alpha[np.where(np.abs(alpha)-th<0)]=0


    return alpha


def simulate_noise(nt1,nt2, size, Fkappa, lensed, PSFconj):
    ##DESCRIPTION:
    ##  Simulates noise levels in source plane from lensing operator and convolution operator.
    ##
    ##INPUTS:
    ##  -nt1,nt2: the shape of the images for which to simulate noise maps.
    ##  -size: scaling factor for the shape of the source.
    ##  -Fkappa: Projection operator between lens and source plane.
    ##  -lensed: mapping of an all at one image to source plane.
    ##  -PSFconj: the conjugate of the PSF
    ##
    ##OPTIONS:
    ##  -n: size of the median filter. Default is 3.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile
    n = 500.
    nb1,nb2 = nt1*size, nt2*size
    lvl = np.int(np.log2(nb1))
    w_levels = np.zeros((lvl,nb1,nb2))
    
    dirac = np.zeros((nb1,nb2))
    dirac[nb1/2.,nb2/2.] = 1.

    wave_dirac = mw.wave_transform(dirac, lvl, newwave = 1)
    lens = 0
    lvl = np.int_(np.log2(nb1))
    nn = 2**(lvl+2)
    storage = np.zeros((lvl, nb1,nb2,n))
    for i in range(np.int(n)):
        noise = np.random.randn(nt1,nt2)
        noise = scp.fftconvolve(noise,PSFconj,mode = 'same')
        noise_lens = (Lens.image_to_source(noise, size, Fkappa, lensed =lensed))
        noise_lens[noise_lens ==0] = 1
        storage[:,:,:,i] = mw.wave_transform(noise_lens, lvl = lvl)
    w_levels = np.std(storage, axis = 3)
    return w_levels



    
