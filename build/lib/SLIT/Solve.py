#from __future__ import division
import wave_transform as mw
import numpy as np
import matplotlib.pyplot as plt 
import pyfits as pf
import matplotlib.cm as cm
from scipy import signal as scp
import scipy.ndimage.filters as med
import MuSCADeT as wine
from numpy import linalg as LA
import multiprocess as mtp
from pathos.multiprocessing import ProcessingPool as Pool
import Lens
import warnings
import tools
warnings.simplefilter("ignore")

##SLIT: Sparse Lens Inversion Technique

def SLIT(Y, Fkappa, kmax, niter, size, PSF, PSFconj, S0 = [0], levels = [0], scheme = 'FB',
         mask = [0], lvl = 0, weightS = 1, noise = 'gaussian', tau = 0):
    ##DESCRIPTION:
    ##    Function that estimates the source light profile from an image of a lensed source given the mass density profile.
    ##
    ##INPUTS:
    ##  -img: a 2-D image of a lensed source given as n1xn2 numpy array.
    ##  -Fkappa: an array giving the mapping between lens and source. This array is calculated from the lens mass density 
    ##          using tools from SLIT.Lens
    ##  -kmax: the detection threshold in units of noise levels. We usualy set this value to 5 to get a 5 sigma 
    ##          detection threshold.
    ##  -niter: maximal number of iterations of the algorithm.
    ##  -size: resoluution factor between lens and source grids such thathe size of the output source
    ##          will be n1sizexn2size
    ##  -PSF: the point spread function of the observation provided as a 2D array.
    ##  -PSFconj: The conjugate of the PSF. Usually computed via np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
    ##          butthe user has to make sure that the conjugate is well centered.
    ##
    ##OPTIONS:
    ##  -levels: an array that contains the noise levels at each band of the wavelet decomposition of the source.
    ##          If not provided, the routine will compute the levels and save them in a fits file 'Noise_levels.fits'
    ##          so that they can be used at a later time. This option allows to save time when running the same
    ##          experiment several times.
    ##  -mask: an array of zeros and one with size ns1xns2. The zeros will stand for masked data.
    ##
    ##OUTPUTS:
    ##  -S: the source light profile.
    ##  -FS: the lensed version of the estimated source light profile
    ##
    ##EXAMPLE:
    ##  S,FS = SLIT(img, Fkappa, 5, 100, 1, PSF,  PSFconj)

    n1,n2 = np.shape(Y)
  #  PSFconj = np.rot90(PSF, 2)
    #Size of the source
    ns1,ns2 = n1*size, n2*size
    #Number of starlet scales in source plane
    if lvl ==0:
        lvl = np.int(np.log2(ns2))
    else:
        lvl = np.min([lvl,np.int(np.log2(ns2))])
    
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((n1,n2))
    img = Y*mask

    #Noise in image plane
    sigma0 = MAD(Y)
    if noise == 'poisson':
        if tau ==0:
            print('error: Give exposure time')
        Y0 =np.copy(Y)
        sigma = np.copy(sigma0)
        Y = 2./tau*np.sqrt(tau*np.abs(Y)+tau*3./8.+sigma0)*np.sign(tau*Y+tau*3./8.+sigma0)

        


    sigma0 = MAD(Y)
   

    #Mapping of an all-at-one image to source plane
    lensed = lens_one(Fkappa, n1,n2, size)
    #estimation of the frame of the image in source plane
    supp = np.zeros((lvl,lensed.shape[0],lensed.shape[1]))
    supp[:,lensed/lensed ==1] =1

    #Useful functions
    def Finv_apply(I):
        return Lens.image_to_source(I, size, Fkappa, lensed = lensed)
    def Lens_op2(I):
        return Lens.image_to_source(I, size, Fkappa, lensed = lensed, square = 1)
    def F_apply(Si):
        return Lens.source_to_image(Si, n1, n2,Fkappa)
    def PSF_apply(i):
        return scp.fftconvolve(i,PSF,mode = 'same')
    def PSFT_apply(ii):
        return scp.fftconvolve(ii,PSFconj,mode = 'same')
    def transform(x):
        return tools.wave_transform(x, lvl, newwave = 1)
    def inverse(x):
        return tools.iuwt(x)

    #Forward operator
    def F_op(X):
        return PSF_apply(F_apply(X))
    #Inverse operator
    def I_op(X):
        return Finv_apply(PSFT_apply(X))
    #Regularisation (Backward term)
    def reg0(X):
        return tools.Hard(X, transform, inverse,levels, (ks), supp=supp)
    def reg00(X):
        return tools.Hard_Threshold(X, transform, inverse,levels, (ks), supp=supp)
    def reg1(X):
        return tools.Soft(X, transform, inverse,levels*weightS, kmax, supp=supp)
    def reg_filter(X):
        return tools.mr_filter(X,levels,ks,20,transform, inverse, lvl = lvl)
    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        #levels = simulate_noise(n1,n2, sigma0, size, I_op, transform,  lvl)
        levels = level_source(n1,n2,sigma0,size,PSFconj, Lens_op2, lensed, lvl)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels.fits', clobber=True)
    

##Compute spectral norms

    op_norm = spectralNorm(ns1,ns2,20,1e-10,F_op,I_op)
    wave_norm = spectralNorm(ns1,ns2,20,1e-10,transform,inverse)
    nu = 0.5#op_norm**2/(2*wave_norm**2)-1./(mu)
    mu = 1./(op_norm+wave_norm)

    if scheme == 'FB':
        repeat =1
    else:
        repeat = 2
    #Initialisation


    Res1= []
    
    tau = 0.5

    for jr in range(repeat):
        trans = (transform(I_op(Y))/levels)[:-1,:,:]
        ks0 = np.max(trans[levels[:-1,:,:]!=0])
        ks=np.copy(ks0)
        steps = (ks0-kmax)/(niter-5)
        karg = np.log(kmax/ks0)/(niter-5.)
        i = 0
        if np.sum(S0) == 0:
            S=np.random.randn(ns1,ns2)*np.median(sigma0)*0
        else:
            S = S0
        ts = 1
        csi = 0
        Res1= []
        Res2 = []
        

 

        alpha =transform(S)

        while i < niter:
            print(i)
            if scheme == 'FB':
                ks = ks0*np.exp(i*karg)
                ks = np.max([ks, kmax])
                S = tools.Forward_Backward(Y, S, F_op, I_op, mu, reg_filter, pos = 1)
                S[S<0] = 0
                FS = F_op(S)*mask
                
            else:                
                alpha, csi, ts = tools.FISTA(Y, alpha, F_op, I_op, mu, ts, csi, reg1, transform, inverse, mask = mask)
                S = inverse(alpha)
                FS = F_op(S)
            
            

    
            
            #Convergence condition
            Res1.append((np.std(Y-FS)**2)/np.median(sigma0)**2)
            Res2.append(ks)
          #  ks = ks-steps

            
            i = i+1

        S[S<0] = 0
        
#        alpha = transform(S)
        weightS = 1./(1.+np.exp(-10.*(levels*kmax-alpha)))
#    plt.show()
     #Final reconstruction of the source
    plt.plot(Res1, 'b'); plt.show()
    plt.plot(Res2, 'r');
    plt.show()
    if noise == 'poisson':
        plt.subplot(211)
        plt.title('S')
        plt.imshow(S); plt.colorbar()
        plt.show()
    FS = F_op(S)*mask
    return S, FS


#############################SLIT MCA for blended lenses############################


def SLIT_MCA(Y, Fkappa, kmax, niter, riter, size,PSF, PSFconj, lvlg = 0, lvls = 0, levels = [0], mask = [0,0], Ginit=0):
    ##DESCRIPTION:
    ##    Function that estimates the source and lens light profiles from an image of a
    ##          strong lens system
    ##
    ##INPUTS:
    ##  -img: a 2-D image of a lensed source given as n1xn2 numpy array.
    ##  -Fkappa: an array giving the mapping between lens and source. This array is calculated from the lens mass density 
    ##          using tools from SLIT.Lens
    ##  -kmax: the detection threshold in units of noise levels. We usualy set this value to 5 to get a 5 sigma 
    ##          detection threshold.
    ##  -niter: maximal number of iterations in the main loop over G.
    ##  -riter: maximal number of iterations in the inner loop over S.
    ##  -size: resoluution factor between lens and source grids such thathe size of the output source
    ##          will be n1sizexn2size
    ##  -PSF: the point spread function of the observation provided as a 2D array.
    ##  -PSFconj: The conjugate of the PSF. Usually computed via np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
    ##          butthe user has to make sure that the conjugate is well centered.
    ##
    ##OPTIONS:
    ##  -levels: an array that contains the noise levels at each band of the wavelet decomposition of the source.
    ##          If not provided, the routine will compute the levels and save them in a fits file 'Noise_levels.fits'
    ##          so that they can be used at a later time. This option allows to save time when running the same
    ##          experiment several times.
    ##  -mask: an array of zeros and one with size ns1xns2. The zeros will stand for masked data.
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
    n1,n2 = np.shape(Y)
    #Initialisation of the source
    ns1= n1*size
    ns2 = n2*size
    #Number of starlet scales in source and image planes
    if lvlg ==0:
        lvlg = np.int(np.log2(n2))
    else:
        lvlg = np.min([lvlg,np.int(np.log2(n2))])
    lvls = lvlg
    if lvls >np.int(np.log2(ns2)):
        print('Error, too many wavelet levels for the source. Choose a smaller value for lvl')
        exit
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((n1,n2))
    Y = Y*mask
    #Noise standard deviation in image plane
    sigma0 = MAD(Y)
    #Mapping of an all-at-one image
    lensed = lens_one(Fkappa, n1,n2, size)

    
    supp = np.zeros((lvls,lensed.shape[0],lensed.shape[1]))
    supp[:,lensed/lensed ==1] =1

    #Limits of the image plane in source plane
    bound = mk_bound(Fkappa, n1,n2, size)
    #Noise levels in image plane  in starlet space
    levelg = level(n1,n2, lvlg)*sigma0


    #Useful functions
    def Finv_apply(I):
        return Lens.image_to_source(I, size, Fkappa, lensed = lensed)
    def F_apply(Si):
        return Lens.source_to_image(Si, n1, n2,Fkappa)
    def PSF_apply(i):
        return scp.fftconvolve(i,PSF,mode = 'same')
    def PSFT_apply(ii):
        return scp.fftconvolve(ii,PSFconj,mode = 'same')
    def transform(x):
        return tools.wave_transform(x, lvlg)
    def inverse(x):
        return tools.iuwt(x)

    #Forward Source operator
    def FS_op(X):
        return PSF_apply(F_apply(X))
    #Inverse Source operator
    def IS_op(X):
        return Finv_apply(PSFT_apply(X))
    #Forward Lens operator
    def FG_op(X):
        return ((X))
    #Inverse Lens operator
    def IG_op(X):
        return ((X))
    #Regularisation (Backward term)
    def regG0(X):
        return tools.Hard_Threshold(X, transform, inverse, levelg*kG)
    def regS0(X):
        return tools.Hard_Threshold(X, transform, inverse, levels*kS)
    def regS1(X):
        return tools.Soft(X, transform, inverse, levels*kmax*weightS, supp = supp)
    def regG1(X):
        return tools.Soft(X, transform, inverse, levelg*(kmax)*weightG, supp = 1)
    def reg_filter(X):
        return tools.mr_filter(X, levelg*kG*sigma0, 20, transform, inverse, Soft = 0, pos = 1)

    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        levels = simulate_noise(n1,n2, sigma0, size, IS_op, transform, lvls)
        levels[:,lensed ==0] = np.max(levels*10)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels_MCA.fits', clobber=True)
    

    
    #Computationt of spectral norms
    FS_norm = spectralNorm(ns1,ns2,20,1e-10,FS_op,IS_op)
    Star_norm_im = spectralNorm(n1,n2,20,1e-10,transform,inverse)
    Star_norm_s = spectralNorm(ns1,ns2,20,1e-10,transform,inverse)
    muG = 1./(Star_norm_im**2)
    muS = 1./(Star_norm_s*FS_norm)**2
    print(muS, muG)

    weightS = 1
    weightG = 1

    #Reweighting loop
    for it in range(3):
    #Initialisations
        FS = 0
        G = np.zeros((n1,n2))
        S = np.zeros((ns1,ns2))
        alphaS = transform(S)
        csiS = np.copy(alphaS)
        alphaG = transform(G)
        csiG = np.copy(alphaG)
        i = 0
        K_s = np.zeros(niter)
        tg=1
        

        #Beginning of main loop
        while i < niter:
            print('main loop: ',i)

            DS = Y-G

            ts = 1
            for j in range(riter):
    #            S = tools.Forward_Backward(DS, S, FS_op, IS_op, muS, regS0)
                alphaS, csiS, ts = tools.FISTA(DS, alphaS, FS_op, IS_op, muS, ts, csiS, regS1, transform, inverse, pos = 1)

            S = inverse(alphaS)#*supp
            S[S<0] = 0
            FS = FS_op(S)



            DG = Y-FS
            for j2 in range(riter):
                alphaG, csiG, tg = tools.FISTA(DG, alphaG, FG_op, IG_op, muG, tg, csiG, regG1, transform, inverse, pos = 1)
            #Image to approximate by solving the problem in G
            G = inverse(alphaG)
           
    #        
            newres = (np.std(Y-FS-G)**2)/sigma0**2
            K_s[i] = newres
            res = np.copy(newres)

            plt.figure(0)
            plt.subplot(221)
            plt.title('S')
            plt.imshow(S)
            plt.subplot(222)
            plt.title('FS')
            plt.imshow(FS)
            plt.subplot(223)
            plt.title('G')
            plt.imshow(G)
            plt.subplot(224)
            plt.title('Residuals')
            plt.imshow(Y-FS-G)
            plt.savefig('Res'+str(i)+'.png')
            i +=1
            #Weighting
            
        weightS = 1./(1.+np.exp(-10.*(levels*kmax*sigma0-alphaS)))
        weightG = 1./(1.+np.exp(-10.*(levelg*kmax*sigma0-alphaG)))

        
    S, FS = SLIT(Y-G, Fkappa, kmax, niter, size, PSF,  PSFconj, S0 = S, levels = levels, mask = mask, lvl = lvls)

    #Final reconstructions
    plt.show()
    plt.plot(K_s); plt.show()
    return S, FS,G






################################### TOOLS ###################################

def MOM(Y, levels, levelg, kmax, transform, inverse, IS_op, sigma, niter, I = 0):
    
    Gw0 = transform((Y))[:-1,:,:]
    levelg1 = levelg[:-1,:,:]
    Gw = Gw0[levelg1!=0]/levelg[levelg1!=0]/sigma
    kG = np.max(Gw)
    kG0 = kG
    stepG = (kG-kmax)/(niter-I-5)


    FS0 = Y
    Sw0 = transform(IS_op(FS0))[:-1,:,:]
    levels1 = levels[:-1,:,:]
    Sw = Sw0[levels1!=0]/levels[levels1!=0]/sigma
    kS = np.max(Sw)
    
    k =np.min([kG,kS])
    k = np.max([k,kmax])+(np.abs(kS-kG))/100.
    step = (k-kmax)/(niter-I-5)
    return k, step

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


def level(n1,n2, lvl):
    ##DESCRIPTION:
    ##    Estimates the noise levels in starlet space in image plane.
    ##
    ##INPUTS:
    ##  -n1,n2: shape of the image for which to get noise levels
    ##
    ##OUTPUTS:
    ##  -levels: units of noise levels at each scale and location of a starlet transform
    dirac = np.zeros((n1,n2))
 #   lvl = np.int(np.log2(n1))
    dirac[n1/2,n2/2] = 1
    wave_dirac = mw.wave_transform(dirac,lvl, newwave = 0)
    
    wave_sum = np.sqrt(np.sum(np.sum(wave_dirac**2,1),1))
    levels = np.multiply(np.ones((lvl,n1,n2)).T,wave_sum).T
    
    return levels

def level_source(n1,n2,sigma,size,PSFT, Lens_op2, lensed, lvl):
    ns1,ns2 = n1*size, n2*size
    ones = np.ones((n1,n2))
    lensed[lensed == 0] = 1
    noise = ones*sigma
    Hnoise = noise*np.sqrt(np.sum(PSFT**2))#np.sqrt(scp.fftconvolve(noise**2, PSFT**2, mode = 'same'))##
    FHnoise = Lens_op2(Hnoise)
    FHnoise[FHnoise==0] = np.mean(FHnoise)*10.
    dirac = np.zeros((ns1,ns2))
    dirac[ns1/2,ns2/2] = 1
    wave_dirac = mw.wave_transform(dirac, lvl)
    levels = np.zeros(wave_dirac.shape)
    for i in range(lvl):
        if np.size(noise.shape) > 2:
            lvlso = (scp.fftconvolve(FHnoise[i, :, :] ** 2, wave_dirac[i, :, :] ** 2,
                                     mode='same'))
        else:
            lvlso = scp.fftconvolve(FHnoise ** 2, wave_dirac[i,:,:] ** 2,
                                    mode='same')
        levels[i, :, :] = np.sqrt(np.abs(lvlso))

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



def lens_one(Fkappa, n1,n2,size):
    ##DESCRIPTION:
    ##    Function that maps an all at one image to source plane.
    ##
    ##INPUTS:
    ##  -Fkappa: the mapping between source and image planes
    ##  -n1,n2: the shape of the image.
    ##  -size: the factor that scales the shape of the source relative to the shape of the image
    ##
    ##OUTPUTS:
    ##  -lensed: the projection to source plane of an all at aone image.
    dirac = np.ones((n1,n2))
    lensed = Lens.image_to_source(dirac, size,Fkappa,lensed = [0])
    return lensed

def mk_bound(Fkappa, n1,n2,size):
    ##DESCRIPTION:
    ##    Function that returns the support of the lens image in source plane.
    ##
    ##INPUTS:
    ##  -Fkappa: the mapping between source and image planes
    ##  -n1,n2: the shape of the image.
    ##  -size: the factor that scales the shape of the source relative to the shape of the image
    ##
    ##OUTPUTS:
    ##  -lensed: the projection to source plane of an all at aone image.
    dirac = np.ones((n1,n2))
    lensed = Lens.image_to_source_bound(dirac, size,Fkappa,lensed = [0])
    bound = lensed/lensed
    bound[lensed==0]=0
    return bound


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
    x = mw.wave_transform(x, np.int(np.log2(x.shape[0])))[0,:,:]
    meda = med.median_filter(x,size = (n,n))
    medfil = np.abs(x-meda)
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def MAD_poisson(x,tau,n=3):
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
    n1,n2 = np.shape(x)
    lvl = np.int(np.log2(x.shape[0]))-1
    new_x, y = wine.MCA.mr_filter(x,20,8,MAD(x), lvl = lvl)
    plt.imshow(new_x); plt.show()
    sigma = np.sqrt(np.abs(new_x)/tau)
    return sigma



def ST(alpha, k, levels, sigma, hard = 0):
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
    th[0,:,:] = th[0,:,:]+3
    th[-1,:,:] = 0

    
    alpha0 = np.copy(alpha)
    th = th*levels*sigma

    if hard == 0:
        alpha= np.sign(alpha0)*(np.abs(alpha0)-th)
    alpha[np.where(np.abs(alpha)-th<0)]=0


    return alpha


def mk_simu(n1,n2,lvl,size, sigma, I_op, transform, n):
    storage = np.zeros((lvl,n1*size, n2*size, n))
    for i in range(n):
        noise = np.random.randn(n1,n2)*sigma
        noise_lens = I_op(noise)
        noise_lens[noise_lens ==0] = 1
        storage[:,:,:,i] = transform(noise_lens)
    return storage


def simulate_noise(n1,n2, sigma, size, I_op, transform, lvl, Npar = np.int(mtp.cpu_count()/2)):
    ##DESCRIPTION:
    ##  Simulates noise levels in source plane from lensing operator and convolution operator.
    ##
    ##INPUTS:
    ##  -n1,n2: the shape of the images for which to simulate noise maps.
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
    n = 500
    if Npar>mtp.cpu_count():
        Npar = mtp.cpu_count()
    ns1,ns2 = n1*size, n2*size
#    lvl = np.int(np.log2(ns1))
    w_levels = np.zeros((lvl,ns1,ns2))

    p = Pool(Npar)
    storage = mk_simu(n1,n2,lvl,size, sigma, I_op, transform,n)

    w_levels = np.std(storage, axis = 3)
#    w_levels[0,:,:] = w_levels[0,:,:]*6/5

    return w_levels



    
