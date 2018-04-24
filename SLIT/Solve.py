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
         mask = [0], lvl = 0, weightS = 1, noise = 'gaussian', tau = 0, verbosity = 0):
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
    PSFconj = PSF.T
    #Size of the source
    ns1,ns2 = n1*size, n2*size
    #Number of starlet scales in source plane
    if lvl ==0:
        lvl = np.int(np.log2(ns2))
    else:
        lvl = np.min([lvl,np.int(np.log2(ns2))])

    lvlg = np.int(np.log2(n2))
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((n1,n2))


    #Noise in image plane
    if noise == 'gaussian':
        print('noise statistic is gaussain')
        sigma0 = tools.MAD(Y)
        print('sigma: ', sigma0)
    if noise == 'poisson':
        print('noise statistic is poisson')
        sigma0 = tools.MAD_poisson(Y,tau)
    if (noise == 'G+P') or (noise == 'P+G'):
        print('noise statistic is poisson and gaussain mixture')
        sigma0 = np.sqrt(tools.MAD_poisson(Y,tau, lvlg)**2+tools.MAD(Y)**2)


        plt.imshow(sigma0); plt.colorbar(); plt.show()

    #Mapping of an all-at-one image to source plane
    lensed = lens_one(Fkappa, n1,n2, size)
    #estimation of the frame of the image in source plane
    supp = np.zeros((lvl,lensed.shape[0],lensed.shape[1]))
    supp[:,lensed/lensed ==1] =1
    supp = 1

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
    #Forward operator
    def FW_op(X):
        return PSF_apply(F_apply(inverse(X)))
    #Inverse operator
    def IW_op(X):
        return transform(Finv_apply(PSFT_apply(X)))
    #Regularisation (Backward term)
    def reg0(X):
        return tools.Hard(X, levels, (ks), supp=supp)
    def reg00(X):
        return tools.Hard_Threshold(X, transform, inverse,levels, (ks), M = [0], supp=supp)
    def reg1(X):
        return tools.Soft(X, levels*weightS, kmax, supp=supp, Kill = 0)
    def reg_plus(X):
        Xreg = np.copy(X)
        Xreg[X<0] = 0
        return Xreg
    def reg_supp(X):
        X[X < 0] = 0
        return X*supp
    def reg_filter(X):
        return tools.mr_filter(X,levels,ks,10,transform, inverse, I_op(sigma0*np.ones((n1,n2))), lvl = lvl, supp = supp)
    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        #levels = simulate_noise(n1,n2, sigma0, size, I_op, transform,  lvl)
        levels = level_source(n1,n2,sigma0,size,PSFconj, Lens_op2, lensed, lvl)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels.fits', clobber=True)
    
    def mk_levels(sigma):
        return level_source(n1,n2,sigma0,size,PSFconj, Lens_op2, lensed, lvl)
##Compute spectral norms

    opwave_norm = spectralNorm(n1,n2,20,1e-10,IW_op,FW_op)
    op_norm = spectralNorm(ns1, ns2, 20, 1e-10, F_op, I_op)
    wave_norm = spectralNorm(ns1,ns2,20,1e-10,transform,inverse)
    if scheme == 'Vu':
        mu = 1.
        tau = 1./(mu*wave_norm**2+0.5*op_norm)
        if verbosity == 1:
            print(mu,tau)
    else:
        mu = .5/(opwave_norm)
        if verbosity == 1:
            print(mu)

    if (scheme == 'FISTA'):
        repeat = 2
    elif (scheme == 'Vu'):
        repeat = 1
    else:
        repeat = 1
    #Initialisation


    Res1= []



    for jr in range(repeat):
        trans = (transform(I_op(Y))/levels)*supp

        #trans[:,lensed==0] = 0
        trans[levels==0] =0
        ks0 = np.max(trans)*0.9
        print(ks0)
        ks=np.copy(ks0)
        steps = (ks0-kmax)/(niter-10.)
        karg = np.log(kmax/ks0)/(niter-10.)
        i = 0

        ts = 1
        csi = 0
        M = [0]
        Res1= []
        Res2 = []

        if np.sum(S0) == 0:
            S = np.random.randn(ns1, ns2) * np.median(sigma0)*0
        else:
            S = S0
        Snew = S
        alpha =transform(S)
        alphanew = np.copy(alpha)
        points = 0
        while i < niter:


            if scheme == 'FB':
                print('FB ', i)
                ks = ks0*np.exp(i*karg)
                ks = np.max([ks, kmax])
                S = np.copy(Snew)
                Snew = tools.Forward_Backward(Y, S, F_op, I_op, transform, inverse, mu, reg1, pos = 1)
                S[S<0] = 0
                FS = F_op(Snew)*mask
                if (noise == 'G+P') or (noise == 'P+G') and (i<10):
                    sigma = (tools.MAD(Y)+np.sqrt(FS/tau))
                    levels = mk_levels(sigma)

            elif scheme == 'FISTA':
                print('FISTA ', i)
                S = np.copy(Snew)
                alphanew = np.copy(alpha)
                alpha, csi, ts = tools.FISTA(Y, alphanew, F_op, I_op, mu, ts, csi, reg1, transform, inverse, mask = mask)
                Snew = inverse(alpha)
                FS = F_op(Snew)
            elif scheme == 'Vu':
                print('Vu ', i)
                S = np.copy(Snew)
                Snew,alpha = tools.Vu_Primal_dual(Y, S, alpha, mu, tau, F_op, I_op, transform, inverse, reg1, reg_plus)
                FS = F_op(Snew)
        #        plt.imshow(S)
        #        plt.show()

            

            SDR = tools.SDR(alpha, alphanew)
            Res = tools.Res(Y,FS,sigma0)
            #Convergence condition
            Res1.append(Res)
            Res2.append(SDR)
          #  ks = ks-steps
            if i>5:
                add = Criteria(i, Res1, Res2)
                if add == 0:
                    points = np.max([0,points-1])
                else:
                    points+=add
            if points >= 10:
                print('BREAK: algorithm converged at iteration: ', i)
                break

            i = i+1
            if i == niter:
                print('BREAK: Maximum number of iterations reached.')
        
#        alpha = transform(S)
        weightS = 1./(1.+np.exp(-10.*(levels*kmax-alpha)))
#    plt.show()
     #Final reconstruction of the source
    if np.size(np.shape(sigma0))>2:
        sigma0[sigma0==0]=np.mean(sigma0)
    if verbosity == 1:
        plt.imshow((Y-FS)/(sigma0)); plt.colorbar(); plt.show()
        plt.plot(Res1, 'b'); plt.show()
        plt.plot(Res2, 'r');
        plt.show()
        if noise == 'poisson':
            plt.subplot(211)
            plt.title('S')
            plt.imshow(S); plt.colorbar()
            plt.show()
    return Snew, FS


#############################SLIT MCA for blended lenses############################


def SLIT_MCA(Y, Fkappa, kmax, niter, riter, size,PSF, PSFconj, lvlg = 0, lvls = 0, noise = 'gaussian', tau =0, levels = [0], WS = 1, WG = 1, mask = [0,0], Ginit=0, Kills = 0, Killg = 0):
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
    PSFconj = PSF.T
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
    if noise == 'gaussian':
        print('noise statistic is gaussain')
        sigma0 = tools.MAD(Y)
    if noise == 'poisson':
        print('noise statistic is poisson')
        sigma0 = tools.MAD_poisson(Y,tau)
    if (noise == 'G+P') or (noise == 'P+G'):
        print('noise statistic is poisson and gaussain mixture')
        sigma0 = np.sqrt(tools.MAD_poisson(Y,tau, lvlg)**2+tools.MAD(Y)**2)

    #Mapping of an all-at-one image
    lensed = lens_one(Fkappa, n1,n2, size)

    
    supp = np.zeros((lvls,lensed.shape[0],lensed.shape[1]))
    supp[:,lensed/lensed ==1] =1

    #Limits of the image plane in source plane
    bound = mk_bound(Fkappa, n1,n2, size)



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
        return (PSF_apply(X))
    #Inverse Lens operator
    def IG_op(X):
        return (PSFT_apply(X))
    #Regularisation (Backward term)
    def regG0(X):
        return tools.Hard_Threshold(X, transform, inverse, levelg*kG)
    def regS0(X):
        return tools.Hard_Threshold(X, transform, inverse, levels*kS)
    def regG1(X):
        return tools.Soft(X, levelg*weightG, k, supp=supp, Kill = Killg)
    def regS1(X):
        return tools.Soft(X, levels*weightS, k, supp=supp, Kill = Kills)
    def reg_filter(X):
        return tools.mr_filter(X,levels,kmax,20,transform, inverse, I_op(sigma0*np.ones((n1,n2))), lvl = lvl, supp = supp)

    # Noise levels in image plane  in starlet space
    levelg = tools.level(n1, n2, lvlg) * sigma0
    #Noise simulations to estimate noise levels in source plane
    if np.sum(levels)==0:
        print('Calculating noise levels')
        levels = level_source(n1, n2, sigma0, size, PSFconj, Lens_op2, lensed, lvls)
        #levels[:,lensed ==0] = np.max(levels*10)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels_MCA.fits', clobber=True)
    

    
    #Computationt of spectral norms
    FS_norm = spectralNorm(ns1,ns2,20,1e-10,FS_op,IS_op)
    Star_norm_im = spectralNorm(n1,n2,20,1e-10,transform,inverse)
    Star_norm_s = spectralNorm(ns1,ns2,20,1e-10,transform,inverse)
    muG = 1./(2*Star_norm_im)**2
    muS = 1./(2*Star_norm_s*FS_norm)**2
    print(muS, muG)

    weightS = WS
    weightG = WG


    #Reweighting loop
    for it in range(3):
    #Initialisations

        i = 0
        K_s = np.zeros(niter)
        tg = 1
        ts = 1


        FS = 0
        FG = 0
        G = np.zeros((n1, n2))
        S = np.zeros((ns1, ns2))
        Gnew = np.copy(G)
        Snew = np.copy(S)
        alphaS = transform(S)
        csiS = np.copy(alphaS)
        alphaG = transform(G)
        csiG = np.copy(alphaG)

        k = tools.MOM(transform(IS_op(Y)), transform(IG_op(Y)), levels, levels)
        step = (k-kmax)/(niter-5)
        print(k)
        #Beginning of main loop
        while i < niter:
            print('main loop: ',i)
            kMOM = tools.MOM(alphaS, alphaG, levels, levelg)
            k = k-step
            if kMOM<k:
                k = kMOM
                print('MOMMYs Threshold: ', k)

            k = np.max([kmax, k])

            DS = Y-FG

            ts = 1
            pointS = 0
            Res1S = []
            Res2S = []
            for j in range(riter):
                S = np.copy(Snew)
                alphaS, csiS, ts = tools.FISTA(DS, alphaS, FS_op, IS_op, muS, ts, csiS, regS1, transform, inverse, pos = 1)
                Snew = inverse(alphaS)
                FS = FS_op(Snew)
                Res1S.append(tools.Res(S,Snew,sigma0))
                Res2S.append(tools.SDR(S,Snew))
                pointS = Criteria(j,Res1S,Res2S, pointS)



            DG = Y-FS
            tg = 1
            pointG = 0
            Res1G = []
            Res2G = []
            for j2 in range(riter):
                G = np.copy(Gnew)
              #  G, M = tools.Forward_Backward(DG, G, FG_op, IG_op, muG, reg_filter, pos=1)
                alphaG, csiG, tg = tools.FISTA(DG, alphaG, FG_op, IG_op, muG, tg, csiG, regG1, transform, inverse, pos = 1)
                Gnew = inverse(alphaG)
                FG = FG_op(Gnew)
                Res1S.append(tools.Res(S,Snew,sigma0))
                Res2S.append(tools.SDR(S,Snew))
                pointS = Criteria(j,Res1S,Res2S, pointS)
    #


            Res1 = tootls.Res(Y, FS+FG, sigma0)
            Res2 = (tools.SDR(Gnew, G)+tools.SDR(Snew, S))/2.
            K_s[i] = np.mean(newres)
            res = np.copy(newres)

            if i>5:
                points = Criteria(i,Res1, Res2, points)
            if points >= 10:
                print('BREAK: algorithm converged at iteration: ', i)
                break

            plt.figure(0)
            plt.subplot(221)
            plt.title('S')
            plt.imshow(Snew)
            plt.subplot(222)
            plt.title('FS')
            plt.imshow(FS)
            plt.subplot(223)
            plt.title('FG')
            plt.imshow(FG)
            plt.subplot(224)
            plt.title('Residuals')
            plt.imshow(Y-FS-FG)
            plt.savefig('Res'+str(i)+'.png')
            i +=1
            #Weighting
            
        weightS = 1./(1.+np.exp(-10.*(levels*kmax-alphaS)))
        weightG = 1./(1.+np.exp(-10.*(levelg*kmax-alphaG)))

        
  #  S, FS = SLIT(Y-G, Fkappa, kmax, niter, size, PSF,  PSFconj, levels = [0], scheme = 'FISTA', mask = mask, lvl = lvls)

    #Final reconstructions
    plt.show()
    plt.figure(1)
    plt.subplot(211)
    plt.plot(Res1)
    plt.subplot(212)
    plt.plot(Res2)
    plt.show()
    return Snew, FS,Gnew, FG






################################### TOOLS ###################################

def Criteria(i, Res1, Res2):

#    if np.abs(Res1[-1]-1) < 0.01:
#        point_res += 1
    if (np.abs(Res2[-1] - Res2[-2])  < 0.01*np.abs(Res2[0]-Res2[1])) and (np.abs(Res1[-1] - Res1[-2]) < 0.001*np.abs(Res1[0]-Res1[1])):
        points = 1
    else:
        points = 0

    return points


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


def level_source(n1,n2,sigma,size,PSFT, Lens_op2, lensed, lvl):
    ns1,ns2 = n1*size, n2*size
    ones = np.ones((n1,n2))
    lensed[lensed == 0] = 1
    noise = ones*sigma
    Hnoise = np.sqrt(scp.fftconvolve(noise**2, PSFT**2, mode = 'same'))#noise*np.sqrt(np.sum(PSFT**2))##
    Hnoise[np.isnan(Hnoise)==1] = 0
    FHnoise_old = Lens_op2(Hnoise)
    FHnoise = np.copy(FHnoise_old)
    FHnoise[FHnoise_old==0] = np.mean(FHnoise_old)*10.
    dirac = np.zeros((ns1,ns2))
    dirac[ns1/2,ns2/2] = 1
    wave_dirac = tools.wave_transform(dirac, lvl)
    levels = np.zeros(wave_dirac.shape)
    for i in range(lvl):
        if np.size(noise.shape) > 2:
            lvlso = (scp.fftconvolve(FHnoise[i, :, :] ** 2, wave_dirac[i, :, :] ** 2,
                                     mode='same'))
        else:
            lvlso = scp.fftconvolve(FHnoise ** 2, wave_dirac[i,:,:] ** 2,
                                    mode='same')
        #lvlso[lensed == 0] = np.max(lvlso)*100000000
        levels[i, :, :] = np.sqrt(np.abs(lvlso))
        levels[i,lvlso == 0] = 0
    return levels

def spectralNorm(n1,n2,Niter,tol,f,finv):
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
    matA = np.random.randn(n1,n2)
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



    
