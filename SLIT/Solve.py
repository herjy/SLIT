#from __future__ import division
import numpy as np
import matplotlib.pyplot as plt 
import astropy.io.fits as pf
import matplotlib.cm as cm
import warnings
import multiprocess as mtp
from numpy import linalg as LA
from scipy import signal as scp
import scipy.ndimage.filters as med

from SLIT import Lens
from SLIT import tools

warnings.simplefilter("ignore")

##SLIT: Sparse Lens Inversion Technique

def SLIT(Y, Fkappa, kmax, niter, size, PSF, PSFconj, S0 = [0], levels = [0], scheme = 'FB',
         mask = [0], lvl = 0, weightS = 1, noise = 'gaussian', tau = 0, verbosity = 0, nweights = 1):
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
    ns1,ns2 = int(n1*size), int(n2*size)
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
    supp[:, lensed/lensed==1] = 1

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
        coeffs, _ = tools.wave_transform(x, lvl, newwave=1)
        return coeffs
    def inverse(x):
        return tools.iuwt(x, newwave=1)

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

    #Initialisation

    niter0 = np.copy(niter)

    Res1= []

    for jr in range(nweights):
        if jr!= nweights-1:
            niter = niter0/2
        else:
            niter = niter0

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
        alphaY = transform(I_op(Y))
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

                #S = np.copy(Snew)
                alphanew = np.copy(alpha)
                alpha, csi, ts = tools.FISTA(Y, alphanew, F_op, I_op, mu, ts, csi, reg1, transform, inverse, mask = mask)
                #Snew = inverse(alpha)
                #FS = F_op(Snew)

            elif scheme == 'Vu':
                print('Vu ', i)
                S = np.copy(Snew)
                Snew,alpha = tools.Vu_Primal_dual(Y, S, alpha, mu, tau, F_op, I_op, transform, inverse, reg1, reg_plus)

               # FS = F_op(Snew)

        #        plt.imshow(S)
        #        plt.show()

            

            SDR = tools.SDR(alpha, alphanew)

         #   Res = tools.Res(Y,FS,sigma0)
            #Convergence condition
          #  Res1.append(Res)
            Res2.append(SDR)
          #  ks = ks-steps
            if i>5:
                add = Criteria(i, SDR, Res2)

                if add == 0:
                    points = np.max([0,points-1])
                else:
                    points+=add

            if points >= 5:
                print('BREAK: algorithm converged at iteration: ', i)
                break

            i = i+1
            if i == niter:
                print('BREAK: Maximum number of iterations reached.')
        
#        alpha = transform(S)

        weightS = 2./(1.+np.exp(-10.*(levels*kmax-alpha)))
#    plt.show()
    Snew = inverse(alpha)
    FS = F_op(Snew)

     #Final reconstruction of the source
    if np.size(np.shape(sigma0))>2:
        sigma0[sigma0==0]=np.mean(sigma0)
    if verbosity == 1:
        plt.imshow((Y-FS)/(sigma0)); plt.colorbar(); plt.show()

    #    plt.plot(Res1, 'b'); plt.show()

        plt.plot(Res2, 'r');
        plt.show()
        if noise == 'poisson':
            plt.subplot(211)
            plt.title('S')
            plt.imshow(S); plt.colorbar()
            plt.show()
    return Snew, FS


#############################SLIT MCA for blended lenses############################


def SLIT_MCA(Y, Fkappa, kmax, niter, riter, size,PSF, PSFconj, lvlg = 0, lvls = 0, noise = 'gaussian', scheme = 'FISTA', decrease = 0,
             tau =0, levels = [0], WS = 1, WG = 1, mask = [0,0], Sinit = 0, Ginit=0, Kills = 0, Killg = 0, verbosity = 0, nweight = 5,
             original_fista=False, noise_levels_file='Noise_levels_MCA.fits'):
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

    niter = max([6, niter])

    #Shape of the image
    n1,n2 = np.shape(Y)
    #Initialisation of the source
    ns1 = int(n1*size)
    ns2 = int(n2*size)
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

    # Y = Y*mask  # instead we put noise where pixels are masked

    #Noise standard deviation in image plane
    if noise == 'gaussian':
        sigma0 = tools.MAD(Y)
        print('noise statistic is gaussian (sigma = {:.5e})'.format(sigma0))
    if noise == 'poisson':
        sigma0 = tools.MAD_poisson(Y,tau)
        print('noise statistic is poisson (sigma = {:.5e})'.format(sigma0))
    if (noise == 'G+P') or (noise == 'P+G'):
        sigma0 = np.sqrt(tools.MAD_poisson(Y,tau, lvlg)**2+tools.MAD(Y)**2)
        print('noise statistic is gaussian-poisson mixture (sigma = {:.3f})'.format(sigma0))

    # replace masked pixels with gaussian noise (fix k computation)
    masked_pixels = np.where(mask == 0)
    gaussian_noise_map = sigma0 * np.random.randn(n1, n2)
    Y[masked_pixels] = gaussian_noise_map[masked_pixels]


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
        coeffs, _ = tools.wave_transform(x, lvlg, newwave=1)
        return coeffs
    def inverse(x):
        return tools.iuwt(x, newwave=1)

    def FWS_op(X):
        return PSF_apply(F_apply(inverse(X)))
    #Inverse operator
    def IWS_op(X):
        return transform(Finv_apply(PSFT_apply(X)))
    def FWG_op(X):
        return PSF_apply(inverse(X))
    #Inverse operator
    def IWG_op(X):
        return transform(PSFT_apply(X))

    #Forward Source operator
    def FS_op(X):
        return PSF_apply(F_apply(X))
    #Inverse Source operator
    def IS_op(X):
        return Finv_apply(PSFT_apply(X))
    #Forward Lens operator
    def FG_op(X):
        return X#(PSF_apply(X))
    #Inverse Lens operator
    def IG_op(X):
        return X#(PSFT_apply(X))

    #Regularisation (Backward term)
    def regG0(X):
        return tools.Hard_Threshold(X, transform, inverse, levelg*kG)
    def regS0(X):
        return tools.Hard_Threshold(X, transform, inverse, levels*kS)
    def regG1(X):
        return tools.Soft(X, levelg*weightG, k, supp=1, Kill = Killg)
    def regS1(X):
        return tools.Soft(X, levels*weightS, k , supp=supp, Kill = Kills)
    def reg_plus(X):
        X[X<0] = 0
        return X
    def reg_filter(X):
        return tools.mr_filter(X,levels,kmax,20,transform, inverse, I_op(sigma0*np.ones((n1,n2))), lvl = lvl, supp = supp)

    # Noise levels in image plane  in starlet space
    levelg = tools.level(n1, n2, lvlg) * sigma0
    #Noise simulations to estimate noise levels in source plane
    if not np.any(levels):
        print('Calculating noise levels')
        levels = level_source(n1, n2, sigma0, size, PSFconj, Lens_op2, lensed, lvls)
        #levels[:,lensed ==0] = np.max(levels*10)
        #Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto(noise_levels_file, clobber=True)

    #Computationt of spectral norms
    FS_norm = spectralNorm(ns1,ns2,20,1e-10,FS_op,IS_op)
    FG_norm = spectralNorm(ns1, ns2, 20, 1e-10, FG_op, IG_op)
    wave_norm_im = spectralNorm(n1,n2,20,1e-10,transform,inverse)
    wave_norm_s = spectralNorm(ns1,ns2,20,1e-10,transform,inverse)
    opwaveS_norm = spectralNorm(n1, n2, 20, 1e-10, IWS_op, FWS_op)
    opwaveG_norm = spectralNorm(n1, n2, 20, 1e-10, IWG_op, FWG_op)


    if scheme == 'Vu':
        mu = 1.
        tauG = 0.5/(mu*wave_norm_im**2+0.5*FG_norm)
        tauS = 0.5 / (mu * wave_norm_s ** 2 + 0.5 * FS_norm)
        if verbosity == 1:
            print(tauS, tauG)

    else:
        muG = 1. / (opwaveG_norm)
        muS = 1. / (opwaveS_norm)
        if verbosity == 1:
            print(muS, muG)
    weightS = WS
    weightG = WG

    niter0 = np.copy(niter)
    riter0 = np.copy(riter)

    #Reweighting loop
    # k = tools.MOM(transform(Y), transform(Y), levelg, levelg)  # original code
    k = tools.MOM(transform(Y), transform(Y), levels, levelg)
    k0 = np.copy(k)
    karg = np.log(kmax / k0) / (niter - 10.)


    if np.sum(Ginit) == 0:
        G = np.random.randn(n1, n2) * sigma0
    else:
        G = Ginit
    if np.sum(Sinit) == 0:
        S = np.random.randn(ns1, ns2) * sigma0
    else:
        S = Sinit

    # FS = FG_op(G)  # original code
    # FG = FS_op(S)  # original code
    FS = 0
    FG = 0
    Gnew = np.copy(G)
    Snew = np.copy(S)
    alphaSnew = transform(S)
    csiS = np.copy(alphaSnew)
    alphaGnew = transform(G)
    csiG = np.copy(alphaGnew)

    for it in range(nweight):
    #Initialisations

        if it<np.max(range(nweight)):
            niter = niter0#/2
            riter = riter0#/2
        else:
            niter = niter0
            riter = riter0

        i = 0
        tg = 1
        ts = 1
        if decrease == 1:
            # k = tools.MOM(transform(Y), transform(Y), levelg, levelg)  # original code
            k = tools.MOM(transform(Y), transform(Y), levels, levelg)
        else:
            k = kmax
        k0 = np.copy(k)
        karg = np.log(kmax / k0) / (niter - 10.)
        print(k)
        step = (k-kmax)/(niter-5)
        Res1 = []
        Res2 = []
        DS = np.copy(Y)
        DG = np.copy(Y)

        #Beginning of main loop
        points  = 0
        Res1G = [1, 2]
        Res1S= [1,2]
        while i < niter:


            k = k-step#k0 * np.exp(i * karg)#
            # kMOM = tools.MOM(transform(DS), transform(DG), levelg, levelg)  # original code
            kMOM = tools.MOM(transform(DS), transform(DG), levels, levelg)

            if kMOM<k:
                k = np.copy(kMOM)
                print('MOMs threshold: ', k)
                step = (k-kmax)/(niter-i-5)

            k = np.max([kmax, k])
            print('main loop: ', i, k, kMOM)

            DS = Y - FG

            ts = 1
            pointS = 0
            Res1S = []
            Res2S = []

            pointS = 0
            for j in range(riter):
                if scheme == 'FISTA':
                    alphaS = np.copy(alphaSnew)
                    alphaSnew, csiS, ts = tools.FISTA(DS, alphaS, FS_op, IS_op, muS, ts, csiS, regS1, transform,
                                                      inverse, pos=0, original_fista=original_fista)

                if scheme == 'Vu':
                    alphaS = np.copy(alphaSnew)
                    S = np.copy(Snew)
                    Snew, alphaSnew = tools.Vu_Primal_dual(DS, S, alphaS, mu, tauS, FS_op, IS_op, transform, inverse,
                                                        regS1, reg_plus)
                Res2S.append(tools.SDR(alphaS, alphaSnew))

                if j > 5:
                    pointS = Criteria(j, Res1S, Res2S)
                if pointS >= 5:
                    if verbosity == 1:
                        print('Convergence on S in:', j, ' iterations.')
                        break
            if scheme == 'FISTA':
                Snew = inverse(alphaSnew)
                Snew[Snew<0] = 0
            FS = FS_op(Snew)

            DG = Y - FS

            tg = 1
            pointG = 0
            Res1G = []
            Res2G = []

            G = np.copy(Gnew)
            pointG = 0
            for j2 in range(1):
                if scheme == 'FISTA':
                    alphaG = np.copy(alphaGnew)
                    alphaGnew, csiG, tg = tools.FISTA(DG, alphaG, FG_op, IG_op, muG, tg, csiG, regG1, transform, inverse, pos = 0, original_fista=original_fista)



                if scheme == 'Vu':
                    alphaG = np.copy(alphaGnew)
                    G = np.copy(Gnew)
                    Gnew, alphaGnew = tools.Vu_Primal_dual(DG, G, alphaG, mu, tauG, FG_op, IG_op, transform, inverse, regG1,
                                                       reg_plus)
                Res2G.append(tools.SDR(alphaG, alphaGnew))

                if j2>5:
                    pointG = Criteria(j2, Res1G, Res2G)
                if pointG >=5:
                    if verbosity == 1:
                        print('Convergence on S in:', j2, ' iterations.')
                        break
            if scheme == 'FISTA':
                Gnew = inverse(alphaGnew)
                Gnew[Gnew<0] = 0
            FG = FG_op(Gnew)




            Res1.append(tools.Res(Y, FS+FG, sigma0))
            Res2.append((tools.SDR(Gnew, G)+tools.SDR(Snew, S))/2.)



            if i>5:
                points = Criteria(i, Res2, Res1)
            if points >= 5:
                if verbosity ==1:
                    print('BREAK: algorithm converged at iteration: ', i)
                break
            if verbosity ==1:
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

        weightS = 2./(1.+np.exp(-10.*(levels*kmax-alphaSnew)))
        weightG = 2./(1.+np.exp(-10.*(levelg*kmax-alphaGnew)))

        
 #   Snew, FS = SLIT(Y-FG, Fkappa, kmax, niter, size, PSF,  PSFconj, levels = [0], scheme = 'FISTA', mask = mask, lvl = lvls)

    #Final reconstructions
    if verbosity == 2:
        plt.show()
        plt.figure(1)
        plt.subplot(211)
        plt.plot(Res1)
        plt.subplot(212)
        plt.plot(Res2)

    return Snew, FS,Gnew, FG, Res1, Res2


def SLIT_MCA_HR(Y, Fkappa, kmax, niter, riter, size, PSF, lvlg=0, lvls=0, noise='gaussian', scheme='FISTA',
             tau=0, levels=[0], WS=1, WG=1, mask=[0, 0], Ginit=0, Kills=0, Killg=0, verbosity=0, nweight=5):
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

    niter = max([6, niter])

    # Shape of the image
    n1, n2 = np.shape(Y)
    # Initialisation of the source
    ns1 = n1 * size
    ns2 = n2 * size
    PSFconj = PSF.T
    # Number of starlet scales in source and image planes
    if lvlg == 0:
        lvlg = np.int(np.log2(n2))
    else:
        lvlg = np.min([lvlg, np.int(np.log2(n2))])
    lvls = lvlg
    if lvls > np.int(np.log2(ns2)):
        print('Error, too many wavelet levels for the source. Choose a smaller value for lvl')
        exit
    # Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((n1, n2))
    Y = Y * mask
    # Noise standard deviation in image plane
    if noise == 'gaussian':
        print('noise statistic is gaussain')
        sigma0 = tools.MAD(Y)
    if noise == 'poisson':
        print('noise statistic is poisson')
        sigma0 = tools.MAD_poisson(Y, tau)
    if (noise == 'G+P') or (noise == 'P+G'):
        print('noise statistic is poisson and gaussain mixture')
        sigma0 = np.sqrt(tools.MAD_poisson(Y, tau, lvlg) ** 2 + tools.MAD(Y) ** 2)

    # Mapping of an all-at-one image
    lensed = lens_one(Fkappa, ns1, ns2, 1)

    supp = np.zeros((lvls, lensed.shape[0], lensed.shape[1]))
    supp[:, lensed / lensed == 1] = 1

    # Limits of the image plane in source plane
    bound = mk_bound(Fkappa, ns1, ns2, 1)

    # Useful functions
    def Down(I):
        return tools.Downsample(I, size)

    def Up(I):
        return tools.Upsample(I, size)

    def Finv_apply(I):
        return Lens.image_to_source(I, 1, Fkappa, lensed=lensed)

    def Lens_op2(I):
        return Lens.image_to_source(I, 1, Fkappa, lensed=lensed, square=1)

    def F_apply(Si):
        return Lens.source_to_image(Si, ns1, ns2, Fkappa)

    def PSF_apply(i):
        return scp.fftconvolve(i, PSF, mode='same')

    def PSFT_apply(ii):
        return scp.fftconvolve(ii, PSFconj, mode='same')

    def transform(x):
        coeffs, _ = tools.wave_transform(x, lvlg, newwave=1)
        return coeffs

    def inverse(x):
        return tools.iuwt(x, newwave=1)

    def FWS_op(X):
        return Down(PSF_apply(F_apply(inverse(X))))

    # Inverse operator
    def IWS_op(X):
        return transform(Finv_apply(PSFT_apply(Up(X))))

    def FWG_op(X):
        return inverse(X)

    # Inverse operator
    def IWG_op(X):
        return transform(X)

    # Forward Source operator
    def FS_op(X):
        return Down(PSF_apply(F_apply(X)))

    # Inverse Source operator
    def IS_op(X):
        return Finv_apply(PSFT_apply(Up(X)))

    # Forward Lens operator
    def FG_op(X):
        return X  # (PSF_apply(X))

    # Inverse Lens operator
    def IG_op(X):
        return X  # (PSFT_apply(X))

    # Regularisation (Backward term)
    def regG0(X):
        return tools.Hard_Threshold(X, transform, inverse, levelg * kG)

    def regS0(X):
        return tools.Hard_Threshold(X, transform, inverse, levels * kS)

    def regG1(X):
        return tools.Soft(X, levelg * weightG, k, supp=1, Kill=Killg)

    def regS1(X):
        return tools.Soft(X, levels * weightS, k, supp=supp, Kill=Kills)

    def reg_plus(X):
        X[X < 0] = 0
        return X

    def reg_filter(X):
        return tools.mr_filter(X, levels, kmax, 20, transform, inverse, I_op(sigma0 * np.ones((n1, n2))), lvl=lvl,
                               supp=supp)

    # Noise levels in image plane  in starlet space
    levelg = tools.level(n1, n2, lvlg) * sigma0
    # Noise simulations to estimate noise levels in source plane
    if np.sum(levels) == 0:
        print('Calculating noise levels')
        levels = level_source_HR(n1, n2, size, sigma0, PSFconj, Lens_op2, Up, lvls)
        # levels[:,lensed ==0] = np.max(levels*10)
        # Saves levels
        hdus = pf.PrimaryHDU(levels)
        lists = pf.HDUList([hdus])
        lists.writeto('Noise_levels_MCA.fits', clobber=True)

    # Computationt of spectral norms
    FS_norm = spectralNorm(ns1, ns2, 20, 1e-10, FS_op, IS_op)
    FG_norm = spectralNorm(ns1, ns2, 20, 1e-10, FG_op, IG_op)
    wave_norm_im = spectralNorm(ns1, ns2, 20, 1e-10, transform, inverse)
    wave_norm_s = spectralNorm(ns1, ns2, 20, 1e-10, transform, inverse)
    opwaveS_norm = spectralNorm(n1, n2, 20, 1e-10, IWS_op, FWS_op)
    opwaveG_norm = spectralNorm(n1, n2, 20, 1e-10, IWG_op, FWG_op)

    if scheme == 'Vu':
        mu = 1.
        tauG = 0.5 / (mu * wave_norm_im ** 2 + 0.5 * FG_norm)
        tauS = 0.5 / (mu * wave_norm_s ** 2 + 0.5 * FS_norm)
        if verbosity == 1:
            print(tauS, tauG)

    else:
        muG = 1. / (opwaveG_norm)
        muS = 1. / (opwaveS_norm)
        if verbosity == 1:
            print(muS, muG)
    weightS = WS
    weightG = WG

    niter0 = np.copy(niter)
    riter0 = np.copy(riter)
    # Reweighting loop

    for it in range(nweight):
        # Initialisations

        if it < np.max(range(nweight)):
            niter = niter0  # /2
            riter = riter0  # /2
        else:
            niter = niter0
            riter = riter0

        i = 0
        tg = 1
        ts = 1

        FS = 0
        FG = 0
        G = np.random.randn(n1, n2) * sigma0
        S = np.random.randn(ns1, ns2) * sigma0
        Gnew = np.copy(G)
        Snew = np.copy(S)
        alphaSnew = transform(S)
        csiS = np.copy(alphaSnew)
        alphaGnew = transform(G)
        csiG = np.copy(alphaGnew)

        k = tools.MOM(transform(Y), transform(Y), levels, levelg) / 100.
        k0 = np.copy(k)
        karg = np.log(kmax / k0) / (niter - 5.)
        print(k)
        step = (k - kmax) / (niter - 5)
        Res1 = []
        Res2 = []
        DS = np.copy(Y)
        DG = np.copy(Y)

        # Beginning of main loop
        points = 0
        Res1G = [1, 2]
        Res1S = [1, 2]
        while i < niter:

            k = k0 * np.exp(i * karg)
            kMOM = tools.MOM(transform(DS), transform(DG), levels, levelg)

            if kMOM < k:
                k = np.copy(kMOM)
                print('MOMs threshold: ', k)
                step = (k - kmax) / (niter - i - 5)

            k = np.max([kmax, k])
            print('main loop: ', i, k, kMOM)

            DG = Y - FS

            tg = 1
            pointG = 0
            Res1G = []
            Res2G = []
            G = np.copy(Gnew)
            pointG = 0
            for j2 in range(1):
                if scheme == 'FISTA':
                    alphaG = np.copy(alphaGnew)
                    alphaGnew, csiG, tg = tools.FISTA(DG, alphaG, FG_op, IG_op, muG, tg, csiG, regG1, transform,
                                                      inverse, pos=0)

                if scheme == 'Vu':
                    alphaG = np.copy(alphaGnew)
                    G = np.copy(Gnew)
                    Gnew, alphaGnew = tools.Vu_Primal_dual(DG, G, alphaG, mu, tauG, FG_op, IG_op, transform, inverse,
                                                           regG1,
                                                           reg_plus)
                Res2G.append(tools.SDR(alphaG, alphaGnew))

                if j2 > 5:
                    pointG = Criteria(j2, Res1G, Res2G)
                if pointG >= 5:
                    if verbosity == 1:
                        print('Convergence on S in:', j2, ' iterations.')
                        break
            if scheme == 'FISTA':
                Gnew = inverse(alphaGnew)
                Gnew[Gnew < 0] = 0
            FG = FG_op(Gnew)

            DS = Y - FG

            ts = 1
            pointS = 0
            Res1S = []
            Res2S = []

            pointS = 0
            for j in range(riter):
                if scheme == 'FISTA':
                    alphaS = np.copy(alphaSnew)
                    alphaSnew, csiS, ts = tools.FISTA(DS, alphaS, FS_op, IS_op, muS, ts, csiS, regS1, transform,
                                                      inverse, pos=0)

                if scheme == 'Vu':
                    alphaS = np.copy(alphaSnew)
                    S = np.copy(Snew)
                    Snew, alphaSnew = tools.Vu_Primal_dual(DS, S, alphaS, mu, tauS, FS_op, IS_op, transform, inverse,
                                                           regS1, reg_plus)
                Res2S.append(tools.SDR(alphaS, alphaSnew))

                if j > 5:
                    pointS = Criteria(j, Res1S, Res2S)
                if pointS >= 5:
                    if verbosity == 1:
                        print('Convergence on S in:', j, ' iterations.')
                        break
            if scheme == 'FISTA':
                Snew = inverse(alphaSnew)
                Snew[Snew < 0] = 0
            FS = FS_op(Snew)

            Res1.append(tools.Res(Y, FS + FG, sigma0))
            Res2.append((tools.SDR(Gnew, G) + tools.SDR(Snew, S)) / 2.)

            if i > 5:
                points = Criteria(i, Res2, Res1)
            if points >= 5:
                if verbosity == 1:
                    print('BREAK: algorithm converged at iteration: ', i)
                break
            if verbosity == 1:
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
                plt.imshow(Y - FS - FG)
                plt.savefig('Res' + str(i) + '.png')
            i += 1
            # Weighting

        weightS = 2. / (1. + np.exp(-10. * (levels * kmax - alphaSnew)))
        weightG = 2. / (1. + np.exp(-10. * (levelg * kmax - alphaGnew)))

    #   Snew, FS = SLIT(Y-FG, Fkappa, kmax, niter, size, PSF,  PSFconj, levels = [0], scheme = 'FISTA', mask = mask, lvl = lvls)

    # Final reconstructions
    if verbosity == 2:
        plt.show()
        plt.figure(1)
        plt.subplot(211)
        plt.plot(Res1)
        plt.subplot(212)
        plt.plot(Res2)
        plt.show()
    return Snew, FS, Gnew, FG


################################### TOOLS ###################################

def Criteria(i, Res1, Res2):

#    if np.abs(Res1[-1]-1) < 0.01:
#        point_res += 1

    if (np.abs(Res2[-1] - Res2[-2])  < 0.1) and Res2[-1]<2.:#*np.abs(Res2[0]-Res2[1])):# and (np.abs(Res1[-1] - Res1[-2]) < 0.01*np.abs(Res1[0]-Res1[1])):
        points = 1
    else:
        points = 0
    if np.size(Res1)>5:
        if (np.abs(Res1[-1] - Res1[-2])  < 0.1*np.abs(Res1[1]-Res1[0])) and Res1[-1]>10.:
            points+=1

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


def level_source_HR(n1,n2, size, sigma,PSFT, Lens_op2, Up, lvl):
    ns1,ns2 = int(n1*size), int(n2*size)
    ones = np.ones((n1,n2))
    noise = Up(ones*sigma)*(size)**2
    Hnoise = np.sqrt(scp.fftconvolve(noise**2, PSFT**2, mode = 'same'))#noise*np.sqrt(np.sum(PSFT**2))##
    Hnoise[np.isnan(Hnoise)==1] = 0
    FHnoise_old = Lens_op2(Hnoise)
    FHnoise = np.copy(FHnoise_old)
    FHnoise[FHnoise_old==0] = np.mean(FHnoise_old)*10.
    dirac = np.zeros((ns1,ns2))
    dirac[int(ns1/2),int(ns2/2)] = 1
    print(dirac.shape, ns1,ns2)
    wave_dirac, _ = tools.wave_transform(dirac, lvl)
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


def level_source(n1,n2,sigma,size,PSFT, Lens_op2, lensed, lvl):
    ns1,ns2 = int(n1*size), int(n2*size)
    ones = np.ones((n1,n2))
    lensed[lensed == 0] = 1
    noise = ones*sigma

    Hnoise = noise*np.sqrt(np.sum(PSFT**2))##np.sqrt(scp.fftconvolve(noise**2, PSFT**2, mode = 'same'))#noise*np.sqrt(np.sum(PSFT**2))##

    Hnoise[np.isnan(Hnoise)==1] = 0
    FHnoise_old = Lens_op2(Hnoise)
    FHnoise = np.copy(FHnoise_old)
    FHnoise[FHnoise_old==0] = np.mean(FHnoise_old)*10.
    dirac = np.zeros((ns1,ns2))

    dirac[int(ns1/2),int(ns2/2)] = 1

    wave_dirac, _ = tools.wave_transform(dirac, lvl)
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

    lensed = Lens.image_to_source_bound(dirac, size, Fkappa,lensed = [0])

    bound = lensed/lensed
    bound[lensed==0]=0
    return bound



def mk_simu(n1,n2,lvl,size, sigma, I_op, transform, n):
    storage = np.zeros((lvl, int(n1*size), int(n2*size), n))
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
    from pathos.multiprocessing import ProcessingPool as Pool

    n = 500
    if Npar>mtp.cpu_count():
        Npar = mtp.cpu_count()
    ns1,ns2 = int(n1*size), int(n2*size)
#    lvl = np.int(np.log2(ns1))
    w_levels = np.zeros((lvl,ns1,ns2))

    p = Pool(Npar)
    storage = mk_simu(n1,n2,lvl,size, sigma, I_op, transform,n)

    w_levels = np.std(storage, axis = 3)
#    w_levels[0,:,:] = w_levels[0,:,:]*6/5

    return w_levels



    
