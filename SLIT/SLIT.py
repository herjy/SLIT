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

def SLIT(img, theta, kmax, niter, size, decrease = 0, mymy = 0, levels = [0],
         pos = True, K=0, posit = False,weight = [0,0], soft = 0, mask = [0,0], PSF = [0], PSFconj = [0,0]):

    beta = theta
    #Initialisation of the lensed sourced
    Im = np.copy(img)*0.
    
    nt1,nt2 = np.shape(img)
    #Initialisation of the source
    nb1= nt1*size
    nb2 = nt2*size

    lvl = np.int(np.log2(nb2))
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((nt1,nt2))
    img = img*mask
    M = [0]
    #Noise in image plane
    sigma0 = MAD(img)
    #Mapping of an all-at-one image
    lensed = lens_one(beta, nt1,nt2, size)

    supp = np.zeros(lensed.shape)
    supp[lensed/lensed ==1] =1

    #Noise level map with convolution by wavelet elements +weird normalisation

    if np.sum(levels)==0:
        levels0 = lens_wnoise(nb1,nb2,lensed,lvl = lvl)
        if np.sum(PSF)!=0:
            levels = simulate_noise(nt1,nt2, size, beta, lensed, PSFconj)
        else:
            levels = levels0

    levels0 = np.copy(levels)
    S = Lens.image_to_source(img, size, beta, lensed = lensed)
    k = kmax


##Compute spectral norms
    def Finv_apply(I):
        return Lens.image_to_source(I, size, beta, lensed = lensed)
    def F_apply(Si):
        return Lens.source_to_image(Si, nt1, nt2,theta)
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
    if np.sum(PSF)>0:
        PSF_norm = spectralNorm(nt1,nt2,20,0.00000001, PSF_deproj, PSF_proj)
        mu = 1./(Star_norm_s*PSF_norm)**2
    else:
        mu = 1./(Star_norm_s*F_norm)**2

    
    Res2 = []
    R=0

    print(mu)
    S=np.random.randn(nb1,nb2)*sigma0
    alpha = np.zeros((lvl,nb1,nb2))
    csi = np.copy(alpha)
    t =1.

    i=0
    Res1 = []
    S=np.random.randn(nb1,nb2)*sigma0
    alpha = np.zeros((lvl,nb1,nb2))
    csi = np.copy(alpha)
    while i < niter:
        if np.sum(PSF) != 0:
            R = mu*scp.fftconvolve((img-Im),(PSFconj),mode = 'same')
            Rlens = Lens.image_to_source(R, size, beta, lensed=lensed)
        else:
            R = mu*(img-Im)
            Rlens = Lens.image_to_source(R, size, beta, lensed=lensed)
        
        alpha_new  = csi+mw.wave_transform(Rlens, lvl, newwave = 1)
        alpha_new= HT(alpha_new, kmax, levels, sigma0, soft = soft)*supp
    
        t_new = (1.+np.sqrt(1.+4.*t**2))/2.

        csi = alpha_new+((t-1)/t_new)*(alpha_new-alpha)
        alpha = alpha_new

        S = mw.iuwt(csi)
        Im = Lens.source_to_image(S, nt1, nt2,theta)
        if np.sum(PSF)!=0:
            Im = scp.fftconvolve(Im,PSF, mode = 'same')

        t = np.copy(t_new)
        Res1.append((np.std(img-Im)**2)/sigma0**2)
        if i >20:
            if np.abs(Res1[i]-Res1[i-10])<0.001 and Res1[i]<kmax:
                print('Converged')
                break
        i = i+1
    

    S = mw.iuwt(alpha)
    Im = Lens.source_to_image(S, nt1, nt2,theta)
    if np.sum(PSF)!=0:
        Im = scp.fftconvolve(Im,PSF, mode = 'same')

    plt.plot(Res1); plt.show()
    S[np.where(S<0)] = 0
    Im = Lens.source_to_image(S, nt1, nt2,theta)
    if np.sum(PSF)!=0:
        Im = scp.fftconvolve(Im,PSF, mode = 'same')
    return S, Im

def plot_cube(cube):
    n,n1,n2 = np.shape(cube)
    i = n/2
    if i == n/2.+0.5:
        i+=1
    j = 2
    for k in range(n):
        plt.subplot(i,j,k)
        plt.imshow(cube[k,:,:]); plt.colorbar()

    return None


def lens_wnoise(nb1,nb2,size,beta,lvl = 6):
    nt1,nt2 = nb1/np.float(size), nb2/np.float(size)
    all_one = np.ones((nt1,nt2))
    lensed = Lens.image_to_source(all_one, size,beta,lensed = [0])
    
    noise = (1./np.sqrt(lensed))
    
    n = np.int_(np.log2(nb1))+2
    nn = 2**n
 #   noise = noise/np.sum(noise)#np.multiply(sigma,1./np.sqrt(lensed))
    noise[np.where(lensed==0)]=1.
    dirac = np.zeros((nb1,nb2))
    dirac[nb1/2.,nb2/2.] = 1.
    wave_dirac = mw.wave_transform(dirac, lvl, newwave = 0)
    n,r,k = np.shape(wave_dirac)
    
    levels = np.zeros((n,r,k))

    for i in range(n):
        if np.size(noise.shape)>2:
            lvlso = (scp.fftconvolve(noise[i,:,:]**2, wave_dirac[i,:,:]**2,
                                    mode = 'same'))
        else:
            X = np.pad(noise,((np.int_((nn-nb1)/2),np.int_((nn-nb1)/2)),(np.int_((nn-nb2)/2),np.int_((nn-nb2)/2))), mode = 'edge')
            Y = np.pad(wave_dirac[i],((np.int_((nn-nb1)/2),np.int_((nn-nb1)/2)),(np.int_((nn-nb2)/2),np.int_((nn-nb2)/2))), mode = 'edge')
            lvlso = scp.fftconvolve(X**2,Y**2,
                                   mode = 'same')
            
            lvlso = lvlso[(nn-nb1)/2.:(nn+nb1)/2.,(nn-nb2)/2.:(nn+nb2)/2.]
 #           plt.imshow((lvls)); plt.colorbar();plt.show()
        
        levels[i,:,:] = np.sqrt(np.abs(lvlso))
         
    return levels


def spectralNorm(nx,ny,Niter,tol,f,finv):

##    Inputs:
##    nx: nombre de lignes l entree
##    ny: nombre de colonnes de l entree
##    nz: nombre d echelles (pour transformation en ondelette redondante)
##    Niter: nombre d iterations
##    tol: erreur de tolerance pour s arreter
##    f: l operateur
##    finv: l operateur inverse
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
        print "Iteration:"+str(it)+",Error:"+str(err)
        spNorm = spNorm_new
        it += 1       
    return spNorm


def level_source(n1,n2,x,y,N1,N2,Fout,lvl,lensed, PSF=[0,0]):

    dirac = np.zeros((n1,n2))
    i1 = (np.max(x)+np.min(x))/2
    i2 = (np.max(y)+np.min(y))/2
    dirac[i1,i2] = 1.
    wave_dirac = mw.wave_transform(dirac, lvl, newwave = 0)
    n, w1, w2 = np.shape(wave_dirac)

    if np.sum(PSF)!=0:
        for j in np.linspace(0,n-1,n):
            wave_dirac[j,:,:] = scp.fftconvolve(wave_dirac[j,:,:],PSF,mode = 'same')
    
    lensed_wave = np.zeros((n,N1,N2))
    for i in range(n):
        lensed_wave[i,:,:] = Lens.image_to_source(wave_dirac[i,:,:],x,y,N1,N2,Fout,lensed=lensed)**2
    wave_sum = np.sum(np.sum(lensed_wave,1),1)
    sig = np.sqrt(wave_sum)
    return sig

def lens_one(beta, nt1,nt2,size, ones = 0):
    # n1,n2: size of the postage stamp in image plane
    # N1,N2: size of the postage stamp in source plane 
    dirac = np.ones((nt1,nt2))
    lensed = Lens.image_to_source(dirac, size,beta,lensed = [0])#multi = 1)#
    if ones ==1:
        lensed[np.where(lensed == 0)] =1
    return lensed

def unlens_one(F, xt,yt,nt1,nt2,nb1,nb2, ones = 0):
    # n1,n2: size of the postage stamp in image plane
    # N1,N2: size of the postage stamp in source plane 
    dirac = np.ones((nb1,nb2))
    unlensed = Lens.source_to_image(dirac,nt1, nt1,F)
    if ones ==1:
        unlensed[np.where(lensed == 0)] =1
    return unlensed

def MOM(RG,RS,levelg,levels, lvlg,lvls, sigma):
    nt1,nt2 = np.shape(RG)
    ns1,ns2 = np.shape(RS)
    levels[levels==0]=1
    wG = mw.wave_transform(RG,lvlg, newwave = 1)/levelg/sigma
    wS = mw.wave_transform(RS,lvls, newwave = 1)/levels/sigma

    wGmax = np.max(wG)
    wSmax = np.max(wS)
    wmax = [wGmax, wSmax]

    k = np.min(wmax)+(max(wmax)-min(wmax))/100
    return k
    

def level(nt1,nt2,lvl = 6):
    dirac = np.zeros((nt1,nt2))
    dirac[nt1/2,nt2/2] = 1
    wave_dirac = mw.wave_transform(dirac,lvl, newwave = 0)
    
    wave_sum = np.sqrt(np.sum(np.sum(wave_dirac**2,1),1))
    levels = np.multiply(np.ones((lvl,nt1,nt2)).T,wave_sum).T
    
    return levels
    
def linorm(A,nit):
    ns,nb = np.shape(A)
    x0 = np.random.rand(nb)
    x0 = x0/np.sqrt(np.sum(x0**2))

    
    for i in np.linspace(0,nit-1,nit):
        x = np.dot(A,x0)
        xn = np.sqrt(np.sum(x**2))
        xp = x/xn
        y = np.dot(A.T,xp)
        yn = np.sqrt(np.sum(y**2)) 
        if yn < np.dot(y.T,x0) :
            break
        x0 = y/yn

    return xn

def MAD(x,n=3,fil=1):
        if fil == 1:
            meda = med.median_filter(x,size = (n,n))
        else:
            meda = np.median(x)
        medfil = np.abs(x-meda)
        sh = np.shape(x)
        sigma = 1.48*np.median((medfil))
        return sigma


def HT(alpha, k, levels, sigma, pen = 0, soft = 0):
    lvl, n1,n2 = np.shape(alpha)
    th = np.ones((lvl,n1,n2))*k
    th[0,:,:] = th[0,:,:]+1
    th[-1,:,:] = 0

    
    alpha0 = np.copy(alpha)
    th = th*levels*sigma

    if soft == 1:
        alpha= np.sign(alpha0)*(np.abs(alpha0)-th)
        alpha[np.where(np.abs(alpha)-th<0)]=0
    else:
        alpha[np.where(np.abs(alpha)-th<0)] = 0

    return alpha


##
############################ SLIT MCA FISTA

def simulate_noise(nt1,nt2, size, beta, lensed, PSFconj):
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
        noise_lens = (Lens.image_to_source(noise, size, beta, lensed =lensed))
        noise_lens[noise_lens ==0] = 1
       
        storage[:,:,:,i] = mw.wave_transform(noise_lens, lvl = lvl)
        
    w_levels = np.std(storage, axis = 3)

    return w_levels

def SLIT_MCA(img, theta, kmax, niter, riter, kinter, ninter, size,decrease = 0, FB = 0,
         pos = True, K=0, posit = False,reweighting = False, soft = 0, mask = [0,0],
             PSF = [0], PSFconj = [0,0], gauss = 0, Ginit=0, repeat = 1, mrfilter = 0):

    beta = theta
    #Initialisation of the lensed sourced
    Im = np.copy(img)*0.
    
    nt1,nt2 = np.shape(img)
    #Initialisation of the source
    nb1= nt1*size
    nb2 = nt2*size

    lvlg = np.int(np.log2(nt1))
    lvls = np.int(np.log2(nb1))
    #Masking if required
    if np.sum(mask) == 0:
        mask = np.ones((nt1,nt2))
    img = img*mask
    M = [0]
    #Noise in image plane
    sigma0 = wine.MCA.MAD(img)
    #Mapping of an all-at-one image
    lensed = lens_one(beta, nt1,nt2, size)
    bound = lensed/lensed
    bound[lensed == 0] = 0
    #Noise level map with convolution by wavelet elements +weird normalisation
    debound = mw.wave_transform(1-bound, lvls)
    
    levelg = level(nt1,nt1,lvl=lvlg)

    levels = simulate_noise(nt1,nt2, size, beta, lensed, PSFconj)
    levels0 = lens_wnoise(nb1,nb2,size,beta, lvl = lvls)

    S =Lens.image_to_source(img, size, beta, lensed = lensed)

    #K thresholds in units of sigma
    if np.sum(Ginit) == 0:
        Ginit = np.random.randn(nt1,nt2)*sigma0
    alphag = mw.wave_transform(Ginit, lvlg, newwave = 1)
    wFS = np.ones((lvlg,nt1,nt2))
    RG = np.copy(img)
    RS = np.copy(S)
    if np.sum(PSF) != 0:
        RS = scp.fftconvolve(RS,(PSFconj),mode = 'same')
    Res1 = []
    Res2 = []
    R=0
    i = 0
    S=np.random.randn(nb1,nb2)*sigma0
    G = Ginit
    FS = 0
    k = 0
    alphas = np.zeros((lvls,nb1,nb2))
    csis = np.copy(alphas)
    csig = np.copy(alphag)
    K_s = np.zeros((niter))
    norm = np.zeros((3,niter))
    tg =1

##
##Compute spectral norms
    def Finv_apply(I):
        return Lens.image_to_source(I, size, beta, lensed = lensed)
    def F_apply(Si):
        return Lens.source_to_image(Si, nt1, nt2,theta)
    def PSF_apply(i):
        return scp.fftconvolve(i,PSF,mode = 'same')
    def PSFT_apply(ii):
        return scp.fftconvolve(ii,PSFconj,mode = 'same')
    def star(x):
        return mw.wave_transform(x, lvlg, newwave = 1)
    def star_inv(x):
        return mw.iuwt(x)

    F_norm = spectralNorm(nt1,nt2,20,1e-10,F_apply,Finv_apply)
    Star_norm_im = spectralNorm(nt1,nt2,20,1e-10,star,star_inv)
    Star_norm_s = spectralNorm(nb1,nb2,20,1e-10,star,star_inv)
    if np.sum(PSF)>0:
        PSF_norm = spectralNorm(nt1,nt2,20,0.0000001,PSF_apply,PSFT_apply)
        muG = 1./(Star_norm_im**2)
        muS = 1./(Star_norm_s*F_norm*PSF_norm)**2
    else:
        muG = 1./(Star_norm_im**2)
        muS = 1./(Star_norm_s*F_norm)**2

    mu = 2./(Star_norm_s**2+F_norm**2)

    def HGT(X):
        return X
    def HG(X):
        return X
    def Phi(X):
        return mw.iuwt(X)
    def PhiT(X):
        return mw.wave_transform(X, lvlg, newwave = 1)

    def HFT(X):
        return Lens.image_to_source(X, size, beta, lensed=lensed)
    def HF(X):
        return Lens.source_to_image(X, nt1,nt2, theta)

    RS = Lens.image_to_source(img, size, beta, lensed = lensed)
    RG = np.copy(img)
    G = np.random.randn(nt1,nt2)*sigma0
    S = np.random.randn(nb1,nb2)*sigma0
    FS = np.random.randn(nt1,nt2)*sigma0
    i=0
    k = kmax
    while i < niter:
        print(i)
        DS = img-G
        j = 0
        ts = 1
        tr = np.zeros(riter)

        tauG = 1
        tauS = 1

        while j < riter:
            RS = muS*(DS-FS)
            if np.sum(PSF)!=0:
                RS = scp.fftconvolve(RS,(PSFconj),mode = 'same')
            RS = Lens.image_to_source(RS, size, beta, lensed=lensed)

            alphas_new  = csis+mw.wave_transform(RS, lvls, newwave = 1)
            
            alphas_new = HT(alphas_new, k, levels, sigma0, soft = soft)*bound
            ts_new = (1.+np.sqrt(1.+4.*ts**2))/2.
            csis = alphas_new+((ts-1)/ts_new)*(alphas_new-alphas)
            alphas = alphas_new
            ts = np.copy(ts_new)
            S = mw.iuwt(csis)

            FS = Lens.source_to_image(S, nt1, nt2,theta)
            

            if np.sum(PSF)!=0:
                FS = scp.fftconvolve(FS,PSF,mode = 'same')
            tr[j] = (np.std(img-FS-G)**2)/sigma0**2
            j+=1
        S = mw.iuwt(alphas)
        FS = Lens.source_to_image(S, nt1, nt2,theta)
        if np.sum(PSF)!=0:
            FS = scp.fftconvolve(FS,PSF,mode = 'same')

        DG = img-FS
        RG = muG*(DG-G)

        alphag_new  = csig+mw.wave_transform(RG, lvlg, newwave = 1)
        alphag_new = HT(alphag_new, kmax, levelg, sigma0,  soft = soft)
        tg_new = (1.+np.sqrt(1.+4.*tg**2))/2.
        csig = alphag_new+((tg-1)/tg_new)*(alphag_new-alphag)
        alphag = alphag_new

        tg = np.copy(tg_new)
        G = mw.iuwt(csig)  

        newres = (np.std(img-FS-G)**2)/sigma0**2
        K_s[i] = newres

        res = np.copy(newres)

        i = i+1

    
    S[np.where(S<0)] = 0
    FS = Lens.source_to_image(S, nt1, nt2,theta)
    if np.sum(PSF)!=0:
        FS = scp.fftconvolve(FS,PSF,mode = 'same')
    
    return S, FS,G



    
