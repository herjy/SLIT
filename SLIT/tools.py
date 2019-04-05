import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
import scipy.signal as scs
import scipy.ndimage.filters as scf

from SLIT import transform as tr

# try:
#     import pysap
# except ImportError:
#     pysap_installed = False
# else:
#     pysap_installed = True
pysap_installed = False

# TODO : terminate proper PySAP inegration (i.e. manage the 'pysap_transform' 
# object returned by wave_transform(), then pass it to iuwt())


def wave_transform(img, lvl, Filter='Bspline', newwave=1, convol2d=0, verbose=False):
    original_warning = "--> using original wavelet algorithm instead"
    
    if pysap_installed:
        if newwave == 0:
            coeffs, pysap_transform = tr.uwt_pysap(img, lvl, Filter=Filter)
        else:
            if verbose:
                print("WARNING : PySAP does not support 2nd gen starlet")
                print(original_warning)
            coeffs = tr.uwt_original(img, lvl, Filter='Bspline', 
                                     newwave=newwave, convol2d=convol2d)
            pysap_transform = None
    else:
        if verbose:
            print("WARNING : PySAP not installed or not found")
            print(original_warning)
        coeffs = tr.uwt_original(img, lvl, Filter='Bspline', 
                                 newwave=newwave, convol2d=convol2d)
        pysap_transform = None
    return coeffs, pysap_transform

def iuwt(wave, newwave=1, convol2d=0, pysap_transform=None, verbose=False):
    original_warning = "--> using original transform algorithm instead"
    
    if pysap_installed:
        if newwave == 0:
            if pysap_transform is None:
                raise RuntimeError("PySAP transform required for synthesis")
            recon = tr.iuwt_pysap(wave, pysap_transform, fast=True)
        else:
            if verbose:
                print("WARNING : PySAP does not support 2nd gen starlet")
                print(original_warning)
            coeffs = tr.iuwt_original(wave, convol2d=convol2d, newwave=newwave, fast=True)
    
    else:
        if verbose:
            print("WARNING : PySAP not installed or not found")
            print(original_warning)
        recon = tr.iuwt_original(wave, convol2d=convol2d, newwave=newwave)
    return recon
        
def MOM(S, G, levelS, levelG):
    S = S[:-1,:,:]
    G = G[:-1,:,:]
    levelS = levelS[:-1,:,:]
    levelG = levelG[:-1,:,:]

    sel = ((levelS!=0))
    Smax = np.max(np.abs(S[sel])/levelS[sel])
    Gmax = np.max(np.abs(G[levelG!=0])/levelG[levelG!=0])

    k = np.min([Smax, Gmax])
    return k+0.001*np.abs(Smax-Gmax)

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
    coeffs, _ = wave_transform(x, np.int(np.log2(x.shape[0])))
    x = coeffs[0,:,:]
    meda = scf.median_filter(x,size = (n,n))
    medfil = np.abs(x-meda)#np.median(x))
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def MAD_box(x, tau):
    n1,n2 = x.shape
    coeffs, _ = wave_transform(x,2)
    xw, _ = coeffs[0,:,:]
    winsize = 6
    xw_pad = np.pad(xw, ((winsize/2, winsize/2),(winsize/2, winsize/2)), mode = 'symmetric')

    sigma = np.zeros((xw.shape))
    for i in range(n1):
        for j in range(n2):
            area = xw_pad[i+winsize-winsize/2:i+winsize+winsize/2,j+winsize-winsize/2:j+winsize+winsize/2]
            sigma[i,j] = 1.48*np.median(np.abs(area-np.median(area)))
    return sigma

def MAD_poisson(x,tau,lvl):
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

    x0 = np.copy(x)
    def transform(i):
        coeffs, _ = wave_transform(i,lvl)
        return coeffs
    levels = level(n1,n2,lvl)*MAD(x)
    new_x = np.copy(x)
    new_x, y = mr_filter(new_x,levels, 8, 20, transform, iuwt, MAD(x), lvl = lvl)


    sigma = np.sqrt(np.abs(new_x)/tau)
    return sigma


def level_poisson(n1,n2, lvl,transform,sigma):
    dirac = np.zeros((n1,n2))
    dirac[n1/2,n2/2] = 1
    wave_dirac = transform(dirac)
    levels = np.zeros(wave_dirac.shape)
    for i in range(lvl):
        if np.size(sigma.shape) > 2:
            lvlso = (scs.fftconvolve(sigma[i, :, :] ** 2, wave_dirac[i, :, :] ** 2,
                                     mode='same'))
        else:
            lvlso = scs.fftconvolve(sigma ** 2, wave_dirac[i,:,:] ** 2,
                                    mode='same')

        levels[i, :, :] = np.sqrt(np.abs(lvlso))

    return levels

def Forward_Backward(Y, X, F_op, I_op, transform, inverse, mu, reg, pos = 1, subiter = 0):
    R = mu*I_op(Y-F_op(X))
    Xnew = np.copy(X+R)
    Xnew = inverse(reg(transform(Xnew)))
    return Xnew

def Vu_Primal_dual(Y, X, Z, mu, tau, F_op, I_op, transform, inverse, reg1, reg2):
    Xtemp = X + tau*(I_op(Y-F_op(X))-inverse(Z))
    Xnew = reg2(Xtemp)
    Ztemp = Z + mu*transform(2*Xnew-X)
    Znew = Ztemp-reg1(Ztemp)
    return Xnew, Znew

def SDR(X, Y):
    return 10*np.log10(np.sqrt(np.sum(X[X!=0]**2))/np.sqrt(np.sum((Y[X!=0]-X[X!=0])**2)))

def Res(X,Y,sigma):
    return np.sqrt(np.sum(((X-Y)/sigma)**2)/X.size)#np.std((X-Y)**2/sigma**2)

def FISTA(Y, alphaX, F_op, I_op, mu, ts, csi, reg, transform, inverse, pos = 1, mask = 1, original_fista=False):
    if not original_fista: 
        S = inverse(alphaX)
    else:
        S = inverse(csi)  # test : back to original FISTA

    R = mu*I_op(Y-F_op(S)*mask)
    alpha = transform(R)+csi
    alpha = reg(alpha)
    tsnew = (1.+np.sqrt(1.+4.*ts**2))/2.
    csi = alpha+((ts-1)/tsnew)*(alpha-alphaX)
    return alpha, csi, tsnew

def Soft(X, level, k, supp =1, Kill = 0):
    Xnew = np.sign(X)*(np.abs(X)-level*(k))
    Xnew[np.where((np.abs(X)-level*(k))<0)] = 0
    Xnew[0,:,:] = np.sign(X[0,:,:]) * (np.abs(X[0,:,:]) - level[0,:,:] * (k+1))
    Xnew[0,np.where((np.abs(X[0,:,:]) - level[0,:,:] * (k+1)) < 0)] = 0

    if Kill == 1:
        Xnew[-1,:,:] = 0
    else:
        Xnew[-1, :, :] = X[-1,:,:]

    #print(Xnew.shape, supp.shape)
    Xnew = Xnew*supp

    return Xnew


def level(n1, n2, lvl):
    ##DESCRIPTION:
    ##    Estimates the noise levels in starlet space in image plane.
    ##
    ##INPUTS:
    ##  -n1,n2: shape of the image for which to get noise levels
    ##
    ##OUTPUTS:
    ##  -levels: units of noise levels at each scale and location of a starlet transform
    dirac = np.zeros((n1, n2))
    #   lvl = np.int(np.log2(n1))

    dirac[int(n1 / 2), int(n2 / 2)] = 1

    wave_dirac, _ = wave_transform(dirac, lvl, newwave=0)

    wave_sum = np.sqrt(np.sum(np.sum(wave_dirac ** 2, 1), 1))

    levels = np.multiply(np.ones((lvl, n1, n2)).T, wave_sum).T

    return levels

def Soft_Threshold(X, transform, inverse, level, k, supp =1, Kill = 0):
    X = transform(X)
    alpha, _ = wave_transform(X,Xw.shape[0],newwave = 0)
    M = np.zeros(alpha.shape)
    M[np.abs(alpha)-level*k>0] = 1
    M[0,:,:] = 0
#    M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1

    Xnew = np.sign(X)*(np.abs(X)-level*k)
    Xnew = Xnew*M
    if Kill ==1:
        Xnew[-1, :, :] = 0
    else:
        Xnew[-1,:,:] = X[-1,:,:]
    Xnew = Xnew*supp
    return inverse(Xnew)

def Hard(X, level, k, supp=1):
    Xnew = np.copy(X)
    Xnew[np.where((np.abs(X)-level*k)<0)] = 0
    
    Xnew[-1,:,:] = X[-1,:,:]
    Xnew = Xnew*supp
##    plt.figure(0)
##    plot_cube(X)
##    plt.figure(1)
##    plot_cube(Xnew)
##    plt.show()
    return Xnew

def Hard_Threshold(X, transform, inverse, level, k, supp=1, M = [0]):
    Xw = transform(X)
    if np.sum(M) == 0:
        alpha, _ = wave_transform(X,Xw.shape[0],newwave = 0)
        M = np.zeros(alpha.shape)
        M[(np.abs(alpha)-level*k)>0] = 1
        M[0,:,:] = 0
        M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1

    Xnew=M*Xw
    Xnew[-1,:,:] = Xw[-1,:,:]
    Xnew = Xnew*supp
    return inverse(Xnew), M


def mr_filter(Y, level, k, niter, transform, inverse, sigma, lvl = 6, Soft = 0, pos = 1, supp = 1):
    Xnew = 0
    alpha,  _ = wave_transform(Y, lvl, newwave=0)
    M = np.zeros(alpha.shape)
    M[np.abs(alpha)-level*k>0] = 1
    M[0,:,:] = 0
 #   M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1
    M[-1,:,:] =1
    i=0

    while i < niter:
        R = Y-Xnew
        if np.std(R/sigma)<1.1:
            print('limit: ', i)
            break
     #   if Soft == True :
     #       Rnew= Soft_threshold(R, transform, inverse, level,k)
     #   else:

     #       Rnew, m0 = Hard_Threshold(R, transform, inverse, level,k)
        Rnew = inverse(transform(R)*M*supp)
        Xnew = Xnew+Rnew

        if pos == True:
            Xnew[Xnew < 0] = 0

        i = i+1

    return (Xnew), M

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

def Downsample(image, factor=1):
    """
    resizes image with nx x ny to nx/factor x ny/factor
    :param image: 2d image with shape (nx,ny)
    :param factor: integer >=1
    :return:
    """
    if factor < 1:
        raise ValueError('scaling factor in re-sizing %s < 1' %factor)
    f = int(factor)
    nx, ny = np.shape(image)
    if int(nx/f) == nx/f and int(ny/f) == ny/f:
        small = image.reshape([int(nx/f), f, int(ny/f), f]).mean(3).mean(1)
        return small
    else:
        raise ValueError("scaling with factor %s is not possible with grid size %s, %s" %(f, nx, ny))

def Upsample(image, factor):
    factor = int(factor)
    n1,n2 = image.shape
    upimage = np.zeros((n1*factor, n2*factor))
    x,y = np.where(upimage==0)
    upimage[x,y] = image[(x/factor),(y/factor)]/factor**2
    return upimage

