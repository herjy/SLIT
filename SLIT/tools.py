import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pf
from scipy import signal as scp
import scipy.ndimage.filters as sc
import scipy.ndimage.filters as med
import scipy.signal as cp


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
    x = wave_transform(x, np.int(np.log2(x.shape[0])))[0,:,:]
    meda = med.median_filter(x,size = (n,n))
    medfil = np.abs(x-meda)#np.median(x))
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def MAD_box(x, tau):
    n1,n2 = x.shape
    xw = wave_transform(x,2)[0,:,:]
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
        return wave_transform(i,lvl)
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
            lvlso = (scp.fftconvolve(sigma[i, :, :] ** 2, wave_dirac[i, :, :] ** 2,
                                     mode='same'))
        else:
            lvlso = scp.fftconvolve(sigma ** 2, wave_dirac[i,:,:] ** 2,
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


def FISTA(Y, alphaX, F_op, I_op, mu, ts, csi, reg, transform, inverse, pos = 1, mask = 1):
    S = inverse(alphaX)

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
    dirac[n1 / 2, n2 / 2] = 1
    wave_dirac = wave_transform(dirac, lvl, newwave=0)

    wave_sum = np.sqrt(np.sum(np.sum(wave_dirac ** 2, 1), 1))

    levels = np.multiply(np.ones((lvl, n1, n2)).T, wave_sum).T

    return levels

def Soft_Threshold(X, transform, inverse, level, k, supp =1, Kill = 0):
    X = transform(X)
    alpha = wave_transform(X,Xw.shape[0],newwave = 0)
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
        alpha = wave_transform(X,Xw.shape[0],newwave = 0)
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
    alpha = wave_transform(Y, lvl, newwave=0)
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

def wave_transform(img, lvl, Filter = 'Bspline', newwave = 1, convol2d = 0):

    mode = 'nearest'
    
    lvl = lvl-1
    sh = np.shape(img)
    if np.size(sh) ==3:
        mn = np.min(sh)
        wave = np.zeros([lvl+1,sh[1], sh[1],mn])
        for h in np.linspace(0,mn-1, mn):
            if mn == sh[0]:
                wave[:,:,:,h] = wave_transform(img[h,:,:],lvl+1, Filter = Filter)
            else:
                wave[:,:,:,h] = wave_transform(img[:,:,h],lvl+1, Filter = Filter)
        return wave

    n1 = sh[1]
    n2 = sh[1]
    
    if Filter == 'Bspline':
        h = [1./16, 1./4, 3./8, 1./4, 1./16]
    else:
        h = [1./4,1./2,1./4]
    n = np.size(h)
    h = np.array(h)
    
    lvl = np.min([lvl,np.int(np.log2(n2))])

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1,n2])
  
    for i in np.linspace(0,lvl-1,lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ######Calculates c(j+1)
        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(c, H, mode='same', boundary='symm')
        else:
            cnew = sc.convolve1d(c,newh[0,:],axis = 0, mode =mode)

            ###### Column convolution
            cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode =mode)

 
      
        if newwave ==1:
            ###### hoh for g; Column convolution
            if convol2d == 1:
                hc = cp.convolve2d(cnew, H, mode='same', boundary='symm')
            else:
                hc = sc.convolve1d(cnew,newh[0,:],axis = 0, mode = mode)
 
                ###### hoh for g; Line convolution
                hc = sc.convolve1d(hc,newh[0,:],axis = 1, mode = mode)
            
            ###### wj+1 = cj-hcj+1
            wave[i,:,:] = c-hc
            
        else:
            ###### wj+1 = cj-cj+1
            wave[i,:,:] = c-cnew


        c = cnew
     
    wave[i+1,:,:] = c

    return wave

def iuwt(wave, convol2d =0):
    mode = 'nearest'
    
    lvl,n1,n2 = np.shape(wave)
    h = np.array([1./16, 1./4, 3./8, 1./4, 1./16])
    n = np.size(h)

    cJ = np.copy(wave[lvl-1,:,:])
    
    
    for i in np.linspace(1,lvl-1,lvl-1):
        
        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0,np.int_(np.linspace(0,np.size(newh)-1,len(h)))] = h
        H = np.dot(newh.T,newh)

        ###### Line convolution
        if convol2d == 1:
            cnew = cp.convolve2d(cJ, H, mode='same', boundary='symm')
        else:
          cnew = sc.convolve1d(cJ,newh[0,:],axis = 0, mode = mode)
            ###### Column convolution
          cnew = sc.convolve1d(cnew,newh[0,:],axis = 1, mode = mode)

        cJ = cnew+wave[lvl-1-i,:,:]

    return np.reshape(cJ,(n1,n2))


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