import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import scipy.ndimage.filters as sc
import scipy.ndimage.filters as med
import scipy.signal as cp

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
    medfil = np.abs(x-meda)
    sh = np.shape(x)
    sigma = 1.48*np.median((medfil))
    return sigma

def Forward_Backward(Y, X, F_op, I_op, mu, reg, pos = 1):
    
    R = mu*I_op(Y-F_op(X))
    Xnew = np.copy(X+R)
    Xnew, M = reg(Xnew)
    if pos == 1:
        Xnew[Xnew<0] = 0
    return Xnew, M

def Primal_dual(Y, X, U, mu, nu, tau, F_op, I_op, transform, inverse, reg):
    p = X+mu*I_op(Y-F_op(X))-mu*inverse(U)
 #   plot_cube(U+nu*transform(2*p-X)); plt.show()
    q = reg(U+nu*transform(2*p-X))
    X =tau*p+(1-tau)*X
    U =tau*q+(1-tau)*U
#    plot_cube(q); plt.show()
    return X,U


def FISTA(Y, alphaX, F_op, I_op, mu, ts, csi, reg, transform, inverse, pos = 1, mask = 1):
    S = inverse(alphaX)
    #S[S>size*np.max(Y)] = np.max(Y)
    #  S[S<0] = 0
    R = mu*I_op(Y-F_op(S)*mask)
    alpha = transform(R)+csi
    alpha, M = reg(alpha)
    tsnew = (1.+np.sqrt(1.+4.*ts**2))/2.
    csi = alpha+((ts-1)/tsnew)*(alpha-alphaX)
    return alpha, csi, tsnew

def Soft(X, transform, inverse, level, k, supp =1):
    Xnew = np.sign(X)*(np.abs(X)-level*k)
    Xnew[np.where((np.abs(X)-level*k)<0)] = 0
    
    Xnew[-1,:,:] = X[-1,:,:]
    
    #print(Xnew.shape, supp.shape)
    Xnew = Xnew*supp
    return Xnew

def Soft_Threshold(X, transform, inverse, level, k, supp =1):
    X = transform(X)
    alpha = wave_transform(X,Xw.shape[0],newwave = 0)
    M = np.zeros(alpha.shape)
    M[np.abs(alpha)-level*k>0] = 1
    M[0,:,:] = 0
    M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1

    Xnew = np.sign(X)*(np.abs(X)-level*k)
    Xnew = Xnew*M
    Xnew[-1,:,:] = X[-1,:,:]
    Xnew = Xnew*supp
    return inverse(Xnew), M

def Hard(X, transform, inverse, level, k, supp=1):
    Xnew = np.copy(X)
    Xnew[np.where((np.abs(X)-level*k)<0)] = 0
    
    Xnew[-1,:,:] = X[-1,:,:]
    Xnew = Xnew*supp
##    plt.figure(0)
##    plot_cube(X)
##    plt.figure(1)
##    plot_cube(Xnew)
##    plt.show()
    return Xnew, M

def Hard_Threshold(X, transform, inverse, level, k, supp=1):
    Xw = transform(X)
    alpha = wave_transform(X,Xw.shape[0],newwave = 0)
    M = np.zeros(alpha.shape)
    M[np.abs(alpha)-level*k>0] = 1
    M[0,:,:] = 0
    M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1

    Xnew=M*Xw
    Xnew[-1,:,:] = Xw[-1,:,:]
    Xnew = Xnew*supp
    return inverse(Xnew), M

def mr_filter(Y, level, k, niter, transform, inverse, lvl = 6, Soft = 0, pos = 1):
    Xnew = 0
    alpha = wave_transform(Y, lvl, newwave=0)
    M = np.zeros(alpha.shape)
    M[np.abs(alpha)-level*k>0] = 1
    M[0,:,:] = 0
    M[0,np.abs(alpha[0,:,:]) - level[0,:,:] * (k+1) > 0] = 1
    M[-1,:,:] =1
    i=0
    while i < niter:
        R = Y-Xnew

        if Soft == True :
            Rnew= Soft_threshold(R, transform, inverse, level,k)
        else:

            Rnew = Hard_Threshold(R, transform, inverse, level,k)
   #     Rnew = inverse(transform(R)*M)
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
    
    if n+2**(lvl-1)*(n-1) >= np.min([n1,n2])/2.:
        lvl = np.int_(np.log2((n1-1)/(n-1.))+1)

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
