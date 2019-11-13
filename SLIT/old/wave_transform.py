import numpy as np
import scipy.signal as cp
import matplotlib.pyplot as plt
import scipy.ndimage.filters as sc


def symmetrise(img, size):

    n3, n4 = np.shape(img)
    n1,n2 = size
    img[:(n3-n1)/2, :] = np.flipud(img[(n3-n1)/2:(n3-n1),:])
    img[:,:(n4-n2)/2] = np.fliplr(img[:,(n4-n2)/2:(n4-n2)])
    img[(n3+n1)/2:,:] = np.flipud(img[n1:(n3+n1)/2,:])
    img[:,(n4+n2)/2:] = np.fliplr(img[:,n2:(n4+n2)/2])

    return img


def fft_convolve(X,Y, inv = 0):
    
    XF = np.fft.rfft2(X)
    YF = np.fft.rfft2(Y)
#    YF0 = np.copy(YF)
#    YF.imag = 0
#    XF.imag = 0
    if inv == 1:
 #       plt.imshow(np.real(YF)); plt.colorbar(); plt.show()
        YF = np.conj(YF)

    SF = XF*YF
    
    S = np.fft.irfft2(SF)
    n1,n2 = np.shape(S)

    S = np.roll(S,-n1/2+1,axis = 0)
    S = np.roll(S,-n2/2+1,axis = 1)

    return np.real(S)

    
def wave_transform(img, lvl, Filter = 'Bspline', newwave = 1, convol2d = 0):

    mode = 'nearest'
    
    lvl = lvl-1
    sh = np.shape(img)
    if np.size(sh) ==3:
        mn = np.min(sh)
        wave = np.zeros([lvl+1,sh[1], sh[1],mn])
        for h in range(mn):
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
        lvl = int(np.log2((n1-1)/(n-1.))+1)

    c = img
    ## wavelet set of coefficients.
    wave = np.zeros([lvl+1,n1,n2])
  
    for i in range(lvl):
        newh = np.zeros((1,n+(n-1)*(2**i-1)))
        newh[0, np.linspace(0,np.size(newh)-1,len(h), dytpe=int)] = h
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
    
    
    for i in range(1, lvl):
        
        newh = np.zeros((1,n+(n-1)*(2**(lvl-1-i)-1)))
        newh[0, np.linspace(0, np.size(newh)-1,len(h), dtype=int)] = h
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
    
