import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm

import SLIT 
import time
from scipy import signal as scp
import warnings
warnings.simplefilter("ignore")

#Example of a run of the SLIT algorithm on simulated images. 
#Here the first part of the file shows how simulations are generated.
#For users intereseted only in seeing the code run, have a look at the running SLIT section.
#The command line that builds the Fkappa operator is also of outmost importance.


###############################Simulation###############################
def SIE(x0,y0,n1,n2,b,beta,q,xc,theta):
    kappa = np.zeros((n1,n2))
    x,y = np.where(kappa == 0)
    eps = (1-q**2)/(1+q**2)
    up = b**(beta-1)
    pre = up/(2*(1-eps)**((beta-1)/2))
    count = 0
    theta = theta*np.pi/180.
    for i in x:
        Xr = (x[count]-x0)*np.cos(theta)-(y[count]-y0)*np.sin(theta)
        Yr = (x[count]-x0)*np.sin(theta)+(y[count]-y0)*np.cos(theta)
        kappa[x[count],y[count]] = pre/((xc**2.)/(1.-eps)+(Xr)**2.+((Yr)**2.)/q**2.)**((beta-1.)/2.)

        count += 1
    return kappa

#Source light profile
newsource = pf.open('../Files/source.fits')[0].data
##N1,N2 are the numbers of pixels in the image plane.
nt1= 100
nt2 = 100
#Size ratio of the source to image number of pixels 
size = 1

#PSF
PSF0 = pf.open('../Files/PSF.fits')[0].data
PSF = PSF0[1:,1:]
PSFconj = np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
PSFconj=PSFconj/np.sum(PSFconj)
PSF = PSF/np.sum(PSF)
  
## Lens mass distribution.
b = 1.53/0.05
xc = 0.95
q = 0.71
betata = 2.1
thetata = 25.2
kappa = SIE(nt1/2.+50,nt2/2.+50,nt1+100,nt2+100,b,betata,q,xc, thetata)
#Mapping between lens and source IMPORTANT
Fkappa = SLIT.Lens.F(kappa, nt1,nt2, size,nt1/2.,nt2/2.)


#Generation of lensed source
I2 = SLIT.Lens.source_to_image(newsource, nt1 ,nt2 , Fkappa)

#Noise levels
SNR = 500
sigma = np.sqrt(np.sum(I2**2)/SNR/(nt1*nt2*size**2))

#Convolution by the PSF and generation of the final image
I2 = scp.fftconvolve(I2, PSF, mode = 'same')

#Final simulated image
Image = I2+np.random.randn(nt1,nt2)*sigma


################################Running SLIT############################
#Parameters
kmax = 5
niter =100
levels = [0]

#Comment the following to have the level estimation routine run (takes more time)
levels = pf.open('../Files/Noise_levels_SLIT.fits')[0].data

#Start clock
start = time.clock()

#Running SLIT
sourcesl, Imsl = SLIT.SLIT(Image, Fkappa, kmax, niter, size, PSF, PSFconj, levels = levels, fb =1)

#Stop clock
elapsed = (time.clock()-start)
print('execution time:', elapsed, 'seconds')

#Reconstruction goodness
real_source = newsource
source_error = np.sum(np.abs(real_source[np.where(real_source!=0)]
                                           -sourcesl[np.where(real_source!=0)])**2
                                           /real_source[np.where(real_source!=0)]**2)/(np.size(
                                           np.where(real_source!=0))/2.)
image_chi2 = np.std(Image-Imsl)**2/sigma**2
print('Residuals in source space', source_error)
print('Residuals in image space',image_chi2)

#Display of results
for i in [1]:
    plt.figure(2)
 #   plt.suptitle('FISTA: error per pixel on the source: '+str(source_error)+' image chi2:'+str(image_chi2))
 #   plt.subplot(2,3,1)
    plt.title('Source from SLIT')
    plt.imshow((sourcesl), vmin = np.min(real_source), vmax = np.max(real_source),cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
#    plt.subplot(2,3,2)
    plt.figure(3)
    plt.title('Original source')
    plt.imshow(real_source, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,3)
    plt.figure(4)
    plt.title('Lensed source')
    plt.imshow(Image, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    plt.figure(41)
    plt.title('Reconstructed lensed source')
    plt.imshow(Imsl, vmin = np.min(Image), vmax = np.max(Image), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,4)
    plt.figure(5)
    plt.title('relative difference')
    diff = (real_source-sourcesl)/real_source
    diff[np.where(real_source==0)] = 0
    diff[np.where(diff>1)]= np.log(0.)
    plt.imshow(np.abs(diff), vmax = 1., vmin = 0., cmap = cm.gist_stern, interpolation = 'nearest' )
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,5)
    plt.figure(6)
    plt.title('difference with reconstructed lensed image')
    plt.imshow(Image-Imsl, vmin = -5*sigma, vmax = 5*sigma, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,6)
    plt.figure(7)
    plt.title('difference with true source')
    plt.imshow((np.abs(real_source-sourcesl)), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
plt.show()
