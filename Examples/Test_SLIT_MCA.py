from SLIT import Lens
import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from scipy import signal as scp
import SLIT as slit
import time
from scipy import signal as scp
import warnings
warnings.simplefilter("ignore")

#Example of a run of the SLIT_MCA algorithm on simulated images. 
#Here the first part of the file shows how simulations are generated.
#For users intereseted only in seeing the code run, have a look at the running SLIT_MCA section.
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
Fkappa = slit.Lens.F(kappa, nt1,nt2, size,nt1/2.,nt2/2.)

#Lens galaxy light profile
gal0 = pf.open('../Files/Galaxy.fits')[0].data

#Generation of lensed source
I2 = slit.Lens.source_to_image(newsource, nt1 ,nt2 , Fkappa)
HI2 = scp.fftconvolve(I2, PSF.astype(float), mode = 'same')

#Noise levels
SNR = 500
sigma = np.sqrt(np.sum(I2**2)/SNR/(nt1*nt2*size**2))

#Convolution of the observed image
simu = scp.fftconvolve(gal0.astype(float)+I2, PSF.astype(float), mode = 'same')
#Sotring the convolved lens light profile:
gal = scp.fftconvolve(gal0.astype(float), PSF.astype(float), mode = 'same')

#Final simulated image
Image = simu+np.random.randn(nt1,nt2)*sigma

################################Running SLIT############################
#Parameters
kmax = 5
niter =100
riter =50
levels = [0]

#Comment the following to have the level estimation routine run (takes more time)
levels = pf.open('../Files/Noise_levels_SLIT_MCA.fits')[0].data

#Start clock
start = time.clock()

#Running SLIT_MCA
S, FS, G = slit.SLIT_MCA(Image, Fkappa, kmax, niter,riter, size,PSF, PSFconj, levels = levels)

#Stop clock
elapsed = (time.clock()-start)
print('execution time:', elapsed, 'seconds')


real_source = newsource

source_error = np.sum(np.abs(real_source[np.where(real_source!=0)]
                                           -S[np.where(real_source!=0)])**2
                                           /real_source[np.where(real_source!=0)]**2)/(np.size(
                                           np.where(real_source!=0))/2.)

image_chi2 = np.std(Image-FS-G)**2/sigma**2
print('Residuals in source space', source_error)
print('Residuals in image space',image_chi2)

for i in [1]:
    ###Source

    plt.figure(0)
    plt.title('Source from SLIT')
    plt.imshow((S), vmin = np.min(real_source), vmax = np.max(real_source), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(1)
    plt.title('Original image of the source')
    plt.imshow(real_source, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(2)
    plt.title('relative difference')
    diff = (real_source-S)
    plt.imshow((np.abs(diff)), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ####Lensed source
    plt.figure(3)
    plt.title('Original lensed galaxy')
    plt.imshow(HI2, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(4)
    plt.title('reconstructed lensed source')
    plt.imshow((FS), vmin = np.min(I2), vmax = np.max(I2), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(5)
    plt.title('error on the source in image plane')
    plt.imshow((HI2-FS), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ###Galaxy
    plt.figure(6)
    plt.title('Original galaxy')
    plt.imshow((gal0), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(12)
    plt.title('Estimated Galaxy')
    plt.imshow((G), vmin = np.min(gal0), vmax = np.max(gal0), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(7)
    plt.title('Error on the galaxy')
    plt.imshow((gal-G), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ###Image
    plt.figure(8)
    plt.title('Image')
    plt.imshow(Image, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(9)
    plt.title('Reconstructed image')
    plt.imshow(FS+G, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()

    plt.figure(10)
    plt.title('difference with reconstructed image')
    plt.imshow(Image-FS-G,cmap = cm.gist_stern, interpolation = 'nearest', vmin = -5*sigma, vmax = 5*sigma)#slit.fft_convolve(Im,PSF)
    plt.axis('off')
    plt.colorbar()


plt.show()
