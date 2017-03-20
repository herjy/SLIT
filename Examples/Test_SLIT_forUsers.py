import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from scipy import signal as scp
import SLIT 
import time
from scipy import signal as scp
import warnings
warnings.simplefilter("ignore")

#Example of a run of the SLIT algorithm on simulated images. 
#Here the first part of the file shows how simulations are generated.
#For users intereseted only in seeing the code run, have a look at the running SLIT section.
#The command line that builds the Fkappa operator is also of outmost importance.

Image = '''Input 2D image of the lens to invert '''
nt1,nt2  = np.shape(Image)
###############################Mass profile###############################
x0,y0 = '''Input the center of mass of the lens with regard to coodinates in Image '''
kappa = '''Input dimensionless pixelated mass density profile here '''
size = '''Input the desired size of the output with regard to Image. Chosing 1 will result in a source with the same number of pixels as in Image. '''
#Mapping between lens and source IMPORTANT
Fkappa = SLIT.Lens.F(kappa, nt1,nt2, size,x0,y0)

PSF = '''Input the PSF for Image here'''
PSFconj = '''Input the conjugate of the PSF here given by
                    np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1])))), but be carefull that the result is still centered '''

################################Running SLIT############################
#Parameters
kmax = 5
niter =50

#Start clock
start = time.clock()

#Running SLIT
S, FS = SLIT.SLIT(Image, Fkappa, kmax, niter, size, PSF, PSFconj)

#Stop clock
elapsed = (time.clock()-start)
print('execution time:', elapsed, 'seconds')

#Reconstruction goodness
real_source = newsource
source_error = np.sum(np.abs(real_source[np.where(real_source!=0)]
                                           -S[np.where(real_source!=0)])**2
                                           /real_source[np.where(real_source!=0)]**2)/(np.size(
                                           np.where(real_source!=0))/2.)
image_chi2 = np.std(Image-FS)**2/sigma**2
print('Residuals in source space', source_error)
print('Residuals in image space',image_chi2)

#Display of results
for i in [1]:
    plt.figure(2)
 #   plt.suptitle('FISTA: error per pixel on the source: '+str(source_error)+' image chi2:'+str(image_chi2))
 #   plt.subplot(2,3,1)
    plt.title('Source from SLIT')
    plt.imshow((S), vmin = np.min(real_source), vmax = np.max(real_source),cmap = cm.gist_stern, interpolation = 'nearest')
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
    plt.imshow(FS, vmin = np.min(Image), vmax = np.max(Image), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,4)
    plt.figure(5)
    plt.title('relative difference')
    diff = (real_source-S)/real_source
    diff[np.where(real_source==0)] = 0
    diff[np.where(diff>1)]= np.log(0.)
    plt.imshow(np.abs(diff), vmax = 1., vmin = 0., cmap = cm.gist_stern, interpolation = 'nearest' )
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,5)
    plt.figure(6)
    plt.title('difference with reconstructed lensed image')
    plt.imshow(Image-FS, vmin = -5*sigma, vmax = 5*sigma, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(2,3,6)
    plt.figure(7)
    plt.title('difference with true source')
    plt.imshow((np.abs(real_source-S)), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
plt.show()
