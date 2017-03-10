
import Lens_better as Lens
import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from scipy import signal as scp
import gaussian as gs
import SLIT as slit
import time
from scipy import signal as scp
import warnings
warnings.simplefilter("ignore")


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

 
real_source = pf.open('/Users/remy/Desktop/These/MuSCADeT/HFF/Abell2744/Colour_images_recombined.fits')[0].data#pf.open('imp_lens.fits')[0].data
##N1,N2 are the numbers of pixels in the image plane.
nt1= 100
nt2 = 100

na1 = nt1
na2 = nt2
## Lens mass distribution.
b = 1.53/0.05
xc = 0.95
q = 0.71
betata = 2.1
thetata = 25.2

gal0 = real_source[0,2871-nt1/2:2871+nt1/2,1878-nt2/2:1878+nt2/2]#gs.sersic(nt1,nt2,nt1/2.,nt2/2.,2,0.08,0.1,0.2,1)

gal2 = gs.gaussian(nt1,nt2,nt1/2.+3,3-nt2/2.,4,0.05,0.12,0)



#gal0[-3:,-3:] = 0
#gal2[-3:,-3:]=0
extra = 100
kappa = SIE(na1/2.+extra/2.,na2/2.+extra/2.,na1+extra,na2+extra,b,betata,q,xc, thetata)
#kappa2 = SIE(na1/2.+extra/2.,na2/2.+extra/2.,na1+extra,na2+extra,b+0.1,betata,q,xc, thetata)
#kappa = barkana(na1/2.,na2/2.,na1,na2,b,betata,q,thetata)

hdus = pf.PrimaryHDU(kappa)
lists = pf.HDUList([hdus])
lists.writeto('kappa.fits', clobber=True)

x00 =3743+1#4843-35
y00 =2030-2#6420+15
size = 1
newsource =real_source[0,x00-nt1*size/2.:x00+nt1*size/2.,y00-nt2*size/2.:y00+nt2*size/2.]
s = wine.MCA.MAD(newsource)
newsource,c = wine.MCA.mr_filter(newsource, 20,5, s, lvl = 5)
gal0,c = wine.MCA.mr_filter(gal0, 20,7, s, lvl = 5)
gal0 = (gal0-np.min(gal0))*4
newsource = newsource-np.min(newsource)



plt.imshow(gal0, cmap = cm.gist_stern, interpolation = 'nearest')
plt.axis('off')
plt.title('Lens galaxy')
plt.colorbar(); plt.show()

start = time.clock()
theta = Lens.F(kappa, nt1,nt2, size,na1/2.,na2/2.)
#theta2 = Lens.F(kappa2, nt1,nt2, size,na1/2.,na2/2.)
elapsed = (time.clock()-start)


npsf1 = 65.
npsf2 = 65.
PSF0 = pf.open('PSF.fits')[0].data#('HST_1430/psf1.fits')[0].data##
PSF = PSF0[1:,1:]
PSFconj = np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF0[:-1,:-1]))))
PSFconj=PSFconj/np.sum(PSFconj)
PSF = PSF/np.sum(PSF)


##PSF0 = gs.gaussian(npsf1,npsf2,33.,33.,1.,0.95,0.95,0)
##PSF = gs.gaussian(npsf1,npsf2,32.,32.,1.,0.95,0.95,0)
##PSF[npsf2-1,npsf1-2] = 0
##PSF0[npsf2-1,npsf1-2] = 0


xt,yt = np.where(np.zeros((nt1,nt2))==0)
I2 = Lens.source_to_image(newsource, nt1 ,nt2 , theta)
PSF0 = np.abs(PSF0/np.sum(PSF0))
PSF = np.abs(PSF/np.sum(PSF))

##PSFconj = np.real(np.fft.ifft2(np.conjugate(np.fft.fft2(PSF))))


I3 = np.copy(I2)
I2 = scp.fftconvolve(I2, PSF, mode = 'same')

SNR = 500
sigma = np.sqrt(np.sum(I2**2)/SNR/(nt1*nt2*size**2))

gal = np.copy(gal0)
gal = scp.fftconvolve(gal, PSF, mode = 'same')

Image = I2+gal+np.random.randn(nt1,nt2)*sigma

plt.imshow(Image, cmap = cm.gist_stern, interpolation = 'nearest')
plt.title('Lens system')
plt.axis('off')
plt.colorbar()
plt.show()

plt.imshow(newsource, cmap = cm.gist_stern, interpolation = 'nearest')
plt.title('Background source')
plt.axis('off')
plt.colorbar()
plt.show()

plt.imshow(kappa, cmap = cm.gist_stern, interpolation = 'nearest')
plt.title('surface mass density')
plt.axis('off')
plt.colorbar()
plt.show()


kmax = 5
niter =500
kinter = 6
ninter = 1
pos_real = 1
pos_wave = 0
weight = 1
#PSF=[0,0]
riter =200

FB = 0
soft = 1
repeat = 1
PD = 0

start = time.clock()

sourcesl, Imsl, G = slit.SLIT_old(Image, theta, kmax, niter,riter, kinter, ninter, size,
                    pos = pos_real, posit = pos_wave, decrease = 1, soft =soft, FB = FB, 
                    reweighting = weight, Ginit =0, PSF = PSF, PSFconj = PSFconj, repeat = repeat, mrfilter = PD)

elapsed = (time.clock()-start)

print('execution time:', elapsed, 'seconds')

source = sourcesl
Im=Imsl

real_source = newsource

source_error = np.sum(np.abs(real_source[np.where(real_source!=0)]
                                           -sourcesl[np.where(real_source!=0)])**2
                                           /real_source[np.where(real_source!=0)]**2)/(np.size(
                                           np.where(real_source!=0))/2.)

GSF = np.copy(G)


image_chi2 = np.std(Image-Imsl-G)**2/sigma**2
print('Residuals in source space', source_error)
print('Residuals in image space',image_chi2)

hdus = pf.PrimaryHDU(sourcesl)
lists = pf.HDUList([hdus])
lists.writeto('Souce.fits', clobber=True)

hdus = pf.PrimaryHDU(G)
lists = pf.HDUList([hdus])
lists.writeto('Galaxy_SLIT.fits', clobber=True)


hdus = pf.PrimaryHDU(real_source-source)
lists = pf.HDUList([hdus])
lists.writeto('Diff.fits', clobber=True)

for i in [1]:
    ###Source
 #   plt.suptitle('FISTA: error per pixel on the source: '+str(source_error)+' image chi2:'+str(image_chi2))
#    plt.subplot(4,3,2)
    plt.figure(0)
    plt.title('Source from SLIT')
    plt.imshow((sourcesl), vmin = np.min(real_source), vmax = np.max(real_source), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,1)
    plt.figure(1)
    plt.title('Original image of the source')
    plt.imshow(real_source, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,3)
    plt.figure(2)
    plt.title('relative difference')
    diff = (real_source-sourcesl)#/real_source
 #   diff[np.where(real_source==0)] = 0
#    diff[np.where(diff>1)]= np.log(0.)
    plt.imshow((np.abs(diff)), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ####Lensed source
 #   plt.subplot(4,3,4)
    plt.figure(3)
    plt.title('Original lensed galaxy')
    plt.imshow(I2, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,5)
    plt.figure(4)
    plt.title('reconstructed lensed source')
    plt.imshow((Imsl), vmin = np.min(I2), vmax = np.max(I2), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,6)
    plt.figure(5)
    plt.title('error on the source in image plane')
    plt.imshow((I2-Imsl), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ###Galaxy
 #   plt.subplot(4,3,7)
    plt.figure(6)
    plt.title('Original galaxy')
    plt.imshow((gal0), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,8)
    plt.figure(12)
    plt.title('Estimated Galaxy')
    plt.imshow((G), vmin = np.min(gal0), vmax = np.max(gal0), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,9)
    plt.figure(7)
    plt.title('Error on the galaxy')
    plt.imshow((gal-G), cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
    ###Image
 #   plt.subplot(4,3,10)
    plt.figure(8)
    plt.title('Image')
    plt.imshow(Image, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,11)
    plt.figure(9)
    plt.title('Reconstructed image')
    plt.imshow(Imsl+GSF, cmap = cm.gist_stern, interpolation = 'nearest')
    plt.axis('off')
    plt.colorbar()
 #   plt.subplot(4,3,12)
    plt.figure(10)
    plt.title('difference with reconstructed image')
    plt.imshow(Image-Imsl-GSF,cmap = cm.gist_stern, interpolation = 'nearest', vmin = -5*sigma, vmax = 5*sigma)#slit.fft_convolve(Im,PSF)
    plt.axis('off')
    plt.colorbar()


plt.show()
