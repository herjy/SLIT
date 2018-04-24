import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
import SLIT
import gaussian as gs
import time
from scipy import signal as scp
import warnings
warnings.simplefilter("ignore")

#Example of a run of the SLIT algorithm on simulated images. 
#Here the first part of the file shows how simulations are generated.
#For users intereseted only in seeing the code run, have a look at the running SLIT section.
#The command line that builds the Fkappa operator is also of outmost importance.



#Source light profile
newsource = pf.open('IMG2.fits')[0].data#('./Files/source.fits')[0].data
##N1,N2 are the numbers of pixels in the image plane.
nt1= 100
nt2 = 100
#Size ratio of the source to image number of pixels 


#PSF
PSF0 = gs.gaussian(64,64,32,32,1,2/(2.*(2*np.log(2))**0.5),2/(2.*(2*np.log(2))**0.5),0)
PSF0[-2:,-2:] = 0
PSF = PSF0[1:,1:]
PSF = PSF/np.sum(PSF)
  
## Lens mass distribution.
Re = 1.3/0.05
xc = 0.8
q = 0.99
gamma_true = 2.0
theta = 25.2


extra = 500
######## SIS parameters
xsie, ysie = 50.,50.
n1sie, n2sie = 100,100
Re = 15
sizeSIE = 2
nsim = 50



kmax = 5
niter =100
levels = [0]
lvl = 4

def  test_center(shift):
    kappa_true = SLIT.Lens.SIE(extra/2.+xsie,extra/2.+xsie,n1sie+extra,n1sie+extra,Re,betata,q,xc, thetata)
    #plt.imshow(kappa_true); plt.show()#
    #SLIT.Lens.SIS(xsis+extra/2,ysis+extra/2,n1sis+extra,n2sis+extra,Re)
    alphax_SIE_true, alphay_SIE_true = SLIT.Lens.alpha_def(kappa_true, n1sie,n2sie, extra)
    #Mapping between lens and source IMPORTANT
    Fkappa_true = SLIT.Lens.F(kappa_true, n1sie,n2sie, sizeSIE, extra = extra)

    hdus = pf.PrimaryHDU(alphax_SIE_true)
    lists = pf.HDUList([hdus])
    lists.writeto('../Results_center/alphax_SIE_true.fits', clobber=True)
    hdus = pf.PrimaryHDU(alphay_SIE_true)
    lists = pf.HDUList([hdus])
    lists.writeto('../Results_center/alphay_SIE_true.fits', clobber=True)

    #Generation of lensed source
    I2 = SLIT.Lens.source_to_image(newsource, n1sie ,n2sie , Fkappa_true)

    #Noise levels
    SNR = 100
    sigma = np.sqrt(np.sum(I2**2)/SNR/(nt1*nt2*sizeSIE**2))
    print('sigma: ',sigma)
    #Convolution by the PSF and generation of the final image
    I2 = scp.fftconvolve(I2, PSF, mode = 'same')


    print(shift)
    for i in range(nsim):
        theta = np.random.rand(1)[0]*np.pi
        x = shift*np.cos(theta)
        y = shift*np.sin(theta)
        kappa = SLIT.Lens.SIE(xsie+x+extra/2,ysie+y+extra/2,n1sie+extra,n2sie+extra,Re,betata,q,xc, thetata)
        alphax_SIE, alphay_SIE = SLIT.Lens.alpha_def(kappa, n1sie,n2sie, extra)
        #Mapping between lens and source IMPORTANT
        Fkappa = SLIT.Lens.F(kappa, nt1,nt2, sizeSIE, extra = extra)

    #Final simulated image
        Image = I2+np.random.randn(n1sie,n2sie)*sigma

        hdus = pf.PrimaryHDU(Image)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_center/Image_'+str(i)+'_'+str(shift)+'_'+str(theta)+'.fits', clobber=True)

        #Running SLIT
        S,FS = SLIT.Solve.SLIT(Image, Fkappa, kmax, niter, sizeSIE, PSF, 0,  scheme = 'FISTA', lvl = lvl)

        hdus = pf.PrimaryHDU(S)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_center/Source_'+str(i)+'_'+str(shift)+'_'+str(theta)+'.fits', clobber=True)

        hdus = pf.PrimaryHDU(FS)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_center/Lensed_source_'+str(i)+'_'+str(shift)+'_'+str(theta)+'.fits', clobber=True)

    return 0


def  test_slope(gamma):

    kappa_true = SLIT.Lens.Power_law(xsie+extra/2,ysie+extra/2,n1sie+extra,n2sie+extra,Re, theta, q, gamma_true,1.)
    alphax_SIE_true, alphay_SIE_true = SLIT.Lens.alpha_def(kappa_true, n1sie,n2sie, extra)
    #Mapping between lens and source IMPORTANT
    Fkappa_true = SLIT.Lens.F(kappa_true, n1sie,n2sie, sizeSIE, extra = extra)

    hdus = pf.PrimaryHDU(alphax_SIE_true)
    lists = pf.HDUList([hdus])
    lists.writeto('../Results_slope/alphax_PL_true.fits', clobber=True)
    hdus = pf.PrimaryHDU(alphay_SIE_true)
    lists = pf.HDUList([hdus])
    lists.writeto('../Results_slope/alphay_PL_true.fits', clobber=True)

    #Generation of lensed source
    I2 = SLIT.Lens.source_to_image(newsource, n1sie ,n2sie , Fkappa_true)

    #Noise levels
    SNR = 100
    sigma = np.sqrt(np.sum(I2**2)/SNR/(nt1*nt2*sizeSIE**2))
    print('sigma: ',sigma)
    #Convolution by the PSF and generation of the final image
    I2 = scp.fftconvolve(I2, PSF, mode = 'same')

    for i in range(nsim):

        kappa = SLIT.Lens.Power_law(xsie+extra/2,ysie+extra/2,n1sie+extra,n2sie+extra,Re, theta, q, gamma,1.)
        alphax_SIE, alphay_SIE = SLIT.Lens.alpha_def(kappa, n1sie,n2sie, extra)
        #Mapping between lens and source IMPORTANT
        Fkappa = SLIT.Lens.F(kappa, nt1,nt2, sizeSIE, extra = extra)

    #Final simulated image
        Image = I2+np.random.randn(n1sie,n2sie)*sigma

        hdus = pf.PrimaryHDU(Image)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_slope/Image_'+str(i)+'_'+str(gamma)+'.fits', clobber=True)

        #Running SLIT
        S,FS = SLIT.Solve.SLIT(Image, Fkappa, kmax, niter, sizeSIE, PSF, 0,  scheme = 'FISTA', lvl = lvl)

        hdus = pf.PrimaryHDU(S)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_slope/Source_'+str(i)+'_'+str(gamma)+'.fits', clobber=True)

        hdus = pf.PrimaryHDU(FS)
        lists = pf.HDUList([hdus])
        lists.writeto('../Results_slope/Lensed_source_'+str(i)+'_'+str(gamma)+'.fits', clobber=True)
    return 0

