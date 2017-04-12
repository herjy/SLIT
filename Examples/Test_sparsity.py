from SLIT import Lens
import pyfits as pf
import matplotlib.pyplot as plt 
import numpy as np
import matplotlib.cm as cm
from scipy import signal as scp
from SLIT import wave_transform as mw
import time
from scipy import signal as scp
import SLIT as slit
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import warnings
warnings.simplefilter("ignore")


 
S = pf.open('../Files/source.fits')[0].data
G = pf.open('../Files/Galaxy.fits')[0].data
##Sizes in image and source planes
nt1= 100
nt2 = 100
size = 1

#Mass profile of the lens
kappa = pf.open('../Files/kappa.fits')[0].data
Fkappa = Lens.F(kappa, nt1,nt2, size,nt1/2.,nt2/2.)
lensed = slit.lens_one(Fkappa, nt1,nt2, size)

#Levels for normalisation
lev = slit.level(nt1,nt1)

#Starlet transforms of the lens and source in their respective planes
wG = mw.wave_transform(G, lvl = 6)/lev
wS = mw.wave_transform(S, lvl = 6)/lev
#Lensed source
FS = Lens.source_to_image(S, nt1, nt2,Fkappa)
#Unlensed lens
FG = Lens.image_to_source(G, size, Fkappa, lensed=lensed)
#Starlet transform of the unlensed lens
wFG = mw.wave_transform(FG, 6)/lev
#Starlet transform of the lensed 
wFS = mw.wave_transform(FS, 6)/lev

def mk_sort(X):
    Y = np.sort(np.resize(np.abs(X), X.size))
    return Y[::-1]

#Function that computes the reconstruction error from the p% highest coefficients
def error_rec_from(X, p, wave = 0):
    Xcopy = np.copy(X)
    Y = mk_sort(X)

    ymin = Y[p*Y.size/1000.]
    Xcopy[np.abs(X)<ymin] = 0
    if wave == 1:
        err  = (mw.iuwt(Xcopy)-mw.iuwt(X))**2
    else:
        err  = ((Xcopy)-(X))**2
    error = np.sum(err)
    return error

#Computation of reconstruction errors for each light profile
error_wS = np.zeros(1000)
error_S = np.zeros(1000)
error_wFS = np.zeros(1000)
error_G = np.zeros(1000)
error_wG = np.zeros(1000)
error_wFG = np.zeros(1000)
for i in np.linspace(0,999, 1000):
    error_wS[i] = error_rec_from(wS, i, wave = 1)
    error_S[i] = error_rec_from(S, i)
    error_wFS[i] = error_rec_from(wFS, i, wave = 1)
    error_G[i] = error_rec_from(G, i)
    error_wG[i] = error_rec_from(wG, i, wave = 1)
    error_wFG[i] = error_rec_from(wFG, i, wave = 1)

print('NLA on the source at 10%: ',error_wS[100]/np.max(error_wS))
print('NLA on the lens at 10%: ', error_wG[100]/np.max(error_wG))
print('NLA on the lensed source at 10%: ', error_wFS[100]/np.max(error_wFS))
print('NLA on the delensed lens at 10%: ', error_wFG[100]/np.max(error_wFG))
#Display
plt.figure(1)
plt.plot(np.linspace(0,100, 1000), error_wS/np.max(error_wS), 'r', label = 'Source in starlet space', linewidth = 3)
plt.plot(np.linspace(0,100, 1000), error_wFG/np.max(error_wFG), 'c', label = 'Lens in source plane in starlet space', linewidth = 3)

plt.xlabel('percentage of coefficients used in reconstruction', fontsize=25)
plt.ylabel('Error on reconstruction', fontsize=25)
plt.title('Non-linear approximation error in source plane', fontsize=25)
plt.legend(fontsize = 25)
a = plt.axes([0.4, 0.2, 0.45, 0.4])
plt.semilogy(np.linspace(0,100, 1000), (error_wFG/np.max(error_wFG)), 'c', linewidth = 3)
plt.semilogy(np.linspace(0,100, 1000), error_wS/np.max(error_wS), 'r', linewidth = 3)
plt.xlim(20,100) 

plt.figure(2)
plt.plot(np.linspace(0,100, 1000), error_wG/np.max(error_wG), 'b', label = 'Galaxy in starlet space', linewidth = 3)
plt.plot(np.linspace(0,100, 1000), error_wFS/np.max(error_wFS), 'm', label = 'Lensed source in starlet space', linewidth = 3)

plt.xlabel('percentage of coefficients used in reconstruction', fontsize=25)
plt.ylabel('Error on reconstruction', fontsize=25)
plt.title('Non-linear approximation error in lens plane', fontsize=25)
plt.legend(fontsize = 25)
a = plt.axes([0.4, 0.2, 0.45, 0.4])
plt.semilogy(np.linspace(0,100, 1000), (error_wFS/np.max(error_wFS)), 'm', linewidth = 3)
plt.semilogy(np.linspace(0,100, 1000), error_wG/np.max(error_wG), 'b', linewidth = 3)
plt.xlim(20,100) 

plt.show()






