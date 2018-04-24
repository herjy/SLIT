import numpy as np
import matplotlib.pyplot as plt
import SLIT
import pyfits as pf
import matplotlib.cm as cm
import os
import glob

nsim =49

ranges = np.array([0.0,0.1,0.2,0.3,0.4,0.5])#np.linspace(0,1,11)

Truth = pf.open('IMG2.fits')[0].data

sigma = 0.00119
Sources = 0
FSs = 0
thetas = np.zeros((nsim, ranges.size))
Reses = np.zeros((nsim, ranges.size))
SDRs = np.zeros((nsim, ranges.size))
shifts = np.zeros((nsim, ranges.size))
L1s = np.zeros((nsim, ranges.size))
xcs = np.zeros((nsim, ranges.size))
ycs = np.zeros((nsim, ranges.size))

Sources = 0
Images = 0
FSs = 0
x = 0
for i in ranges:
    y = 0
    Images = 0
    Sources = 0
    FSs = 0
    for sim in range(nsim):    
        files = glob.glob('../Results_center/Source_'+str(sim)+'_'+str(i)+'_*.fits')[0]
#        print(files)
        Source = pf.open(files)[0].data
        FS = pf.open('../Results_center/Lensed_source'+files[24:])[0].data
        Image = pf.open('../Results_center/Image'+files[24:])[0].data
        FSs+=FS
        Sources += Source
        Images += Image
        Reses[y,x] = (np.std(Image-FS)/sigma)
        SDRs[y,x] = 10*np.log10(np.sqrt(np.sum(Truth[Truth!=0]**2))/np.sqrt(np.sum((Source[Truth!=0]-Truth[Truth!=0])**2)))
        L1s[y,x] = np.sum(np.abs(SLIT.tools.wave_transform(Source, 6)))
   
        log = files.split('_')
        xcs[y,x] = i*np.cos(np.float(log[4][:6]))
        ycs[y,x] = i*np.sin(np.float(log[4][:6]))
        
        thetas[y,x] = np.float(np.float(log[4][:6]))*180./np.pi
        shifts[y,x] = np.float(i)
        
        y+=1
##    plt.figure(10)
##    plt.imshow(Images-FSs+np.random.randn(FSs.shape[0],FSs.shape[1])*sigma*nsim, cmap = cm.gist_stern, interpolation = 'nearest')
##    plt.colorbar()
##    plt.title('Residuals'+str(i))
##    plt.figure(30)
##    plt.imshow(Images, cmap = cm.gist_stern, interpolation = 'nearest')
##    plt.colorbar()
##    plt.title('Images'+str(i))
##    plt.figure(20)
##    plt.imshow(Sources, cmap = cm.gist_stern, interpolation = 'nearest')
##    plt.colorbar()
##    plt.title('Source'+str(i))
##    plt.figure(40)
##    plt.imshow(Sources-Truth*nsim, cmap = cm.gist_stern, interpolation = 'nearest')
##    plt.colorbar()
##    plt.title('Source Residuals'+str(i))
##    plt.show()
    x+=1


mean_SDRs = np.mean(SDRs,axis = 0)
sigma_SDRs = np.std(SDRs,axis = 0)
mean_L1s = np.mean(L1s,axis = 0)
sigma_L1s = np.std(L1s,axis = 0)
mean_Reses = np.mean(Reses,axis = 0)
sigma_Reses = np.std(Reses,axis = 0)

plt.figure(-1)
sc1 = plt.scatter(xcs.flatten(), ycs.flatten(), c = SDRs.flatten())
plt.colorbar(sc1)
plt.figure(0)
plt.errorbar(ranges, mean_SDRs, yerr = sigma_SDRs)
plt.xlabel('Shifts')
plt.ylabel('SDR')
plt.axis([-0.1,ranges.max()+0.1,mean_SDRs.min()-2*sigma_SDRs[mean_SDRs==mean_SDRs.min()],mean_SDRs.max()+2*sigma_SDRs[mean_SDRs==mean_SDRs.max()]])
plt.figure(1)
plt.errorbar(ranges, mean_Reses, yerr = sigma_Reses)
plt.axis([-0.1,ranges.max()+0.1,mean_Reses.min()-2*sigma_Reses[mean_Reses==mean_Reses.min()],mean_Reses.max()+2*sigma_Reses[mean_Reses==mean_Reses.max()]])
plt.xlabel('Shifts')
plt.ylabel('Residuals')
plt.figure(2)
plt.errorbar(ranges, mean_L1s, yerr = sigma_L1s)
plt.xlabel('Shifts')
plt.ylabel('L1')
plt.figure(3)
sc0 = plt.scatter(thetas.flatten(), SDRs.flatten(), c = np.array(shifts.flatten()))
plt.colorbar(sc0)
plt.xlabel('theta')
plt.ylabel('SDR')
plt.figure(4)
sc = plt.scatter(thetas.flatten(), Reses.flatten(), c = np.array(shifts.flatten()))
plt.colorbar(sc)
plt.xlabel('theta')
plt.ylabel('Residuals')
plt.show()
plt.close()
