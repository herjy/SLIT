import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from SLIT import Lens
import scipy.signal as scp
import scipy.misc as misc

def plot_critical(kappa, Fkappa, n1,n2, size,  extra= 1):

    det = 1./Lens.Jacobian_det(kappa, n1,n2)[extra/2:-extra/2., extra/2:-extra/2.]


    kernel = np.array([[1,0],[0,-1]])
    kernelT = np.array([[0,1],[-1,0]])
    diff = np.abs(-det+np.abs(det))

    diff[diff!=0]=1
    xderiv = scp.convolve2d(diff, kernel, mode='same')
    yderiv = scp.convolve2d(diff, kernelT, mode='same')



    x,y = np.where(np.abs(xderiv)+np.abs(yderiv)>0)
    #newx,newy = mk_curve(x,y)
    critical = np.zeros((n1,n2))
    critical[x,y] = 1

    Splane = Lens.image_to_source(critical, Fkappa, 0)

    xs,ys = np.where(Splane !=0)
    #newxs,newys = mk_curve(xs,ys)
    factor = np.float(size)
    return x/factor,y/factor, xs, ys

def mk_curve(x,y):

    newx, newy = np.zeros(np.size(x)+1), np.zeros(np.size(y)+1)
    newx[0], newy[0] = x[0], y[0]
    newx[-1], newy[-1] = x[0], y[0]
    x[0],y[0] = -999,-999
    for i in range(x.size-1):


        r = (newx[i]-x)**2+(newy[i]-y)**2


        if np.size(np.where(r == np.min(r)))>1:
            newx[i + 1], newy[i + 1] = int(x[(r == np.min(r)) * (r != 0)][0]), int(y[(r == np.min(r)) * (r != 0)][0])

        else:
            newx[i+1], newy[i+1] = int(x[(r == np.min(r))*(r!=0)]), int(y[(r == np.min(r))*(r!=0)])
        x[(newx[i+1]-x)**2+(newy[i+1]-y)**2==0], y[(newx[i+1]-x)**2+(newy[i+1]-y)**2==0] =-999,-999
    return (newy.astype(int)), (newx.astype(int))

def Plot_SLIT_Results(Y,S,FS, TrueS, TrueFS, sigma, TitleFont = 40, ColorbarFont = 25, x=[0], y=[0], xs = [0], ys = [0], delta_pix = 0):
    ns1,ns2 = S.shape
    n1,n2 = Y.shape
    size = ns1 / np.float(n1)
    if delta_pix > 0:
        L = 1. / delta_pix  # Length of 1 arceseconds in pixels
        Ls = L * size
        XXarc = [n1 / 10, n1 / 10 + L]
        XYarc = [n2 / 10, n2 / 10]
        YYarc = [n2 / 10, n2 / 10 + L]
        YXarc = [n1 / 10, n1 / 10]
        XXarcs = [ns1 / 10, ns1 / 10 + Ls]
        XYarcs = [ns2 / 10, ns2 / 10]
        YYarcs = [ns2 / 10, ns2 / 10 + Ls]
        YXarcs = [ns1 / 10, ns1 / 10]

    plt.figure(0)
    plt.title('$\~{S}$', fontsize=TitleFont)
    plt.imshow((S), vmin=np.min(TrueS), vmax=np.max(TrueS), cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(ys, xs, 'w.', ms=2)
    if delta_pix >0:
        plt.plot(XXarc, XYarc, 'w', linewidth = 10 )
        plt.plot(YXarc, YYarc, 'w', linewidth = 10 )
        plt.text(n1/5+10, n2/5+10, '$1\"$', color = 'white', fontsize = 25 )
    plt.axis('off')
    plt.xlim(xmax = ns1)
    plt.ylim(ymax = ns2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.figure(1)
    plt.title('$S$', fontsize=TitleFont)
    plt.imshow(TrueS, cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(ys, xs, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = ns1)
    plt.ylim(ymax = ns2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.figure(2)
    plt.title('$|S-\~S|$', fontsize=TitleFont)
    diff = (TrueS - S)
    plt.imshow((np.abs(diff)), cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(ys, xs, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = ns1)
    plt.ylim(ymax = ns2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    ####Lensed source
    plt.figure(3)
    plt.title('$HFS$', fontsize=TitleFont)
    plt.imshow(TrueFS, cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(y, x, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = n1)
    plt.ylim(ymax = n2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.figure(4)
    plt.title('$HF\~S$', fontsize=TitleFont)
    plt.imshow((FS), vmin=np.min(TrueFS), vmax=np.max(TrueFS), cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(y, x, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = n1)
    plt.ylim(ymax = n2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.figure(5)
    plt.title('$|HFS-HF\~S|$', fontsize=TitleFont)
    plt.imshow((TrueFS - FS), cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(y, x, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = n1)
    plt.ylim(ymax = n2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    ###Image
    plt.figure(8)
    plt.title('$Y$', fontsize=TitleFont)
    plt.imshow(Y, cmap=cm.gist_stern, interpolation='nearest')
    plt.plot(y, x, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = n1)
    plt.ylim(ymax = n2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.figure(10)
    plt.title('$|Y-HF\~S|$', fontsize=TitleFont)
    plt.imshow(Y - FS, cmap=cm.gist_stern, interpolation='nearest', vmin=-5 * sigma,
               vmax=5 * sigma)  # slit.fft_convolve(Im,PSF)
    plt.plot(y, x, 'w.', ms=2)
    plt.axis('off')
    plt.xlim(xmax = n1)
    plt.ylim(ymax = n2)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=ColorbarFont)
    plt.show()
    return 0

def Plot_MCA_Results(Y,S,FS,G,FG, TrueS, TrueG, TrueFS, sigma, TitleFont = 40, ColorbarFont = 25, x=[0], y=[0], xs = [0], ys = [0], delta_pix = 0):

     ns1,ns2 = S.shape
     n1,n2 = G.shape
     size = ns1/np.float(n1)
     if delta_pix > 0:
         L = 1. / delta_pix  # Length of 1 arceseconds in pixels
         Ls = L*size
         XXarc = [n1 / 10, n1 / 10 + L]
         XYarc = [n2 / 10, n2 / 10]
         YYarc = [n2 / 10, n2 / 10 + L]
         YXarc = [n1 / 10, n1 / 10]
         XXarcs = [ns1 / 10, ns1 / 10 + Ls]
         XYarcs = [ns2 / 10, ns2 / 10]
         YYarcs = [ns2 / 10, ns2 / 10 + Ls]
         YXarcs = [ns1 / 10, ns1 / 10]

     plt.figure(0)
     plt.title('$\~{S}$', fontsize=TitleFont)
     plt.imshow((S), vmin=np.min(TrueS), vmax=np.max(TrueS), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(ys,xs,'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=ns1)
     plt.ylim(ymax=ns2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarcs, XYarcs, 'w', linewidth=3)
         plt.plot(YXarcs, YYarcs, 'w', linewidth=3)
         plt.text(ns1 / 10 + 5, ns2 / 10 + 10, '1 "', color='white', fontsize=25)

     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(1)
     plt.title('$S$', fontsize=TitleFont)
     plt.imshow(TrueS, cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(ys, xs, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=ns1)
     plt.ylim(ymax=ns2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarcs, XYarcs, 'w', linewidth=3)
         plt.plot(YXarcs, YYarcs, 'w', linewidth=3)
         plt.text(ns1 / 10 + 5, ns2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(2)
     plt.title('$S-\~S$', fontsize=TitleFont)
     diff = (TrueS - S)
     plt.imshow(diff, cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(ys, xs, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=ns1)
     plt.ylim(ymax=ns2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarcs, XYarcs, 'w', linewidth=3)
         plt.plot(YXarcs, YYarcs, 'w', linewidth=3)
         plt.text(ns1 / 10 + 5, ns2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     ####Lensed source
     plt.figure(3)
     plt.title('$HFS$', fontsize=TitleFont)
     plt.imshow(TrueFS, cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(4)
     plt.title('$HF\~S$', fontsize=TitleFont)
     plt.imshow((FS), vmin=np.min(TrueFS), vmax=np.max(TrueFS), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(5)
     plt.title('$HFS-HF\~S$', fontsize=TitleFont)
     plt.imshow((TrueFS - FS), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     ###Galaxy
     plt.figure(6)
     plt.title('$HG$', fontsize=TitleFont)
     plt.imshow((TrueG), vmin=np.min(TrueG), vmax=np.max(TrueG), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(12)
     plt.title('$H\~G$', fontsize=TitleFont)
     plt.imshow((G), vmin=np.min(TrueG), vmax=np.max(TrueG), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(7)
     plt.title('$HG-H\~G$', fontsize=TitleFont)
     plt.imshow((TrueG - G), cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     ###Image
     plt.figure(8)
     plt.title('$Y$', fontsize=TitleFont)
     plt.imshow(Y, cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(9)
     plt.title('$H\~G+HF\~S$', fontsize=TitleFont)
     plt.imshow(FS + FG, cmap=cm.gist_stern, interpolation='nearest')
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.figure(10)
     plt.title('$Y-H\~G-HF\~S$', fontsize=TitleFont)
     plt.imshow(Y - FS - FG, cmap=cm.gist_stern, interpolation='nearest', vmin=-5 * sigma,
                vmax=5 * sigma)  # slit.fft_convolve(Im,PSF)
     plt.plot(y, x, 'w.', ms = 2)
     plt.axis('off')
     plt.xlim(xmax=n1)
     plt.ylim(ymax=n2)
     plt.xlim(xmin=0)
     plt.ylim(ymin=0)
     if delta_pix > 0:
         plt.plot(XXarc, XYarc, 'w', linewidth=3)
         plt.plot(YXarc, YYarc, 'w', linewidth=3)
         plt.text(n1 / 10 + 5, n2 / 10 + 10, '1 "', color='white', fontsize=25)
     cbar = plt.colorbar()
     cbar.ax.tick_params(labelsize=ColorbarFont)
     plt.show()
     return 0