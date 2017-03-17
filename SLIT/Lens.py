import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import scipy.signal as scp
import warnings
warnings.simplefilter("ignore")


#Tool box for lensing



def alpha_def(kappa, n1,n2,x0,y0,extra):
	#Computes the deflection angle of a single photon at coordinates theta in the source plane and a lens 
	#mass distribution kappa
	
    nk1,nk2 = np.shape(kappa)
    
    #Coordonnees de la grille de l'espace image
    [x,y] = np.where(np.zeros([n1,n2])==0)
    
    xc = np.reshape((x)-x0+1.,(n1,n2))
    yc = np.reshape((y)-y0+1.,(n1,n2))
    

    r = np.reshape((xc**2+yc**2),(n1,n2))
    lx,ly = np.where(r==0)
    tabx = np.reshape((xc)/r,(n1,n2))
    taby = np.reshape((yc)/r,(n1,n2))
    tabx[lx,ly]=0
    taby[lx,ly]=0

    l = 0


    kappa = kappa.astype(float)
    tabx = tabx.astype(float)
    intex = scp.fftconvolve(tabx, (kappa), mode = 'same')/np.pi
    intey = scp.fftconvolve(taby, (kappa), mode = 'same')/np.pi

    x0+=extra/2.
    y0+=extra/2.

    
    return intex[x0-(n1)/2.:x0+(n1)/2,y0-(n2)/2.:y0+(n2)/2.],intey[x0-(n1)/2.:x0+(n1)/2,y0-(n2)/2.:y0+(n2)/2.]

def beta(kappa,theta):
    #Computes beta
    beta = theta-alpha(theta,kappa)
    return beta

def theta(alpha, beta):
    #Computes beta
    theta = beta+alpha
    return beta

def F(kappa, nt1,nt2, size, x0, y0, extra=100,local = False, alpha_file = 'none'):
	# Theta positions for each pixel in beta
    
    if local == False:
        nk1,nk2 = np.shape(kappa)
        alpha = np.zeros((2,nt1,nt2))
        alpha_x,alpha_y = alpha_def(kappa,nt1,nt2,x0,y0,extra)
        

    alpha[0,:,:] = alpha_x
    alpha[1,:,:] = alpha_y
    
    if alpha_file != 'none':
        alpha = pf.open(alpha_file)[0].data
        xa = alpha[0,:,:]
        ya = alpha[1,:,:]
        x = xi
        y = yi
        N = len(xi)

  
    
    na1,na2 = np.shape(alpha_x)
    xa,ya = np.where(np.zeros((na1,na2)) == 0)
    xt, yt = np.where(np.zeros((nt1,nt2)) == 0)
    
    nb1=nt1*size
    nb2=nt2*size
    xb, yb = np.where(np.zeros((nb1,nb2))==0)

    #Scaling of the source grid
    
    #Scaling of the deflection grid
    xa = xa*(np.float(nt1)/np.float(na1))-1
    ya = ya*(np.float(nt2)/np.float(na2))-1
    #Setting images coordinates in 2d
    xa2d = np.reshape(xa,(na1,na2))
    ya2d = np.reshape(ya,(na1,na2))


    
    F = []
    rec = []
    for i in range(np.size(xb)):
        #Deflection of photons emitted in xb[i],yb[i]
        theta_x = (xb[i])*(np.float(nt1)/np.float(nb1))+alpha_x
        theta_y = (yb[i])*(np.float(nt2)/np.float(nb2))+alpha_y
        #Matching of arrivals with pixels in image plane
        xprox = np.int_(np.abs((xa2d-theta_x)*2))
        yprox = np.int_(np.abs((ya2d-theta_y)*2))

        if np.min(xprox+yprox) <3:
            loc = np.where((xprox+yprox)==np.min(xprox+yprox))#
        else:
            loc = []
        if (np.size(loc)==0):

            F.append([0])
        else:
            F.append(np.int_(np.array(loc)))
    return F


def source_to_image(Source, nt1,nt2, theta, ones = 1):
    # Source: Image of the source in the source plane
    # n1,n2: size in pixels of the postage stamp in image plane
    # F: the coordinates of the lens mapping
    F = (theta)
    nb1,nb2 = np.shape(Source)

    if ones == 1:
        onelens = source_to_image(np.ones(Source.shape), nt1,nt2, theta, ones = 0)
        onelens[np.where(onelens==0)]=1
        
    else:
        onelens = 1.
    

    Image = np.zeros((nt1,nt2))
    xb,yb = np.where(np.zeros((nb1,nb2)) == 0)

    N = np.size(xb)
    k=0
    for pos in F:
                
                if np.size(np.shape(pos)) != 1:
                        
                        Image[np.array(pos[0][np.where((pos[0][:]<nt1))]),
                              np.array(pos[1][np.where(pos[1][:]<nt2)])] += Source[xb[k],yb[k]]#fullSource
                k=k+1
    return Image/onelens


def image_to_source(Image, size,beta,lensed = 0):
    # Image: postagestamp of the observed image
    # nsize1,nsize2: size of the postagestamp in source plane
    # F: lens mapping matrix

    F = (beta)
    nt1,nt2 = np.shape(Image)
    nb1 = nt1*size
    nb2 = nt2*size
    Source = np.zeros((nb1,nb2))
    xb,yb = np.where(Source == 0)
    N = np.size(xb)
    for k in range(N):
                    pos = F[k]
                    if np.size(np.shape(pos)) > 1:
                        if np.sum(lensed) !=0:
                            Source[xb[k],yb[k]] += np.sum(Image[np.array(pos[0][:]),
                                                           np.array(pos[1][:])])/np.max([1,np.size(pos[0][:])])
                        else:
                            Source[xb[k],yb[k]] += np.sum(Image[np.array(pos[0][:]),
                                                           np.array(pos[1][:])])
                        
    return Source








