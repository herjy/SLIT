import numpy as np
import matplotlib.pyplot as plt
import pyfits as pf
import scipy.signal as scp
import warnings
warnings.simplefilter("ignore")


#Tool box for lensing
def SIS(x0,y0,n1,n2,Re):
    kappa = np.zeros((n1,n2))
    x,y = np.where(kappa == 0)
    count = 0
    for i in x:
        kappa[x[count],y[count]] = Re/(2*np.sqrt((x[count]-x0)**2+(y[count]-y0)**2))
        
        count += 1
    if np.isfinite(kappa[x0,y0]) == False:
        kappa[x0,y0] = 1
    
    return kappa

def SIE_xy(x,y,x0,y0,b,beta,q,xc,theta):
    eps = (1-q**2)/(1+q**2)
    up = b**(beta-1)
    pre = up/(2*(1-eps)**((beta-1)/2))
    count = 0
    theta = theta*np.pi/180.
    Xr = (x-x0)*np.cos(theta)-(y-y0)*np.sin(theta)
    Yr = (x-x0)*np.sin(theta)+(y-y0)*np.cos(theta)
    kappa = pre/((xc**2.)/(1.-eps)+(Xr)**2.+((Yr)**2.)/q**2.)**((beta-1.)/2.)
    
    return kappa

def SIE(x0,y0,n1,n2,b,beta,q,xc,theta):
    kappa = np.zeros((n1,n2))
    x,y = np.where(kappa == 0)
    x2d = np.reshape(x, (n1,n2))
    y2d = np.reshape(y, (n1,n2))
    kappa = SIE_xy(x2d,y2d,x0,y0,b,beta,q,xc,theta)
    return kappa


def alpha_def(kappa, n1,n2,extra=0):
	#Computes the deflection angle of a single photon at coordinates theta in the source plane and a lens 
	#mass distribution kappa
	
    nk1,nk2 = np.shape(kappa)
    #Coordonnees de la grille de l'espace image
    [x,y] = np.where(np.zeros([nk1,nk2])==0)

    x0 = nk1/2
    y0 = nk2/2


    xc = np.reshape((x)-x0,(nk1,nk2))
    yc = np.reshape((y)-y0,(nk1,nk2))

    

    r = (xc**2+yc**2)
    lx,ly = np.where(r==0)
    tabx = np.reshape(np.float_(xc)/(r),(nk1,nk2))
    taby = np.reshape(np.float_(yc)/(r),(nk1,nk2))
    tabx[lx,ly]=0
    taby[lx,ly]=0

    kappa = kappa.astype(float)
    tabx = tabx.astype(float)

#   kappa[rk>(nk1)/2.] = 0
    
    intex = scp.fftconvolve(tabx, (kappa), mode = 'same')/np.pi
    intey = scp.fftconvolve(taby, (kappa), mode = 'same')/np.pi

    return intex[x0-(n1)/2:x0+(n1)/2,y0-(n2)/2:y0+(n2)/2], intey[x0-(n1)/2:x0+(n1)/2,y0-(n2)/2:y0+(n2)/2]


def beta(kappa,theta):
    #Computes beta
    beta = theta-alpha(theta,kappa)
    return beta

def theta(alpha, beta):
    #Computes beta
    theta = beta+alpha
    return beta

def F(kappa, nt1,nt2, size, extra=100, x_shear = 0, y_shear = 0, alpha_x_in = [-99], alpha_y_in = [-99]):
	# Theta positions for each pixel in beta


    if np.sum(alpha_x_in) != [-99]:
        alpha_x = alpha_x_in
        alpha_y = alpha_y_in
    else:

        nk1,nk2 = np.shape(kappa)
        
        alpha_x,alpha_y = alpha_def(kappa,nt1,nt2,extra = extra)


    alpha_x = alpha_x+x_shear
    alpha_y = alpha_y+y_shear


    na1,na2 = np.shape(alpha_x)
    xa,ya = np.where(np.zeros((na1,na2)) == 0)

    
    nb1=nt1*size
    nb2=nt2*size
    xb, yb = np.where(np.zeros((nb1,nb2))==0)

    #Scaling of the source grid
    
    #Scaling of the deflection grid
    xa = xa*(np.float(nt1)/np.float(na1))#-0.68
    ya = ya*(np.float(nt2)/np.float(na2))#-0.68
    #Setting images coordinates in 2d
    xa2d = np.reshape(xa,(na1,na2))
    ya2d = np.reshape(ya,(na1,na2))



    F2 = []
    for i in range(np.size(xb)):
        #Deflection of photons emitted in xb[i],yb[i]
        theta_x = (xb[i])*(np.float(nt1)/np.float(nb1))+alpha_x
        theta_y = (yb[i])*(np.float(nt2)/np.float(nb2))+alpha_y

        #Matching of arrivals with pixels in image plane
        xprox = np.int_(np.abs((xa2d-theta_x)*2))
        yprox = np.int_(np.abs((ya2d-theta_y)*2))

            
        if np.min(xprox+yprox) <1:
            loc2 = np.array(np.where((xprox+yprox)==np.min(xprox+yprox)))*np.float(nt1)/np.float(na1)#
        else:
            loc2 = []
        if (np.size(loc2)==0):

            F2.append([0])
        else:
            F2.append(np.int_(loc2))
    return F2


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
                        
                        Image[np.array(pos[0][:]),
                              np.array(pos[1][:])] += Source[xb[k],yb[k]]#fullSource
                k=k+1
    return Image/onelens


def image_to_source(Image, size,beta,lensed = 0, square = 0):
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
                            if square == 0:
                                Source[xb[k],yb[k]] += np.sum(Image[np.array(pos[0][:]),
                                                               np.array(pos[1][:])])/np.max([1,np.size(pos[0][:])])
                            else:
                                Source[xb[k], yb[k]] += np.sum((Image[np.array(pos[0][:]),
                                                                np.array(pos[1][:])] / np.max([1, np.size(pos[0][:])]))**2)
                        else:
                            Source[xb[k],yb[k]] += np.sum(Image[np.array(pos[0][:]),
                                                           np.array(pos[1][:])])
    if square == 1:
        Source = np.sqrt(Source)
    return Source

def image_to_source_bound(Image, size,beta,lensed = 0):
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

