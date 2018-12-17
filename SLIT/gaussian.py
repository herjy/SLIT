import numpy as np
import scipy.misc as spm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def gaussian(n1,n2,x0,y0,A,e1,e2,alpha):
    #img = gaussian(n1,n2,x0,y0,A,e1,e2,alpha)
    #produces a gaussian profile image
    #INPUTS:
    #   n1,n2: size of the output image
    #   x0,y0: centroid of the gaussian profile
    #   A: value of the maximum value for the gaussian profile
    #   e1,e2: ellipticity og the profile
    #   alpha: inclination of the profile
    #OUTPUTS:
    #   img: n1xn2 image containing the gaussian profile

    Img = np.zeros([n1,n2])
    valcor = np.zeros([2,n1*n2])

    AA = np.zeros([n1,n2])

    xx0 = np.zeros(n1*n2)
    xx0[:]=x0
    yy0 = np.zeros(n2*n1)
    yy0[:]=y0
    coord0 = np.zeros([2, n1*n2])
    coord = np.zeros([2, n1*n2])

    # terme d'amplitude
    ampli = A/(2*np.pi*np.sqrt(e1*e2))

    mat_rot = [[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]]
    tmat_rot = np.transpose(mat_rot)
    matell = [[(e1*e1),0],[0,(e2*e2)]]

    # Matrice des moments quadripolaires
    matA = np.mat(np.dot(np.dot(tmat_rot,matell),mat_rot))

    xc, yc = np.where(Img == 0)
    ii = np.array(xc)
    jj = np.array(yc)
    #print(np.shape(i), np.shape(xx0))
    count = np.linspace(0,n1*n2-1., n1*n2-1.)
    count = np.int_(count) 
    
    valcor = np.array([ii,jj]) - np.array([xx0,yy0])
    valcor = np.array(valcor)
    for k in count:
        val = np.mat(valcor[:,k])
        invA = np.array(np.linalg.inv(matA))
        var = np.dot(np.dot(val,invA),np.transpose(val))
        AA[ii[k],jj[k]]= var
    

    Img = (ampli*np.exp(-0.5*AA))

    return Img

def moffat(n1,n2,x0,y0,A,e1,e2,alpha,beta):
    #img = gaussian(n1,n2,x0,y0,A,e1,e2,alpha)
    #produces a gaussian profile image
    #INPUTS:
    #   n1,n2: size of the output image
    #   x0,y0: centroid of the gaussian profile
    #   A: value of the maximum value for the gaussian profile
    #   e1,e2: ellipticity og the profile
    #   alpha: inclination of the profile
    #OUTPUTS:
    #   img: n1xn2 image containing the gaussian profile

    Img = np.zeros([n1,n2])
    valcor = np.zeros([2,n1*n2])

    AA = np.zeros([n1,n2])

    xx0 = np.zeros(n1*n2)
    xx0[:]=x0
    yy0 = np.zeros(n2*n1)
    yy0[:]=y0
    coord0 = np.zeros([2, n1*n2])
    coord = np.zeros([2, n1*n2])

    # terme d'amplitude
    ampli = A/(2*np.pi*np.sqrt(e1*e2))

    mat_rot = [[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]]
    tmat_rot = np.transpose(mat_rot)
    matell = [[1./(e1*e1),0],[0,1./(e2*e2)]]

    # Matrice des moments quadripolaires
    matA = np.mat(np.dot(np.dot(tmat_rot,matell),mat_rot))

    xc, yc = np.where(Img == 0)
    i = np.array(xc)
    j = np.array(yc)
    #print(np.shape(i), np.shape(xx0))
    count = np.linspace(0,n1*n2-1, n1*n2-1)
    count = np.int_(count) 
    
    valcor = np.array([i,j]) - np.array([xx0,yy0])
    valcor = np.array(valcor)
    for k in count:
        val = np.mat(valcor[:,k])
        invA = np.array(np.linalg.inv(matA))
        var = np.dot(np.dot(val,invA),np.transpose(val))
        AA[i[k],j[k]]= var
    

    Img = (ampli*(1+AA**2)**(-beta))

    return Img

def sersic(n1,n2,x0,y0,A,e1,e2,alpha,n):
    #img = gaussian(n1,n2,x0,y0,A,e1,e2,alpha)
    #produces a gaussian profile image
    #INPUTS:
    #   n1,n2: size of the output image
    #   x0,y0: centroid of the gaussian profile
    #   A: value of the maximum value for the gaussian profile
    #   e1,e2: ellipticity og the profile
    #   alpha: inclination of the profile
    #OUTPUTS:
    #   img: n1xn2 image containing the gaussian profile

    Img = np.zeros([n1,n2])
    valcor = np.zeros([2,n1*n2])

    AA = np.zeros([n1,n2])

    xx0 = np.zeros(n1*n2)
    xx0[:]=x0
    yy0 = np.zeros(n2*n1)
    yy0[:]=y0
    coord0 = np.zeros([2, n1*n2])
    coord = np.zeros([2, n1*n2])

    # terme d'amplitude
    ampli = A/(2*np.pi*np.sqrt(e1*e2))

    mat_rot = [[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]]
    tmat_rot = np.transpose(mat_rot)
    matell = [[1./(e1*e1),0],[0,1./(e2*e2)]]

    # Matrice des moments quadripolaires
    matA = np.mat(np.dot(np.dot(tmat_rot,matell),mat_rot))

    xc, yc = np.where(Img == 0)
    i = np.array(xc)
    j = np.array(yc)
    #print(np.shape(i), np.shape(xx0))
    count = np.linspace(0,n1*n2-1, n1*n2-1)
    count = np.int_(count) 
    
    valcor = np.array([i,j]) - np.array([xx0,yy0])
    valcor = np.array(valcor)
    for k in count:
        val = np.mat(valcor[:,k])
        invA = np.array(np.linalg.inv(matA))
        var = np.dot(np.dot(val,invA),np.transpose(val))
        AA[i[k],j[k]]= var
    

    Img = (ampli*np.exp(-AA**(1/n)))

    return Img


def add_noise(img, mean, sigma):
    shp = np.shape(img)
    n1 = shp[0]
    cov = numpy.identity(2)
    noise = np.random.multovariate_normal([mean,mean], cov, [128,128] )
    imfinal = img+noise[:,:,0]
    return imfinal
    

