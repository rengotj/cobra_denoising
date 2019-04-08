# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 20:22:25 2018
@author: juliette rengot
"""
import numpy as np
import cv2

class noisyImage :
    def __init__(self, path, file_name,                    # original image information
                 color=0,                                  # if 1 use color image, if 0 use grayscale image
                 gauss_mu=0, gauss_sigma=0.3,              # default parameters for gaussian noise
                 sp_ratio=0.5, sp_amount=0.004,            # default parameters for salt and pepper noise
                 suppr_patch_size=1, suppr_patch_nb=1,     # default parameters for random patch suppression
                 verbose = False):                         # If True, print values of noisy images when computed         
        """ Create a class gathering all noisy version of an original image
        PARAMETERS
        path : path where is located the original image
        file_name : name of the original image
        gauss_mu : mean of gaussian noise
        gauss_sigma : variance of gaussian noise
        """
        
        self.verbose = verbose
        self.str2int = {"gaussian" : 0, "salt_pepper" : 1, "poisson" : 2, "speckle" : 3, "suppression" : 4, "multi" : 5}
        self.int2str = {0 : "gaussian", 1 : "salt_pepper", 2 : "poisson", 3 : "speckle", 4 : "suppression", 5 : "multi"}
        self.method_nb = len(self.str2int)                 # How many denoising methods are available 
        self.Ilist = [None for i in range(self.method_nb)] # List of all available noisy images
        
        self.name = path+file_name
        assert(color==0 or color==1)
        original = cv2.imread(self.name, color)
        if (np.max(original)!=np.min(original)):
            self.Ioriginal = (original-np.min(original))/(np.max(original)-np.min(original))
        else:
            self.Ioriginal = original
            
        self.shape = (self.Ioriginal).shape
        self.size = (self.Ioriginal).size
    
        self.mu = gauss_mu
        self.sigma = gauss_sigma
        self.Igauss = np.empty(self.shape)

        self.s_vs_p = sp_ratio
        self.amount = sp_amount
        self.Isp = np.empty(self.shape)
        
        self.Ipoiss = np.empty(self.shape)
        
        self.Ispeckle = np.empty(self.shape)
        
        self.patch_size=suppr_patch_size
        self.patch_nb=suppr_patch_nb
        self.Isuppr = np.empty(self.shape)

        self.Imulti = np.empty(self.shape)
        
    def add_gauss(self):
        """ Add gaussian noise to the original image """
        I_gauss = np.random.normal(self.mu, self.sigma, self.shape)
        I_gauss = self.Ioriginal+I_gauss
        if (np.max(I_gauss)!=np.min(I_gauss)):
            I_gauss = (I_gauss-np.min(I_gauss))/(np.max(I_gauss)-np.min(I_gauss))
        self.Igauss = I_gauss
        self.Ilist[0] = I_gauss
        
        if self.verbose :
            print("Gauss ", self.Igauss)
        return()
    
    def salt_and_pepper(self):
        """ Apply salt and pepper noise on the original image """
        I_sp = np.copy(self.Ioriginal)
        # Salt mode
        num_salt = np.ceil(self.amount * self.size * self.s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.shape]
        if len(I_sp.shape) == 3 :
            I_sp[coords[0], coords[1], :] = [1, 1, 1]
        else :
            I_sp[tuple(coords)] = 1
        # Pepper mode
        num_pepper = np.ceil(self.amount* self.size * (1. - self.s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.shape]
        if len(I_sp.shape) == 3 :
            I_sp[coords[0], coords[1], :] = [0, 0, 0]
        else :
            I_sp[tuple(coords)] = 0
        
        if (np.max(I_sp)!=np.min(I_sp)) :
            I_sp = (I_sp-np.min(I_sp))/(np.max(I_sp)-np.min(I_sp))    
        self.Isp = I_sp
        self.Ilist[1]=self.Isp
        
        if self.verbose :
            print("Salt and pepper ", self.Isp)
        return()
        
    def poisson(self):
        """ Apply noise with poisson distribution on the original image """
        val = len(np.unique(self.Ioriginal))
        val = 2 ** np.ceil(np.log2(val))
        I_poisson = np.random.poisson(self.Ioriginal*val)/float(val)
        
        if (np.max(I_poisson)!=np.min(I_poisson)): 
            I_poisson = (I_poisson-np.min(I_poisson))/(np.max(I_poisson)-np.min(I_poisson))
        self.Ipoiss = I_poisson
        self.Ilist[2]=self.Ipoiss
        if self.verbose :
            print("Poisson ", self.Ipoiss)
        return()
    
    def speckle(self):
        """ Apply speckle noise on the original image """
        if len(self.shape) == 2 :
            gauss = np.random.randn(self.shape[0],self.shape[1])
        if len(self.shape) == 3 :
            gauss = np.random.randn(self.shape[0], self.shape[1], self.shape[2])
        I_speckle = self.Ioriginal + self.Ioriginal * gauss
        
        if (np.max(I_speckle)!=np.min(I_speckle))  :
            I_speckle = (I_speckle-np.min(I_speckle))/(np.max(I_speckle)-np.min(I_speckle))
        self.Ispeckle = I_speckle
        self.Ilist[3]=self.Ispeckle
        if self.verbose :
            print("Speckle ", self.Ispeckle)
        return()
    
    def suppr(self, I):
        """ Suppress random patch from an image I """
        I_lack=np.copy(I)
        for i in range(self.patch_nb):
            x = np.random.randint(0,self.shape[0]-self.patch_size)
            y = np.random.randint(0,self.shape[1]-self.patch_size)
            I_lack[x:x+self.patch_size,y:y+self.patch_size]=1
        return(I_lack)
        
    def random_patch_suppression(self):
        """ Suppress random patch from the original image """
        Isuppr = self.suppr(self.Ioriginal)
        self.Isuppr = Isuppr
        self.Ilist[4] = self.Isuppr
        if self.verbose :
            print("Random patch suppression ", self.Isuppr)
        return()
        
    def all_noise(self):
         """Apply all available noise methods on the original image """
         self.add_gauss()
         self.salt_and_pepper()
         self.poisson()
         self.speckle()
         self.random_patch_suppression()
    
    def multi_noise(self):
        self.all_noise()
        Imulti = np.zeros(self.shape)
        Imulti[0:self.shape[0]//2, 0:self.shape[1]//2] = self.Igauss[0:self.shape[0]//2, 0:self.shape[1]//2]
        Imulti[0:self.shape[0]//2, self.shape[1]//2:self.shape[1]] = self.Isp[0:self.shape[0]//2, self.shape[1]//2:self.shape[1]]
        Imulti[self.shape[0]//2:self.shape[0], 0:self.shape[1]//2] = self.Ipoiss[self.shape[0]//2:self.shape[0], 0:self.shape[1]//2]
        Imulti[self.shape[0]//2:self.shape[0], self.shape[1]//2:self.shape[1]] = self.Ispeckle[self.shape[0]//2:self.shape[0], self.shape[1]//2:self.shape[1]]
        Imulti = self.suppr(Imulti)
        self.Imulti = Imulti
        self.Ilist[5] = self.Imulti
        if self.verbose :
            print("Multi noise ", self.Imulti)
        return
     
    def show(self, I, title=''):
        """ Display the image I with window entitled 'title' """
        cv2.imshow(title, I)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
    def all_show(self):
         """Create and show all possible noise images """
         self.multi_noise()
         print("Salt and Pepper noise")
         self.show(self.Isp, "Salt and Pepper noise")
         print("Speckle noise noise")
         self.show(self.Ispeckle, "Speckle noise noise")
         print("Poisson noise")
         self.show(self.Ipoiss, "Poisson noise")
         print("Gaussian additive noise")
         self.show(self.Igauss, "Gaussian additive noise")
         print("Missing part")
         self.show(self.Isuppr,"Missing part")
         print("Multi noise")
         self.show(self.Imulti,"Multi noise")

if (__name__ == "__main__"):
    path = "images//"
    file_name ="lena.png"
    
    noise_class=noisyImage(path, file_name, 1, 0.5, 0.1, 0.2, 0.3, 10, 20)
    noise_class.all_show()
    