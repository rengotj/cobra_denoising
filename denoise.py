# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:21:07 2018
@author: juliette rengot
"""
import numpy as np
import cv2
import skimage.restoration
import scipy.ndimage

import noise

class denoisedImage :
    def __init__(self, noisy, original=None,
                 gauss_sigma=2,
                 median_size=3,     #default parameter for gaussian filter
                 verbose = False):  #If True, print values of denoised images when computed         
        """ Create a class gathering all denoised version of a noisy image
        PARAMETERS
        noisy : noisy image
        original : original version of the noisy image if available
        gauss_sigma : standard deviation for the gaussian filter
        """
        self.verbose = verbose
        self.str2int = {"bilateral" : 0, "nl_means" : 1, "gaussian" : 2, "median" : 3}
        self.int2str = {0 : "bilateral", 1 : "nl_means", 2 : "gaussian", 3 : "median"}
        self.method_nb = len(self.str2int.keys())          # How many denoising methods are available 
        self.Ilist = [None for i in range(self.method_nb)] # List of all available denoised images
        
        self.Inoisy = noisy
        self.shape = self.Inoisy.shape
            
        self.Ioriginal = original
        
        self.Ibilateral = np.empty(self.shape)
        
        self.Inlmeans = np.empty(self.shape)
        
        self.sigma = gauss_sigma
        self.Igauss = np.empty(self.shape)
        
        self.median_size = median_size
        self.Imedian = np.empty(self.shape)
        
               
    def bilateral(self):        
        """ Apply a bilateral filter on the noisy image """
        self.Ibilateral = skimage.restoration.denoise_bilateral(self.Inoisy, multichannel=False)
        self.Ilist[0] = self.Ibilateral
        if self.verbose :
            print('Bilateral :', self.Ibilateral)
        return()
    
    def NLmeans(self):
        """ Apply a Non-local means denoising on the noisy image """
        self.Inlmeans = skimage.restoration.denoise_nl_means(self.Inoisy, multichannel=False)
        self.Ilist[1] = self.Inlmeans
        if self.verbose :
            print('Non-local means :', self.Inlmeans)
        return()
    
    def gauss(self):
        """ Apply a gaussian filter on the noisy image """
        self.Igauss = scipy.ndimage.gaussian_filter(self.Inoisy, self.sigma)
        self.Ilist[2] = self.Igauss
        if self.verbose :
            print('gauss :', self.Igauss)
        return()
    
    def median(self):
        """ Apply a median filter on the noisy image """
        self.Imedian = scipy.ndimage.median_filter(self.Inoisy, self.median_size)
        self.Ilist[3] = self.Imedian
        if self.verbose :
            print('med ', self.Imedian)
        return()
    
    def all_denoise(self):
        """Apply all available denoise methods on the noisy image """
        self.bilateral()
        self.NLmeans()
        self.gauss()
        self.median()
        return()
    
    def show(self, I, title=''):
        """ Display the image I with window entitled 'title' """
        cv2.imshow(title, I)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        return()
    
    def all_show(self):
         """Create and show all possible denoised images of the noisy image """
         self.all_denoise()
         self.show(self.Ibilateral, "Bilateral filter")
         self.show(self.Inlmeans, "Non-local means denoising")
         self.show(self.Igauss, "Gaussian Filter")
         self.show(self.Imedian, "Median Filter")
         return()
            
if (__name__ == "__main__"):
    path = "C://Users//juliette//Desktop//enpc//3A//Graphs_in_Machine_Learning//projet//images//"
    file_name ="lena.png"
    
    noise_class = noise.noisyImage(path,file_name)
    noise_class.all_noise()
    
    im = noise_class.Ioriginal
    im_noise= noise_class.Ipoiss
    
    denoise_class = denoisedImage(im_noise, im)
    denoise_class.all_show()