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
    def __init__(self, noisy, original=None, color=0,
                 gauss_sigma=0.8,   #default parameter for gaussian filter
                 median_size=3,     #default parameter for the median filter
                 point_spread_rl=5, #default parameter for the richardson lucy deconvolution
                 verbose = False):  #If True, print values of denoised images when computed         
        """ Create a class gathering all denoised version of a noisy image
        PARAMETERS
        noisy : noisy image
        original : original version of the noisy image if available
        gauss_sigma : standard deviation for the gaussian filter
        """
        self.verbose = verbose
        self.str2int = {"bilateral" : 0, "nl_means" : 1, "gaussian" : 2, "median" : 3, "tv_chambolle" : 4, "richardson_lucy" : 5, "inpainting" : 6}
        self.int2str = {0 : "bilateral", 1 : "nl_means", 2 : "gaussian", 3 : "median", 4 : "tv_chambolle", 5 : "richardson_lucy", 6 : "inpainting"}
        self.method_nb = len(self.str2int)                 # How many denoising methods are available 
        self.Ilist = [None for i in range(self.method_nb)] # List of all available denoised images
        
        self.color = color
        
        self.Inoisy = noisy
        self.shape = self.Inoisy.shape
            
        self.Ioriginal = original
        
        self.Ibilateral = np.empty(self.shape)
        
        self.Inlmeans = np.empty(self.shape)
        
        self.gauss_sigma = gauss_sigma
        self.Igauss = np.empty(self.shape)
        
        self.median_size = median_size
        self.Imedian = np.empty(self.shape)
        
        self.Ichambolle = np.empty(self.shape)
        
        self.point_spread_rl = point_spread_rl
        self.Irl = np.empty(self.shape)

        self.Iinpaint = np.empty(self.shape)
               
    def bilateral(self):        
        """ Apply a bilateral filter on the noisy image """
        if self.color==0 :
            self.Ibilateral = skimage.restoration.denoise_bilateral(self.Inoisy, multichannel=False)
        else :
            self.Ibilateral = skimage.restoration.denoise_bilateral(self.Inoisy, multichannel=True)
        self.Ilist[self.str2int['bilateral']] = self.Ibilateral
        if self.verbose :
            print('Bilateral :', self.Ibilateral)
        return()
    
    def NLmeans(self):
        """ Apply a Non-local means denoising on the noisy image """
        if self.color==0 :
            self.Inlmeans = skimage.restoration.denoise_nl_means(self.Inoisy, multichannel=False)
        else :
            self.Inlmeans = skimage.restoration.denoise_nl_means(self.Inoisy, multichannel=True)
        self.Ilist[self.str2int['nl_means']] = self.Inlmeans
        if self.verbose :
            print('Non-local means :', self.Inlmeans)
        return()
    
    def gauss(self):
        """ Apply a gaussian filter on the noisy image """
        self.Igauss = scipy.ndimage.gaussian_filter(self.Inoisy, self.gauss_sigma)
        self.Ilist[self.str2int['gaussian']] = self.Igauss
        if self.verbose :
            print('gauss :', self.Igauss)
        return()
    
    def median(self):
        """ Apply a median filter on the noisy image """
        self.Imedian = scipy.ndimage.median_filter(self.Inoisy, self.median_size)
        self.Ilist[self.str2int['median']] = self.Imedian
        if self.verbose :
            print('med ', self.Imedian)
        return()

    def TVchambolle(self):
        """ Perform total-variation denoising on n-dimensional images. """
        if self.color==0 :
            self.Ichambolle = skimage.restoration.denoise_tv_chambolle(self.Inoisy, multichannel=False)
        else :
            self.Ichambolle = skimage.restoration.denoise_tv_chambolle(self.Inoisy, multichannel=True)
        self.Ilist[self.str2int['tv_chambolle']] = self.Ichambolle
        if self.verbose :
            print('TV chambolle :', self.Ichambolle)
        return()

    def richardson_lucy(self):
        """Richardson-Lucy deconvolution."""
        psf = np.ones((self.point_spread_rl, self.point_spread_rl)) / self.point_spread_rl**2
        if self.color == 0:
            I = skimage.restoration.richardson_lucy(self.Inoisy, psf, self.point_spread_rl)
        else:
            I = np.zeros(self.shape)
            I[:,:,0] =  skimage.restoration.richardson_lucy(self.Inoisy[:,:,0], psf, self.point_spread_rl)
            I[:,:,1] =  skimage.restoration.richardson_lucy(self.Inoisy[:,:,1], psf, self.point_spread_rl)
            I[:,:,2] =  skimage.restoration.richardson_lucy(self.Inoisy[:,:,2], psf, self.point_spread_rl)
        self.Irl = I
        self.Ilist[self.str2int['richardson_lucy']] = self.Irl
        if self.verbose :
            print('Richardson Lucy :', self.Irl)
        return()

    def inpaint(self):
        """Inpainting"""
        
        if self.color == 0:
            mask = (self.Inoisy==1)
            I = skimage.restoration.inpaint.inpaint_biharmonic(self.Inoisy, mask, multichannel=False)
        else:
            mask = (self.Inoisy.mean(axis=2)==1)
            I = skimage.restoration.inpaint.inpaint_biharmonic(self.Inoisy, mask, multichannel=True)            

        self.Iinpaint = I
        self.Ilist[self.str2int['inpainting']] = self.Iinpaint
        if self.verbose :
            print('inpainting :', self.Iinpaint)
        return()
    
    def all_denoise(self):
        """Apply all available denoise methods on the noisy image """
        self.bilateral()
        self.NLmeans()
        self.gauss()
        self.median()
        self.TVchambolle()
        self.richardson_lucy()
        self.inpaint()
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
         self.show(self.Ichambolle, "TVchambolle")
         self.show(self.Irl, "Richardson Lucy deconvolution")
         self.show(self.Iinpaint, "inpainting")
         return()
            
if (__name__ == "__main__"):
    path = "C://Users//juliette//Desktop//enpc//3A//Graphs_in_Machine_Learning//projet//images//"
    file_name ="lena.png"
    color = 1
    
    noise_class = noise.noisyImage(path, file_name, color, 0.5, 0.1, 0.2, 0.3, 10, 20)
    noise_class.all_noise()
    
    im = noise_class.Ioriginal
    im_noise= noise_class.Isuppr
    
    denoise_class = denoisedImage(im_noise, im, color)
    #denoise_class.show(im_noise, "noisy image")
    denoise_class.all_show()