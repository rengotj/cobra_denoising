# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:21:07 2018
@author: juliette rengot
"""
import numpy as np
import cv2
import skimage.restoration
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
import scipy.ndimage
#import pybm3d # This library is compatible with linux. It is possible not to use it
from ksvd import ApproximateKSVD

import noise

class denoisedImage :
    def __init__(self, noisy, original=None, color=0,
                 gauss_sigma=0.8,   #default parameter for gaussian filter
                 median_size=3,     #default parameter for the median filter
                 point_spread_rl=5, #default parameter for the richardson lucy deconvolution
                 bm3d_std=40,       #default parameter for the BM3D algorithm
                 ksvd_components=32, ksvd_patch=(5, 5), #default parameter for K-SVD algorithm
                 verbose = False):  #If True, print values of denoised images when computed       
        """ Create a class gathering all denoised version of a noisy image
        PARAMETERS
        noisy : noisy image
        original : original version of the noisy image if available
        gauss_sigma : standard deviation for the gaussian filter
        """
        self.verbose = verbose
        self.str2int = {"bilateral" : 0, "nl_means" : 1, "gaussian" : 2,
                        "median" : 3, "tv_chambolle" : 4, "richardson_lucy" : 5,
                        "inpainting" : 6, "ksvd" : 7, "lee" : 8 #, "bm3d" :9
                        }
        self.int2str = {0 : "bilateral", 1 : "nl_means", 2 : "gaussian",
                        3 : "median", 4 : "tv_chambolle", 5 : "richardson_lucy",
                        6 : "inpainting", 7 : "ksvd", 8 : "lee" #, 9 : "bm3d",
                        }
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
        
#        self.bm3d_std = bm3d_std
#        self.Ibm3d = np.empty(self.shape)
        
        self.ksvd_components = ksvd_components
        self.ksvd_patch = ksvd_patch
        self.Iksvd = np.empty(self.shape)
        
        self.Ilee = np.empty(self.shape)
        
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

#    def bm3d(self):
#        """BM3D denoising algorithm"""
#        self.Ibm3d = pybm3d.bm3d.bm3d((self.Inoisy*255).astype(np.int), self.bm3d_std)
#        self.Ibm3d = np.clip(self.Ibm3d/255., 0, 1)
#        self.Ilist[self.str2int['bm3d']] = self.Ibm3d
#        if self.verbose :
#              print('BM3D :', self.Ibm3d)
#        return()
 
    def ksvd(self):
        """K-SVD denoising algorithm"""
        P = extract_patches_2d(self.Inoisy, self.ksvd_patch)
        patch_shape = P.shape
        P = P.reshape((patch_shape[0], -1))
        mean = np.mean(P, axis=1)[:, np.newaxis]
        P -= mean

        aksvd = ApproximateKSVD(n_components=self.ksvd_components)
        dico = aksvd.fit(P).components_
        reduced = (aksvd.transform(P)).dot(dico) + mean
        reduced_img = reconstruct_from_patches_2d(reduced.reshape(patch_shape), self.shape)
          
        self.Iksvd = np.clip(reduced_img, 0 ,1)
        self.Ilist[self.str2int['ksvd']] = self.Iksvd
        if self.verbose :
            print('K-SVD :', self.Iksvd)
        return()

    def lee(self):
        """ Lee filter """
        img_mean = scipy.ndimage.filters.uniform_filter(self.Inoisy, self.shape)
        img_sqr_mean = scipy.ndimage.filters.uniform_filter(self.Inoisy**2, self.shape)
        img_variance = img_sqr_mean - img_mean**2
    
        overall_variance = scipy.ndimage.measurements.variance(self.Inoisy)
    
        img_weights = img_variance / (img_variance + overall_variance)
        self.Ilee = img_mean + img_weights * (self.Inoisy - img_mean)
        self.Ilist[self.str2int['lee']] = self.Ilee
        if self.verbose :
            print('Lee filter :', self.Ilee)       
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
            print('Inpainting :', self.Iinpaint)
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
        self.ksvd()
#        self.bm3d()
        self.lee()
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
         self.show(self.Iinpaint, "Inpainting")
         self.show(self.Iksvd,"K-SVD")
#         self.show(self.Ibm3d, "BM3D")
         self.show(self.Ilee,"Lee Filter")
         return()
            
if (__name__ == "__main__"):
    path = "images//"
    file_name ="lena.png"
    color = 1
    
    noise_class = noise.noisyImage(path, file_name, color, 0.5, 0.1, 0.2, 0.3, 10, 20)
    noise_class.all_noise()
    
    im = noise_class.Ioriginal
    im_noise= noise_class.Isuppr
    
    denoise_class = denoisedImage(im_noise, im, color)
    print("noisy image to denoise")
    denoise_class.show(im_noise, "noisy image")
    print('denoising...')
    denoise_class.all_show()