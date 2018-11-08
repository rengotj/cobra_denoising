# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:16:52 2018
@author: juliette rengot
"""

import numpy as np

import noise
import denoise

class eval_denoising :
    def __init__(self, I1, I2,   # I1 and I2 are the two images to compare
                 PSNR_peak=255):     # default value for PSNR
        self.I1 = I1    #result
        self.I2 = I2    #objective
        self.Idiff = I2 - I1
        
        self.euclidian_distance = None
        
        self.peak = PSNR_peak
        self.PSNR = None
        
        self.RMSE = None
      
    def compute_euclidian_distance(self):
        """
        Compute euclidian distance between two images
        """
        self.euclidian_distance = np.linalg.norm(self.I1 - self.I2)
        return()
    
    def compute_PSNR(self):
        """
        Computes the PSNR 'metric' between two images assumed to be in the range [0,1]
        """
        x = ((np.array(self.I1).squeeze() - np.array(self.I2).squeeze()).flatten())
        self.PSNR = 10*np.log10(self.peak**2 / np.mean(x**2))
        return ()
    
    def compute_RMSE(self):
        """
        Computes the RMSE 'metric' between two images
        """
        self.RMSE = np.sqrt(((self.I1 - self.I2) ** 2).mean())
        return ()
    
    def all_evaluate(self):
        """
        Compute and display all available results
        """
        self.compute_euclidian_distance()
        print("Euclidian distance : ", self.euclidian_distance)
        self.compute_PSNR()
        print("PSNR : ", self.PSNR)
        self.compute_RMSE()
        print("RMSE : ", self.RMSE)
        return()
    
if (__name__ == "__main__"):
    path = "C://Users//juliette//Desktop//enpc//3A//Graphs_in_Machine_Learning//projet//images//"
    file_name ="lena.png"
    
    noise_class = noise.noisyImage(path, file_name)
    noise_class.all_noise()
    
    for i in range(noise_class.method_nb) :
        im = noise_class.Ioriginal
        im_noise = noise_class.Ilist[i]
        denoise_class = denoise.denoisedImage(im_noise, im)
        denoise_class.all_denoise()    
        
        for j in range(denoise_class.method_nb) :
            print("noise method : ", i, " | denoise method : ", j)
            im_denoise = denoise_class.Ilist[j]
            evaluation = eval_denoising(im_denoise, im)
            evaluation.all_evaluate()
            
            denoise_class.show(im, 'Original image')
            denoise_class.show(im_denoise, 'Denoised image')
            denoise_class.show(evaluation.Idiff, 'Difference')
            
            