# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:16:52 2018
@author: juliette rengot
"""

import numpy as np
from sklearn.metrics import mean_absolute_error
from sewar.full_ref import uqi, vifp

import noise
import denoise

class eval_denoising :
    def __init__(self, I1, I2,   # I1 and I2 are the two images to compare
                 I3=None,        #Image bruitée
                 PSNR_peak=255):     # default value for PSNR
        self.I1 = I1    #result
        self.I2 = I2    #objective
        self.Idiff = I2 - I1
        
        if I3 != None :
            self.I_noise_only = I3 - I1
        else:
            self.I_noise_only = None
        
        self.euclidian_distance = None

        self.MAE = None
        
        self.RMSE = None
        
        self.peak = PSNR_peak
        self.PSNR = None
        
        self.UQI = None
        
        self.VIF = None
      
    def compute_euclidian_distance(self):
        """
        Compute euclidian distance between two images
        """
        self.euclidian_distance = np.linalg.norm(self.I1 - self.I2)
        return()
  
    def compute_MAE(self):
        """
        Computes the Mean Absolute Error between two images
        """
        self.MAE = np.mean(np.abs(self.I1 - self.I2))
        return()
    
    def compute_RMSE(self):
        """
        Computes the Root Mean Square Error between two images
        """
        self.RMSE = np.sqrt(((self.I1 - self.I2) ** 2).mean())
        return ()
    
    def compute_PSNR(self):
        """
        Computes the Peak Signal to Noise Ratio between two images
        """
        img_1 = np.array(self.I1/np.max(self.I1))
        img_2 = np.array(self.I2/np.max(self.I2))
        x = (img_1.squeeze() - img_2.squeeze()).flatten()
        self.PSNR = 10*np.log10(self.peak**2 / np.mean(x**2))
        return ()
    
    def compute_UQI(self) :
        """
        Compute Universal Quality Image Index between two images
        """
        self.UQI = uqi(self.I1, self.I2)
        return ()

    def compute_VIF(self) :
        """
        Compute Visual Information Fidelity between two images
        """
        self.VIF = vifp(self.I1, self.I2)
        return ()
    
    def all_evaluate(self):
        """
        Compute and display all available results
        """
        self.compute_euclidian_distance()
        print("Euclidian distance : ", self.euclidian_distance)
        self.compute_MAE()
        print("MAE : ", self.MAE)
        self.compute_RMSE()
        print("RMSE : ", self.RMSE)
        self.compute_PSNR()
        print("PSNR : ", self.PSNR)
        self.compute_UQI()
        print("UQI : ", self.UQI)
        self.compute_VIF()
        print("VIF : ", self.VIF)
        return()
    
if (__name__ == "__main__"):
    path = "images//"
    file_name ="lena.png"
    color = 1
    
    noise_class = noise.noisyImage(path, file_name, color)
    noise_class.multi_noise()
    
    for i in range(noise_class.method_nb) :
        im = noise_class.Ioriginal
        im_noise = noise_class.Ilist[i]
        denoise_class = denoise.denoisedImage(im_noise, im, color)
        denoise_class.all_denoise()    
        
        for j in range(denoise_class.method_nb) :
            print("noise method : ", i, " | denoise method : ", j)
            im_denoise = denoise_class.Ilist[j]
            evaluation = eval_denoising(im_denoise, im)
            evaluation.all_evaluate()
            
            denoise_class.show(im, 'Original image')
            denoise_class.show(im_denoise, 'Denoised image')
            denoise_class.show(evaluation.Idiff, 'Difference')
            
            