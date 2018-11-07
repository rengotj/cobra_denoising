# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:09:25 2018
@author: juliette rengot
"""

import numpy as np
import os
import cv2

from pycobra.cobra import Cobra

import noise
import denoise


class machine:
    def __init__(self, name, num_denoised_method):
        self.name = name
        self.num_denoised_method = num_denoised_method
        
    def predict(self, Inoisy) :
        #print("Predict in machine :", self.name)
        Inoisy = np.array(Inoisy)
        if self.name == 'bilateral' :
            denoise_class = denoise.denoisedImage(Inoisy)
            denoise_class.bilateral()
            image_denoised = denoise_class.Ibilateral            
        elif self.name == 'nlmeans' :
            denoise_class = denoise.denoisedImage(Inoisy)
            denoise_class.NLmeans()
            image_denoised = denoise_class.Inlmeans
        elif self.name == 'gauss' :
            denoise_class = denoise.denoisedImage(Inoisy)
            denoise_class.gauss()
            image_denoised = denoise_class.gauss            
        elif self.name == 'median' :
            denoise_class = denoise.denoisedImage(Inoisy)
            denoise_class.median()
            image_denoised = denoise_class.Imedian
        return(image_denoised)

  
      
def load_training_data(path):
    """ Load images, apply noise methods on it and
    return three different training sets.
    
    INPUT
    path : where are the training images
    
    OUTPUT
    Xtrain : list of pixel of noisy images
    Xtrain1 : list of pixel of noisy images +  gaussian noise
    Xtrain2 : list of pixel of noisy images - gaussian noise
    Ytrain : list of pixel of original images
    """
    Xtrain  = []
    Xtrain1 = []
    Xtrain2 = []
    Ytrain  = []
    for file in os.listdir(path):
        noise_class = noise.noisyImage(path, file)
        noise_class.all_noise()
        for i in range(noise_class.method_nb):
            Ytrain += list(noise_class.Ioriginal.reshape(-1,1))
            Inoisy = noise_class.Ilist[i]
            
            sigma = 0.1 # gaussian noise
            eps = np.random.normal(0, sigma, noise_class.shape)
            y1 = Inoisy + sigma * eps
            y1 = (y1-y1.min())/(y1.max()-y1.min())
            y2 = Inoisy - sigma * eps
            y2 = (y2-y2.min())/(y2.max()-y2.min())
            
            Xtrain  += list(Inoisy.reshape(-1,1))
            Xtrain1 += list(y1.reshape(-1,1))
            Xtrain2 += list(y2.reshape(-1,1))
    return(Xtrain, Xtrain1, Xtrain2, Ytrain)

def denoise_cobra(im_noise, train_path, verbose=False) :
    """
    Denoise an noisy image using cobra aggregation
    
    INPUT :
    im_noise : noisy image
    train_path : where are the training images
    verbose : print or not information during the training
        
    OUTPUT :
    Y : denoised image
    """
    #cobra parameters
    Alpha = 1 #proportion parameter
    Lambda = 0.1 # confidence parameter
    M = 2 # number of preliminary estimators
    
    print("Training cobra model...")
    Xtrain, Xtrain1, Xtrain2, Ytrain = load_training_data(train_path)
    cobra = Cobra(epsilon=Lambda, machines=M) # create a cobra machine
    cobra.fit(Xtrain, Ytrain, default=False, X_k=Xtrain1, X_l=Xtrain2, y_k=Ytrain, y_l=Ytrain) # fit the cobra machine with our data

    print("Loading machines...")
    #cobra.load_machine('bilateral', machine('bilateral',0))
    #cobra.load_machine('nlmeans', machine('nlmeans',1))
    cobra.load_machine('gauss', machine('gauss',2))
    cobra.load_machine('median', machine('median',3))

    print("Loadin machine predictions...")
    cobra.load_machine_predictions() #agregate
    if verbose :
        cobra.machine_predictions_
    
    print("Image denoising...")
#    Y = cobra.pred(list(im_noise.reshape(-1,1)), Alpha)
    Y = np.zeros(noise_class.shape)
    for i in range(noise_class.shape[0]):
        for j in range(noise_class.shape[1]):
            Y[i,j] = cobra.pred([im_noise[i,j]], Alpha)
            if verbose :
                print('noisy pixel : ',im_noise[i,j])
                print('denoised : ', Y[i,j])
                
    if verbose :
        print(Y)  
        cv2.imshow('Denoised image (cobra)', np.array(Y))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return(Y)

    
#def eval_cobra(noise_class, train_path):
#    """
#    Compute euclidian distance between the original image (target) and the denoised image obtained with cogra aggregation
#    """
#    score = []
#    for i in range(noise_class.method_nb) :
#        im_noise = noise_class.Ilist[i]
#        im_denoised = denoise_cobra(im_noise, train_path)
#        score.append(np.linalg.norm(noise_class.Ioriginal - im_denoised))
#    return(score)
        
    
    
if (__name__ == "__main__"):
    path = "C://Users//juliette//Desktop//enpc//3A//Graphs_in_Machine_Learning//projet//images//"
    file_name ="sub_lena.png"
    
    noise_class = noise.noisyImage(path,file_name)
    noise_class.all_noise()
    
    im_noise = noise_class.Ipoiss

    #cobra denoising
    Y = denoise_cobra(im_noise, path+"//train//", True)
    cv2.imwrite('denoised_cobra.png', Y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
