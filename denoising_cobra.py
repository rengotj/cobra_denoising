# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 14:09:25 2018
@author: juliette rengot
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import os

from pycobra.cobra import Cobra
from pycobra.diagnostics import Diagnostics
from pycobra.visualisation import Visualisation

import noise
import denoise
import evaluation


def list_neighbours(I,x,y,patch_size):
    """
    INPUT
    I : image
    x,y : coordinates of the central pixel of the consider patch
    k : patch of size (2*k+1)(2*k+1)
    
    OUPUT
    L : list of I(x',y') where (x',y') is a pixel of the patch
    """
    assert(0<=x-patch_size)
    assert(x+patch_size<I.shape[0])
    assert(0<=y-patch_size)
    assert(y+patch_size<I.shape[1])
    
    L = []
    for x1 in range(x-patch_size, x+patch_size+1):
        for y1 in range(y-patch_size, y+patch_size+1):
            L.append(I[x1,y1])
    return(L)

def load_training_data(path, noise_kind, k=0):
    """ Load images, apply noise methods on it and
    return three different training sets.
    
    INPUT
    path : where are the training images
    k : patch of size (2*k+1)(2*k+1) => (2*k+1)(2*k+1) features
    
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
        noise_class = noise.noisyImage(path, file, 0, 0.5, 0.1, 0.2, 0.3, 5, 10)
        noise_class.multi_noise()
        for i in noise_kind:
            Ytrain += [[noise_class.Ioriginal[x, y]] for x in range(k,noise_class.shape[0]-k) for y in range(k,noise_class.shape[1]-k)]
            Inoisy = noise_class.Ilist[i]
            
            sigma = 0.1 # gaussian noise
            eps = np.random.normal(0, sigma, noise_class.shape)
            y1 = Inoisy + sigma * eps
            y1 = (y1-y1.min())/(y1.max()-y1.min())
            y2 = Inoisy - sigma * eps
            y2 = (y2-y2.min())/(y2.max()-y2.min())
            
            Xtrain  += [list_neighbours(Inoisy, x, y, k) for x in range(k,noise_class.shape[0]-k) for y in range(k,noise_class.shape[1]-k)]
            Xtrain1 += [list_neighbours(y1, x, y, k) for x in range(k,noise_class.shape[0]-k) for y in range(k,noise_class.shape[1]-k)]
            Xtrain2 += [list_neighbours(y2, x, y, k) for x in range(k,noise_class.shape[0]-k) for y in range(k,noise_class.shape[1]-k)]
    return(Xtrain, Xtrain1, Xtrain2, Ytrain)

class machine:
    def __init__(self, name, num_denoised_method, patch_size):
        self.name = name
        self.num_denoised_method = num_denoised_method
        self.patch_size = patch_size
        
    def predict(self, Inoisy) :     
        #print("Predict in machine :", self.name)
        Inoisy = np.array(Inoisy)
        Idenoised = []
        
        if len(Inoisy.shape)==1:
          iter_max = 1
        else:
          iter_max = Inoisy.shape[0]
        
        for i in range(iter_max):
          if len(Inoisy.shape)==1:
            image_noisy = Inoisy.reshape((2*self.patch_size+1, 2*self.patch_size+1))
          else:
            image_noisy = Inoisy[i].reshape((2*self.patch_size+1, 2*self.patch_size+1))
          
          if self.name == 'bilateral' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.bilateral()
              image_denoised = denoise_class.Ibilateral            
          elif self.name == 'nlmeans' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.NLmeans()
              image_denoised = denoise_class.Inlmeans
          elif self.name == 'gauss' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.gauss()
              image_denoised = denoise_class.Igauss            
          elif self.name == 'median' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.median()
              image_denoised = denoise_class.Imedian
          elif self.name == 'TVchambolle' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.TVchambolle()
              image_denoised = denoise_class.Ichambolle
          elif self.name == 'richardson_lucy' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.richardson_lucy()
              image_denoised = denoise_class.Irl
          elif self.name == 'inpainting' :
              if image_noisy.shape == (1, 1) or np.all(image_noisy==1):
                  image_denoised = image_noisy
              else :
                  denoise_class = denoise.denoisedImage(image_noisy)
                  denoise_class.inpaint()
                  image_denoised = denoise_class.Iinpaint
          elif self.name == 'ksvd' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.ksvd()
              image_denoised = denoise_class.Iksvd
          elif self.name == 'lee' :
              denoise_class = denoise.denoisedImage(image_noisy)
              denoise_class.lee()
              image_denoised = denoise_class.Ilee  
#          elif self.name == 'bm3d' :
#              denoise_class = denoise.denoisedImage(image_noisy)
#              denoise_class.bm3d()
#              image_denoised = denoise_class.Ibm3d
          else :
            print("Unknown name : ", self.name)
            return()
          
          if self.patch_size>0:
            Idenoised.append([image_denoised[self.patch_size, self.patch_size]])
          else :
            Idenoised.append([image_denoised])
        return(Idenoised)

def define_cobra_model(train_names, training_noise_kind, patch_size=1, optimi=True, verbose=False) :
    """
    Train a cobra model for denoising task
    
    INPUT :
    train_names : list containing the name of images used to train the model
    patch_size : use patch of size (2*patch_size+1)*(2*patch_size+1) as features
    verbose : print or not information during the training
        
    OUTPUT :
    cobra : trained model
    """
    #initial cobra parameters
    Alpha = 4     #how many machines must agree
    Epsilon = 0.2 # confidence parameter
    
    print("Training cobra model...")
    Xtrain, Xtrain1, Xtrain2, Ytrain = load_training_data(train_names, training_noise_kind, patch_size)
    cobra = Cobra(epsilon=Epsilon, machines=Alpha) # create a cobra machine
    #cobra.fit(Xtrain, Ytrain, default=False, X_k=Xtrain1, X_l=Xtrain2, y_k=Ytrain, y_l=Ytrain) # fit the cobra machine with our data
    cobra.fit(Xtrain, Ytrain) # fit the cobra machine with our data

    print("Loading machines...")
    cobra.load_machine('bilateral', machine('bilateral', 0, patch_size))
    cobra.load_machine('nlmeans', machine('nlmeans', 1, patch_size))
    cobra.load_machine('gauss', machine('gauss', 2, patch_size))
    cobra.load_machine('median', machine('median', 3, patch_size))
    cobra.load_machine('TVchambolle', machine('TVchambolle', 4, patch_size))
    cobra.load_machine('richardson_lucy', machine('richardson_lucy', 5, patch_size))
    cobra.load_machine('inpainting', machine('inpainting', 6, patch_size))
    cobra.load_machine('ksvd', machine('ksvd', 7, patch_size))
    cobra.load_machine('lee', machine('lee', 8, patch_size))
#    cobra.load_machine('bm3d', machine('bm3d', 9, patch_size))

    print("Loading machine predictions...")
    cobra.load_machine_predictions() #agregate
    if verbose :
        print(cobra.machine_predictions_)
   
    if optimi : 
        print("Parameter optimisation")
        cobra_diagnostics = Diagnostics(cobra, Xtrain, Ytrain)
        Epsilon_opt, MSE = cobra_diagnostics.optimal_epsilon(Xtrain, Ytrain, line_points=100, info=False)
        Alpha_opt, MSE = cobra_diagnostics.optimal_alpha(Xtrain, Ytrain, epsilon=Epsilon_opt, info=False)
        if verbose :
            print("epsilon = ", Epsilon_opt)
            print("alpha = ", Alpha_opt)

        print("Training cobra model again...")
        cobra = Cobra(epsilon=Epsilon_opt, machines=Alpha_opt)
        cobra.fit(Xtrain, Ytrain, default=False, X_k=Xtrain1, X_l=Xtrain2, y_k=Ytrain, y_l=Ytrain)
        cobra.load_machine('bilateral', machine('bilateral', 0, patch_size))
        cobra.load_machine('nlmeans', machine('nlmeans', 1, patch_size))
        cobra.load_machine('gauss', machine('gauss', 2, patch_size))
        cobra.load_machine('median', machine('median', 3, patch_size))
        cobra.load_machine('TVchambolle', machine('TVchambolle', 4, patch_size))
        cobra.load_machine('richardson_lucy', machine('richardson_lucy', 5, patch_size))
        cobra.load_machine('inpainting', machine('inpainting', 6, patch_size))
        cobra.load_machine('ksvd', machine('ksvd', 7, patch_size))
        cobra.load_machine('lee', machine('lee', 8, patch_size))
#       cobra.load_machine('bm3d', machine('bm3d', 9, patch_size))
        cobra.load_machine_predictions()
        if verbose :
            print("Loading machine predictions...")
            print(cobra.machine_predictions_)

    return(cobra, Alpha, Epsilon)

def denoise_cobra(im_noise, model, n_machines, patch_size=1, verbose=False) :
    """
    Denoise an noisy image using cobra aggregation
    
    INPUT :
    im_noise : noisy image
    model : trained cobra model
    n_machines : optimal number of machines to take into account in the aggregation
    patch_size : use patch of size (2*patch_size+1)*(2*patch_size+1) as features
    verbose : print or not information during the training
        
    OUTPUT :
    Y : denoised image
    """    
    print("Image denoising...")
    Xtest = [list_neighbours(im_noise, x, y, patch_size) for x in range(patch_size, noise_class.shape[0]-patch_size) for y in range(patch_size,noise_class.shape[1]-patch_size)]
    Y = model.predict(Xtest, n_machines)
    #Padding
    if patch_size > 0 :
      Y_sol = im_noise.copy()
      for x in range(patch_size, noise_class.shape[0]-patch_size) :
        for y in range(patch_size,noise_class.shape[1]-patch_size) : 
          Y_sol[x, y] = Y[(x-patch_size)*(noise_class.shape[1]-2*patch_size)+(y-patch_size)]
      Y = Y_sol.reshape(-1)
      
    if verbose :
        print("The denoised matrix has the following data matrix : ")
        print(Y)
        
    return(Y)      
  
if (__name__ == "__main__"):
    path = "images//"
    file_name ="lena.png"
    
    noise_class = noise.noisyImage(path, file_name, 0, 0.5, 0.1, 0.2, 0.3, 2, 2)  
    noise_class.multi_noise()
    im_noise = noise_class.Imulti
    noise_class.show(im_noise, "noisy image")
    
    testing_noise_kind = [5]
    training_noise_kind = [i for i in range(noise_class.method_nb)]
    
    # Cobra denoising
    patch = 5
    cobra_model, Alpha, Epsilon = define_cobra_model(path+"//train//", training_noise_kind, patch_size=patch, optimi=False, verbose=True)
    Y = denoise_cobra(im_noise, cobra_model, Alpha, patch_size=patch, verbose=False)
    im_denoise = np.array(Y).reshape(im_noise.shape)
       
    # Evaluation
    print("Evaluation...")
    evaluate = evaluation.eval_denoising(im_denoise, noise_class.Ioriginal)
    evaluate.all_evaluate()
   
    # Diagnostic
    Xtest = [list_neighbours(im_noise, x, y, patch) for x in range(patch, noise_class.shape[0]-patch) for y in range(patch,noise_class.shape[1]-patch)]
    Y_without_padding = Y.reshape(im_noise.shape)
    Y_without_padding = Y_without_padding[patch:-patch, patch:-patch]
    Y_without_padding = Y_without_padding.reshape(-1)
    cobra_diagnostics = Diagnostics(cobra_model, Xtest, Y_without_padding, load_MSE=True)
    print("The machine MSE are : ")
    print(cobra_diagnostics.machine_MSE)
    print("The optimal machine is : ")
    print(cobra_diagnostics.optimal_machines(Xtest, Y_without_padding)) # Find the optimal combination of machines
    
    # Display results
    print("Displaying the result...")
    noise_class.show(noise_class.Ioriginal, 'Original image')
    noise_class.show(im_denoise, 'Denoised image')
    noise_class.show(evaluation.Idiff, 'Difference')
    
    # Visualisation
    Yreal = [noise_class.Ioriginal[x,y] for x in range(patch, noise_class.shape[0]-patch) for y in range(patch, noise_class.shape[1]-patch)]
    Yreal_full = [noise_class.Ioriginal[x,y] for x in range(noise_class.shape[0]) for y in range(noise_class.shape[1])]

    plt.title("Prediction Vs Real pixel value")
    plt.scatter(Yreal_full, Y)
    plt.plot([x/100 for x in range(100)], [x/100 for x in range(100)], color='red')
    plt.xlabel('real value')
    plt.ylabel('prediction')

    # Plot the results of the machines versus the actual answers (testing space).
    visualise = Visualisation(cobra_model, np.array(Xtest), np.array(Yreal))
    visualise.plot_machines(['bilateral', 'nlmeans', 'gauss', 'median', 'TVchambolle', 'richardson_lucy', 'inpainting', 'ksvd', 'lee'])