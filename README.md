This project has been realised by Juliette Rengot, from November, 2018 to January, 2019, within the scope of MVA master (Graph in Machine Learning class). Benjamin Guedj has supervised this work. 

This project proposes a new approach for image denoising. It aims at using several classical denoising methods to get predictions for each noisy pixel and to use the aggreation scheme of pycobra librairy to generate a new prediction.

The folder images contains some images to train and test the model.

The code is contained into 4 files :
* noise.py : a module to add artificial noises to images (Gaussian, Poisson, salt-and-pepper, speckle, random suppression, multi) 
* denoise.py : a module to denoise images with classical methods (Gaussian filter, Median filter, Bilateral filter, non-local means, TV-Chambolle, Richardson-Lucy deconvolution, inpainting)
* evalute.py : a module to evaluate denoising quality (RMSE, PSNR)
* denoising_cobra.py : the main file to create a cobra model and use it for denoising task

Two notebooks are available :
* demo.ipyng : it shows an example of cobra denoising
* demo_configuration_median.ipyng : it shows an example of how to use the method for median filter configuration
