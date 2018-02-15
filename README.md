# MLProject


Typing load('data.mat') you could load data (just preprocessed and organizeded in data.mat); data are organized into three matrices:
one for RGB images, one for Grayscale images and the last one for LAB images.
Images are taken from CIFAR-10 dataset.
Changing the first parameter given to the function category in main.m you could change category of the images to use (category are: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck).

Then I've done K-fold Cross-Validation (function holdoutCV.m) and evaluated for RGB, grayscale and LAB images the error for each feature extracted.
