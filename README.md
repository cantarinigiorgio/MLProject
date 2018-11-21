# Machine Learning Project

# Pre-processing and features extraction for image classification

Before starting you have to download CIFAR-10 dataset (Matlab version) from (https://www.cs.toronto.edu/~kriz/cifar.html).
Then you unzip the folder and put data_batch_1.mat , data_batch_2.mat , data_batch_3.mat , data_batch_4.mat , data_batch_5.mat into cifar-10 folder of this project (the sizes of this file.mat are greater than the maximum file size accepted by github).

After that the function load_cifar() in main.m load images and labels and three functions (atorgb, rgbtog, rgbtolab) allows to have three different matrices: one for images in RGB color model, one for images in grayscale color model and the last one for images in LAB color model.

Changing the first parameter given to the function category in main.m you could change category of the images to use (category are: airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck).

K-fold Cross-Validation is done by the function holdoutCV.m that evaluates for RGB, grayscale and LAB images the percentage of error
 for each feature extracted thanks to SVM algorithm and at least plot results in three graphics (one for each color model).
