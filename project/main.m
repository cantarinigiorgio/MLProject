clear all;

[label_names,trainingImgs,trainingLabels] = load_cifar();
%%%%%%%%PRE-PROCESSING
rgbtrain = atorgb(trainingImgs);
grayscaletrain = rgbtog(rgbtrain);
labtrain = rgbtolab(rgbtrain);
% filename = 'data.mat';
% save(filename);

% load('data.mat');

%6000 images for each category GRAYSCALE
X1 = category("bird",label_names,grayscaletrain,trainingLabels);
X2 = category("airplane",label_names,grayscaletrain,trainingLabels);

%6000 images for each category RGB
X3 = category("bird",label_names,rgbtrain,trainingLabels);
X4 = category("airplane",label_names,rgbtrain,trainingLabels);
    
%6000 images for each category LAB
X5 = category("bird",label_names,labtrain,trainingLabels);
X6 = category("airplane",label_names,labtrain,trainingLabels);

% Select a subset of training and test set GRAYSCALE
X1trainval = (X1(1:6000,:,:));
X2trainval = (X2(1:6000,:,:));

X1test = double(X1(5001:6000,:,:));
X2test = double(X2(5001:6000,:,:));

% Select a subset of training and test set RGB
X3trainval = double(X3(1:6000,:,:,:));
X4trainval = double(X4(1:6000,:,:,:));

X3test = double(X3(5001:6000,:,:,:));
X4test = double(X4(5001:6000,:,:,:));

% Select a subset of training and test set LAB
X5trainval = double(X5(1:6000,:,:,:));
X6trainval = double(X6(1:6000,:,:,:));

X5test = double(X5(5001:6000,:,:,:));
X6test = double(X6(5001:6000,:,:,:));


X = reshape(cat(1,X1trainval,X2trainval),[],1024);%1:6000 first category, 6001:12000 second category
Xtest = reshape(cat(1,X1test,X2test),[],1024);%1:1000 first category, 1001:2000 second category

Xrgb = reshape(cat(1,X3trainval,X4trainval),[],3072);%1:6000 first category, 6001:12000 second category
Xrgbtest = reshape(cat(1,X3test,X4test),[],3072);%1:1000 first category, 1001:2000 second category
    
Xlab = reshape(cat(1,X5trainval,X6trainval),[],3072);%1:6000 first category, 6001:12000 second category
Xlabtest = reshape(cat(1,X5test,X6test),[],3072);%1:1000 first category, 1001:2000 second category
    
clr = "gs";
holdoutCV(X,clr);

clr = "rgb";
holdoutCV(Xrgb,clr);

clr = "lab";
holdoutCV(Xlab,clr);

