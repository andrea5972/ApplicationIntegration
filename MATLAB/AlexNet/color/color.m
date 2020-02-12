% AlexNet

% Authors: Andrea Murphy
% Date: Spring 2020
% DESC: Using the pre-trained network AlexNet
% Resize all images in a Datastore

% Display the images in the current folder and imports AlexNet
ls *.jpg
net = alexnet;

% Create datastore
imds = imageDatastore('*.jpg');

% Create augmentedImageDatastore
auds = augmentedImageDatastore([227, 227, 3],...
imds,'ColorPreprocessing','gray2rgb');

% Classify
preds = classify(net,auds);