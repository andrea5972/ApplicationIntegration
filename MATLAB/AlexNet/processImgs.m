% AlexNet

% Authors: Andrea Murphy
% Date: Spring 2020
% DESC: Using the pre-trained network AlexNet
% Process Images for Classification

img = imread('seashore.jpg');
imshow(img)

% View image size
sz = size(img);

% import AlexNet and find the expected size 
net = alexnet;
inlayer = net.Layers(1);
insz = inlayer.InputSize;

% Resize image and display
img = imresize(img,[227 227]);
imshow(img)