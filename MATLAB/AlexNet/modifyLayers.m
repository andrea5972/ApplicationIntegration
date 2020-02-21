% AlexNet

% Authors: Andrea Murphy
% Date: Spring 2020
% DESC: Using the pre-trained network AlexNet
% Modify Network Layers

anet = alexnet;
layers = anet.Layers;

% Create new layers
fc = fullyConnectedLayer(12);

% Replace the last fully connected layer (23)
layers(23) = fc;

layers(end) = classificationLayer;


