# MATLAB Training Course: Deep Learning with AlexNet

Author: Andrea Murphy

Spring 2020

**Initial setup in MATLAB:**

Check if the required package is installed-



       >> alexnet
          ans =
          SeriesNetwork with properties:
          Layers: [25×1 nnet.cnn.layer.Layer]

If the required support package is installed, then the function returns a SeriesNetwork object

Next load a pre-trained AlexNet network-

    >> net = alexnet;

Then open Deep Network Designer

    >> deepNetworkDesigner

## Edit Network for Transfer Learning

To retrain a pre-trained network to classify new images, replace the final layers with new layers adapted to the new data set. You must change the number of classes to match your data.

Drag a new fullyConnectedLayer from the Layer Library onto the canvas

Edit the OutputSize to the number of classes in the new data to 5

Edit learning rates to learn faster in the new layers than in the transferred layers. Set WeightLearnRateFactor and BiasLearnRateFactor to 10

Delete the original layer and connect up your new layer instead

Make sure to check the Network is working

To make sure your edited network is ready for training, click Analyze and ensure that the Deep Learning Network Analyzer reports zero errors

Export Network for Training

Return to the Deep Network Designer and click Export. The Deep Network Designer exports the network to a new variable called layers_1 containing the edited network layers. You can now supply the layer variable to the trainNetwork function

The variable net represents a deep convolutional network. You can inspect the layers of the network by referencing the Layers property of the variable, using variable.

Property indexing:

    ly = net.Layers;

The variable ly is an array of network layers. You can inspect an individual layer by indexing into ly with regular MATLAB array indexing:

    layer3 = ly(3)

Each layer of the network has properties relevant to that type of layer. An important property for an input layer is InputSize, which is the size (dimensions) of images the network expects as input.

    insz = inlayer.InputSize;

Extract the last (output) layer of the network into a variable called outlayer

outlayer = ly(end);

The Classes property of an output layer gives the names of the categories the network is trained to predict.

    categorynames = outlayer.Classes;

## Creating and Labeling Images for a Datastore

By default, imageDatastore looks for image files within the given folder. You can use the 'IncludeSubfolders' option to look for images within subfolders of the given folder.

    ds = imageDatastore('folder name','IncludeSubfolders',true)

    flwrds = imageDatastore('Flowers','IncludeSubfolders',true)

## Label Images in a Datastore

    load pathToImages

    flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true);

    flowernames = flwrds.Labels

    flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,

    'LabelSource','foldernames')

    flowernames = flwrds.Labels

## Split Data in Datastore for Training and Testing

    load pathToImages

    flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,

    'LabelSource','foldernames')

    [flwrTrain,flwrTest] = splitEachLabel(flwrds, 0.6)

    (ds , p)

The proportion p (a value from 0 to 1) indicates the proportion of images with each label from ds that should be contained in ds1. The remaining files are assigned to ds2.

You can also randomly shuffle the files

    [flwrTrain,flwrTest] = splitEachLabel(flwrds, 0.8, "randomized")

You can also use a “validation” set to monitor the performance of the network during training. In this case, you can split your data into three sets: one for training, one for validation during training, and one for a separate test of the final result. Try using splitEachLabel to divide the Flowers images into multiple sets. Give multiple values of p or n as inputs and ask for the appropriate number of datastores as outputs.

## Modify Network Layers

The fullyConnectedLayer function creates a new fully connected layer, with a given number of neurons.

    fclayer = fullyConnectedLayer(n)

You can use standard array indexing to modify individual elements of an array of layers.

    mylayers(n) = mynewlayer

You can use the classificationLayer function to create a new output layer for an image classification network.

    cl = classificationLayer

You can create new layers and overwrite an existing layer with the new layer in a single command:

    mylayers(n) = classificationLayer

## Set Training Options

You can see the available options for a chosen training algorithm by using the trainingOptions function.

    opts = trainingOptions('sgdm')

This creates a variable opts that contains the default options for the training algorithm

“stochastic gradient descent with momentum”

## Changing the learning rate

Often you will first want to try training with most of the options left at their default values. However, when performing transfer learning, you will typically want to start with the InitialLearnRate set to a smaller value than the default of 0.01.

The learning rate controls how aggressively the algorithm changes the network weights. The goal of transfer learning is to fine-tune an existing network, so you typically want to change the weights less aggressively than when training from scratch.

    opts = trainingOptions('sgdm','InitialLearnRate',0.001)

## Transfer Learning with AlexNet

To perform transfer learning, you need to create three components:

 1. An array of layers representing the network architecture. For transfer learning, this is created by modifying a preexisting network such as AlexNet.
 2. Images with known labels to be used as training data. This is typically provided as a datastore.
 3. A variable containing the options that control the behavior of the training algorithm.

These three components are provided as the inputs to the  trainNetwork  function which returns the trained network as output.

You should test the performance of the newly trained network. If it is not adequate, typically you should try adjusting some of the training options and retraining.

Mini-Batch

At each iteration, a subset of the training images, known as a mini-batch, is used to update the weights. Each iteration uses a different mini-batch. Once the whole training set has been used, that's known as an epoch.

The maximum number of epochs (MaxEpochs) and the size of the mini-batches (MiniBatchSize) are parameters you can set in the training algorithm options.

Note that the loss and accuracy reported during training are for the mini-batch being used in the current iteration

By default, the images are shuffled once prior to being divided into mini-batches. You can control this behavior with the Shuffle option.

    >>> net = trainNetwork(data, layers, options)

## Transfer Learning Example Script

    % Get training images

    flower_ds = imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');

    [trainImgs,testImgs] = splitEachLabel(flower_ds,0.6);

    numClasses = numel(categories(flower_ds.Labels));

    % Create a network by modifying AlexNet

    net = alexnet;

    layers = net.Layers;

    layers(end-2) = fullyConnectedLayer(numClasses);

    layers(end) = classificationLayer;

    % Set training algorithm options

    options = trainingOptions('sgdm','InitialLearnRate', 0.001);

    % Perform training

    [flowernet,info] = trainNetwork(trainImgs, layers, options);

    Use trained network to classify test images

    testpreds = classify(flowernet,testImgs);

    % Evaluate Performance

    load pathToImages

    load trainedFlowerNetwork flowernet info

    plot(info.TrainingLoss)

    dsflowers = imageDatastore(pathToImages,'IncludeSubfolders',true,

    'LabelSource','foldernames');

    [trainImgs,testImgs] = splitEachLabel(dsflowers,0.98);

    flwrPreds = classify(flowernet,testImgs)

## Test Performance

You can determine how many of the test images the network correctly classified by comparing the predicted classification with the known classification. The known classifications are stored in the Labels property of the datastore.

    % Setup the Workspace

    load pathToImages.mat

    pathToImages

    flwrds = imageDatastore(pathToImages,'IncludeSubfolders',true,

    'LabelSource','foldernames');

    [trainImgs,testImgs] = splitEachLabel(flwrds,0.98);

    load trainedFlowerNetwork flwrPreds

    Extract Lables

    flwrActual = testImgs.Labels

### Count correct

You can use logical comparison and the nnz function to determine the number of elements of two arrays that match:

    numequal = nnz(a == b)

    numCorrect = nnz(flwrPreds == flwrActual)

### Calculate fraction correct

You can use the numel function to determine the number of elements in flwrPreds

    fracCorrect = numCorrect/numel(flwrPreds)

Loss and accuracy give overall measures of the network's performance. But it can be informative to investigate how the network performs on the different image classes

### Display confusion matrix

The confusionchart function calculates and displays the confusion matrix for the predicted classifications.

    confusionchart(knownclass,predictedclass)

The (j,k) element of the confusion matrix is a count of how many images from class j the network predicted to be in class k.
Diagonal elements represent correct classifications, off-diagonal elements represent misclassifications

Display Confusion Matrix:

      confusionchart(testImgs.Labels,flwrPreds)

# Transfer Learning Function Summary

### Create a network
[AlexNet](https://www.mathworks.com/help/deeplearning/ref/alexnet.html;jsessionid=5eceb89473f4dbb3c6872ad483c3)
Pre-trained network named "AlexNet"

[Supported Networks](https://www.mathworks.com/solutions/deep-learning/models.html)
A list of other MATLAB compatible pre-trained networks

 [classificationLayer](https://www.mathworks.com/help/deeplearning/ref/classificationlayer.html)
 Create new output layer for a classification network

### Get training images
[imageDatastore](http://www.mathworks.com/help/matlab/ref/matlab.io.datastore.imagedatastore.html)

Create datastore reference to image files

[augmentedImageDatastore](https://www.mathworks.com/help/deeplearning/ref/augmentedimagedatastore.html)

Preprocess a collection of image files

[splitEachLabel](http://www.mathworks.com/help/matlab/ref/datastore.spliteachlabel.html)

Divide datastore into multiple datastores

### Set training algorithm options

[trainingOptions](http://www.mathworks.com/help/nnet/ref/trainingoptions.html)
Create variable containing training algorithm options

###  Perform training


[trainNetwork](http://www.mathworks.com/help/nnet/ref/trainnetwork.html)

Perform training

### Use trained network to perform classifications

[classify](https://www.mathworks.com/help/nnet/ref/classify.html)

Obtain trained network's classifications of input images

### Evaluate trained network

[nnz](http://www.mathworks.com/help/matlab/ref/nnz.html)

Count non-zero elements in an array

[confusionchart](http://www.mathworks.com/help/nnet/ref/confusionchart.html)

Calculate confusion matrix

[heatmap](http://www.mathworks.com/help/matlab/ref/heatmap.html)

Visualize confusion matrix as a heatmap
