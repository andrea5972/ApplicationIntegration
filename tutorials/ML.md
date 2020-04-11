# MATLAB Training Course: Machine Learning

Author: Andrea Murphy

Spring 2020

##  Overview
The Classification of handwritten letters A-Z using MATLAB online and Machine Learning techniques.

## Features

A feature is simply a value calculated from the signal, such as its duration

## View Features

The scatter function to plot the extracted features, with aspect ratio on the horizontal axis and duration on the vertical axis.

    scatter(features.AspectRatio, features.Duration)

The gscatter function makes a grouped scatter plot – that is, a scatter plot where the points are colored according to a grouping variable.

    gscatter(x,y,g)

## The Model

A classification model is a partitioning of the space of predictor variables into regions. Each region is assigned one of the output classes. In this simple example with two predictor variables, you can visualize these regions in the plane.

## Fitting the Model

An easy way to classify an observation is to use the same class as the nearest known examples. This is called a **_k-nearest neighbor_ (kNN) model**

You can fit a kNN model by passing a table of data to the fitcknn function

    mdl = fitcknn(data,"ResponseVariable");

The second input is the name of the response variable in the table (the class you want the model to predict). The output is a variable containing the fitted model.

## Making Predictions

Having built a model from the data, you can use it to classify new observations. This just requires calculating the features of the new observations and determining which region of the predictor space they are in.

The predict function determines the predicted class of new observations:

    predClass = predict(model,newdata)

The inputs are the trained model and a table of observations, with the same predictor variables as were used to train the model. The output is a categorical array of the predicted class for each observation in newdata

You can specify the value of k in a kNN model by setting the "NumNeighbors" property when calling fitcknn

    mdl = fitcknn(data,"ResponseVariable","NumNeighbors",10);

## Evaluate the Model

The table testdata includes the known class for the test observations. You can compare the known classes to the kNN model's predictions to see how well the model performs on new data.

    iscorrect = (predictions == testdata.Character);

Calculate the proportion of correct predictions by dividing the number of correct predictions by the total number of predictions.

    accuracy = sum(iscorrect)/numel(predictions)

Rather than just accuracy (the proportion of correct predictions), a commonly-used metric to evaluate a model is misclassification rate (the proportion of incorrect predictions)

    notcorrrect = (predictions ~= testdata.Character);

    misclassrate = sum(notcorrrect)/numel(predictions);

Accuracy and misclassification rate give a single value for the overall performance of the model, but it can be useful to see a more detailed breakdown of which classes the model confuses.

A confusion matrix shows the number of observations for each combination of true and predicted class.

    confusionchart(testdata.Character,predictions);

## Test Loss

The response classes are not always equally distributed in either the training or test data. Loss is a fairer measure of misclassification that incorporates the probability of each class (based on the distribution in the data).

    loss(model,testdata)

## Datastore

You can use wildcards to make a datastore to files or folders matching a particular pattern

This datastore function makes a datastore to all files containing the letter M, these files have _M_ in their name and a .txt extension:

    ds = datastore("*_M_*.txt");

You can use the read function to import the data from a file in the datastore.

    data = read(ds);

The readall function imports the data from all the files in the datastore into a single variable.

    data = readall(ds);

## Preprocessing Function

Add a custom function at the end of your script for data preprocessing. The function should take the data returned from the datastore as input, then return the transformed data as output.

    function data = scale(data)

    data.Time = (data.Time - data.Time(1))/1000;

    data.X = 1.5*data.X;

    end

To use a function as an input to another function, create a function handle by adding the @ symbol to the beginning of the function name.

    transform(ds,@scale)

A function handle is a reference to a function. Without the @ symbol, MATLAB will interpret the function name as a call to that function.

Any calculations (including the default use of functions such as mean) involving NaNs will result in NaN. This is important in machine learning, where you often have missing values in your data.

You can use the "omitnan" option to have statistical functions like mean ignore missing values.

    mean(x,"omitnan")

    function data = scale(data)

    data.Time = (data.Time - data.Time(1))/1000;

    data.X = 1.5*data.X;

    data.X = data.X - mean(data.X,"omitnan");

    data.Y = data.Y - mean(data.Y,"omitnan");

    end

## Engineering Features

### Statistical Functions

[mean](https://www.mathworks.com/help/matlab/ref/mean.html)

Arithmetic mean

[median](https://www.mathworks.com/help/matlab/ref/median.html)

Median (middle) value

[mode](https://www.mathworks.com/help/matlab/ref/mode.html)

Most frequent value

[trimmean](https://www.mathworks.com/help/stats/trimmean.html)

Trimmed mean (mean, excluding outliers)

[geomean](https://www.mathworks.com/help/stats/geomean.html)

Geometric mean

[harmean](https://www.mathworks.com/help/stats/harmmean.html)

Harmonic mean

### Measures of Spread

[range](https://www.mathworks.com/help/stats/range.html)

Range of values (largest – smallest)

[std](https://www.mathworks.com/help/matlab/ref/std.html)

Standard deviation

[var](https://www.mathworks.com/help/matlab/ref/var.html)

Variance

[mad](https://www.mathworks.com/help/stats/mad.html)

Mean absolute deviation

[iqr](https://www.mathworks.com/help/stats/prob.normaldistribution.iqr.html)

Interquartile range (75th percentile minus 25th percentile)

### Measures of Shape

[skewness](https://www.mathworks.com/help/stats/skewness.html)

Skewness (third central moment)

[kurtosis](https://www.mathworks.com/help/stats/kurtosis.html)

Kurtosis (fourth central moment)

[moment](https://www.mathworks.com/help/stats/moment.html)

## Central moment of arbitrary order

Local minima and maxima are often important features of a signal. The  islocalmin  and islocalmax functions take a signal as input and return a logical array the same length as the signal.

    idx = islocalmin(x);

    idx = islocalmax(x);

The value of idx is true whenever the corresponding value in the signal is a local minimum.

Local minima and maxima are defined by computing the prominence of each value in the signal. The prominence is a measure of how a value compares to the other values around it. You can obtain the prominence value of each point in a signal by obtaining a second output from islocalmin or islocalmax.

    [idx,p] = islocalmin(x);

    [idx,p] = islocalmax(x);

By default, islocalmin and islocalmax find points with any prominence value above 0. This means that a maximum is defined as any point that is larger than the two values on either side of it. For noisy signals you might want to consider only minima and maxima that have a prominence value above a given threshold.

    idx = islocalmin(x,"MinProminence",threshvalue)

When choosing a threshold value, note that prominence values can range from 0 to range(x).

## Calculating Derivatives

The diff function calculates the difference between successive elements of an array.

That is, if:

    y = diff(x)

    then y1= x2 - x1, y2 = x3 - x2

and so on…

Note that y will be one element shorter than x

## Calculating Correlations

The corr function calculates the linear correlation between variables

    C = corr(x,y);

If both variables contain missing data, C is NaN. You can use the "Rows" option to specify how to avoid missing values.

    C = corr(x,y,"Rows","complete");

The correlation coefficient is always between -1 and +1.

A coefficient of -1 indicates a perfect negative linear correlation

A coefficient of +1 indicates a perfect positive linear correlation

A coefficient of 0 indicates no linear correlation

To calculate the correlation between each pair of several variables, you can pass a matrix to the corr function, where each variable is a column of the matrix.

    M = [x y z];

    C = corr(M);

A **boxplot** is a simple way to visualize multiple distributions

    boxplot(x,c)

This creates a plot where the boxes represent the distribution of the values of x for each of the classes in c. If the values of x are typically significantly different for one class than another, then x is a feature that can distinguish between those classes. The more features you have that can distinguish different classes, the more likely you are to be able to build an accurate classification model from the full data set.

## Investigate Misclassifications

When making a confusion chart, you can add information about the false negative and false positive rate for each class by adding row or column summaries, respectively

    confusionchart(...,"RowSummary","row-normalized");

Use relational and logical operators (such as ==, ~=, &, and |) to identify observations to study further

    % Test the false negative for “U”

    falseneg = (testdata.Character == "U") & (predLetter ~= "U");

    % Find what is common misclassification for letter U

    fnfiles = testfiles(falseneg)

    fnpred = predLetter(falseneg)

    % Visualizes what the sample input looks like

    badU = readtable(fnfiles(4));

    plot(badU.X,badU.Y)

A parallel coordinates plot shows the value of the features (or “coordinates”) for each observation as a line

    parallelcoords(data)

To compare the feature values of different classes, use the "Group" option.

    parallelcoords(data,"Group",classes)
