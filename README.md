# Project Title:

## To build a handwritten digit classifier using Support Vector Machines

### In this project, we will use Support Vector Machines (SVMs) to build a handwritten digit classifier. We will be solving the SVM optimization problem using the Pegasos algorithm and also use a customized solver known as LIBSVM.

### There are separate training and test example files. Each row in the (train/test) data file corresponds to an image of size 28x28, represented as a vector of grayscale pixel intensities followed by the label associated with the image. Every column represents a feature where the feature value denotes the grayscale value (0-255) of the corresponding pixel in the image. There is a feature for every pixel in the image. Last column gives the corresponding label. This data is a subset of the original MNIST dataset available at [this link](https://drive.google.com/file/d/1OgQOTgODBKCuYX1B3E1gDmhjbOOcq4Wq/view).

* Used the mini-batch version of Pegasos algorithm and used a batch size of 100 in SGD implementation.
* Extended the SVM formulation for a binary classification problem. In order to extend this to the multi-class setting, we train a model on each pair classes to get k:2 combinations of classifiers ,k being the number of classes. During prediction time, we output the classifier which has the maximum number of wins among all the classifiers. You can read more about one-vs-one classifier setting at the [following link](https://en.wikipedia.org/wiki/Multiclass_classification). Classified the given MNIST dataset and reported train and test accuracy for C = 1:0. In case of ties, choosen the label with the highest digit value.
