#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:48:00 2020

@author: gad, jon, jamie
"""

import pandas as pd
import numpy as numpy
import math

mnistDataFrame = pd.read_csv('../Data/mnist_train.csv', sep=',')

# =============================================================================
# Closure Example
# =============================================================================
def getCounter():
    count = 0 
    def useCount(x):
        return x + count
    def setCount(x):
        nonlocal count
        count = x
    return Bunch(useCount=useCount,setCount=setCount)


class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def perceptron(x,bias=0):
    return 1 if x > bias else 0

def sigmoid_neuron(x, bias=0):
    return 1/(1+ numpy.exp(-x-bias))

def sigmoid_neuronGrad(x):
    return numpy.exp(-x)/numpy.square((1+numpy.exp(-x)))


def getLabelsAndImagesInNumpy(dataframe):
    numLabels = len(dataframe)
    labels = dataframe.iloc[0:numLabels, :1].to_numpy()
    images = dataframe.iloc[0:numLabels, 1:].to_numpy()
    labelVectors = numpy.zeros( (numLabels,10), dtype=object)
    for i, label in enumerate(labels):
        labelVectors[i][label]=1
    
    return [labelVectors,images]
    

def getTrainAndTest(dataframe, percentTrain=0.025):
    divider = math.floor(len(dataframe)*percentTrain)
    train_df = dataframe.iloc[0:divider]
    test_df = dataframe.iloc[divider:len(dataframe)]
    return [train_df,test_df]

def hiddenLayer(layerSize=100, length=784, func=sigmoid_neuron ):
    # weights matrix as a dataframe 
    layerWeights = numpy.random.normal(0, 1, size=(layerSize,length))
    def activationFunction(inputValue, weights, bias): 
        return func(numpy.dot(weights, inputValue ), bias)
    ## derivative of sigmoidNeuron = (SN)*(1-SN)
    def gradientActivationFunction( weights, bias): 
        return func( weights, bias )* (1-func(weights,bias))
    def setWeights(newWeights):
        nonlocal layerWeights 
        
        layerWeights = newWeights
        return
    def weights():
        nonlocal layerWeights
        return layerWeights
    return Bunch(activationFunction=activationFunction, weights=weights, setWeights=setWeights, gradientActivationFunction=gradientActivationFunction)

def getBatch(batch, labels, images):
    batchLabels = numpy.array_split(labels, batch)
    batchImages = numpy.array_split(images, batch)
    return [batchLabels, batchImages]

def batchTraining(epoch, batchSize, model, labels, images):
    batchLabels, batchImages = getBatch(batchSize, labels, images)
    
    for i in range(epoch):
        for j in range(batchSize):
            gradientDescent(model, batchLabels[j], batchImages[j])
    return model

# =============================================================================
#    General idea for gradient descent 
# =============================================================================
# def gradientDescent(f, gradF, point, stepSize, threshold):
#     thresholdArray = numpy.full(point.shape, threshold)
#     value = f(point)
#     newPoint = point - stepSize * gradF(point)
#     newValue = f(newPoint)
#          
#     if numpy.linalg.norm(newValue - value) < threshold:
#          return value
#     return gradientDescent(f, gradF, newPoint, stepSize, threshold)
# =============================================================================
def gradientDescent(model,labels, images, stepSize=.01, threshold=0.001):
    approxLabels = numpy.ones((len(images), 10), dtype=object)
    newApproxLabels = numpy.zeros((len(images),10), dtype=object)
    print('***newbatch')
    newWeights = numpy.zeros( (len(images), len(model)), dtype=object)
    iter =0
    bias = stepSize* numpy.linalg.norm(approxLabels - labels )/len(images)
    while(abs(numpy.linalg.norm(approxLabels - labels) -  numpy.linalg.norm(newApproxLabels - labels)) > threshold and iter <30 ):
 #      print('***** current distance of batch *****')
        #print('*Approx',numpy.linalg.norm(approxLabels - labels))
       # print('*newApprox',numpy.linalg.norm(newApproxLabels - labels))
        #print('*diff',abs(numpy.linalg.norm(approxLabels - labels) - numpy.linalg.norm(newApproxLabels - labels)))

 #      print('*************************************')

        for imageIndex, image in enumerate(images):
            layerValues = numpy.zeros((len(model)+1), dtype=object)
            
            ##used to get value of layer with weights from gradient
            newLayerValues = numpy.zeros((len(model)+1), dtype=object)
            
            ## init values
            layerValues[0] = numpy.transpose(image)


            # value = f(point)
            for layerIndex,layer in enumerate(model):
                layerValues[layerIndex+1] = layer.activationFunction(layerValues[layerIndex], layer.weights(), bias)

               
                gradStep = stepSize * layer.gradientActivationFunction(layer.weights(), bias) 
                newWeights[imageIndex][layerIndex] = layer.weights() - gradStep 
                
            approxLabels[imageIndex] = layerValues[-1]
  
            
                
            newLayerValues[0] = numpy.transpose(image)
            
            
            for layerIndex,layer in enumerate(model):
                newLayerValues[layerIndex+1] = layer.activationFunction(newLayerValues[layerIndex], newWeights[imageIndex][layerIndex], bias)
            
            newApproxLabels[imageIndex] = newLayerValues[-1]
            bias = stepSize* numpy.linalg.norm(approxLabels - labels )/len(images)
        updateWeights(model, averageArray(newWeights, len(model)) )        
        iter+=1
    print('*finaldiff',abs(numpy.linalg.norm(approxLabels - labels) - numpy.linalg.norm(newApproxLabels - labels)))
    return 

def averageArray(X, size ):
    average = numpy.zeros(size , dtype=object)

    for j in range(size):
        for x in X:  
            average[j] +=  x[j]
    return average/ len(X)

def updateWeights(model, newWeights):
    for layerIndex, layer in enumerate(model):
        layer.setWeights(newWeights[layerIndex])
        
        

def testModel(model, images):
    approxLabels = numpy.zeros((len(images),10), dtype=object)

    for imageIndex, image in enumerate(images):
        layerOutput = image
        for layer in model:
            layerOutput = layer.activationFunction(layerOutput, 0)
        approxLabels[imageIndex] = layerOutput
    return approxLabels

def compareResults(approxLabels, labels):
    print('******', approxLabels.shape, labels.shape)
    print(numpy.linalg.norm( approxLabels - labels ))
    print('******')
    
def feed_forward_network(trainingLabels, trainingImages, epochs=1, batchSize=10, learningRate=3.0):      
    model= [ hiddenLayer(100, 784, sigmoid_neuron), hiddenLayer(10,100, sigmoid_neuron) ]
    return batchTraining(epochs, batchSize, model, trainingLabels, trainingImages)        

trainingDataFrame, testDataFrame = getTrainAndTest(mnistDataFrame)
trainingLabels, trainingImages = getLabelsAndImagesInNumpy(trainingDataFrame)



trainedModel = feed_forward_network(trainingLabels, trainingImages)

testLabels, testImages = getLabelsAndImagesInNumpy(testDataFrame)

approxLabels = testModel(trainedModel, testImages)

compareResults(approxLabels, testLabels)