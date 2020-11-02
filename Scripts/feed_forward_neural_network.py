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

def sigmoid_neuron(x, bias=1):
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
    

def getTrainAndTest(dataframe, percentTrain=0.7):
    divider = math.floor(len(dataframe)*percentTrain)
    train_df = dataframe.iloc[0:divider]
    test_df = dataframe.iloc[divider:len(dataframe)]
    return [train_df,test_df]

def hiddenLayer(layerSize=100, length=784, func=sigmoid_neuron, bias=0 ):
    # weights matrix as a dataframe 
    layerWeights = numpy.random.normal(0, 1, size=(layerSize,length))
    def activationFunction(inputValue, weights=layerWeights): 
        return func(numpy.dot(weights, inputValue ), bias)
    ## derivative of sigmoidNeuron = (SN)*(1-SN)
    def gradientActivationFunction(inputValue, weights=layerWeights): 
        return func(numpy.dot(weights, inputValue ))* (1-func(numpy.dot(weights, inputValue )))
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
    newModel = model
    for i in range(epoch):
        for j in range(batchSize):
            newModel= gradientDescent(newModel, batchLabels[j], batchImages[j])
    return newModel

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
def gradientDescent(model,labels, images, stepSize=0.02, threshold=2):
    approxLabels = numpy.zeros((len(images), 10), dtype=object)
    newApproxLabels = numpy.zeros((len(images)), dtype=object)
    
    newWeights = numpy.zeros( (len(images), len(model)), dtype=object)
    
    while(numpy.linalg.norm(approxLabels - labels) > threshold):
        print('***** current distance of batch *****')
        print(numpy.linalg.norm(approxLabels - labels))
        print('*************************************')

        for imageIndex, image in enumerate(images):
            layerValues = numpy.zeros((len(model)+1), dtype=object)
            
            ##used to get value of layer with weights from gradient
            newLayerValues = numpy.zeros((len(model)+1), dtype=object)
            
            ## init values
            layerValues[0] = numpy.transpose(image)

            for layerIndex,layer in enumerate(model):
                layerValues[layerIndex+1] = layer.activationFunction(layerValues[layerIndex])
            approxLabels[imageIndex] = layerValues[-1]


            for layerIndex,layer in enumerate(model):
                ## e.g layer1: (781,1) -> (100,1) // layer2: (100,1) -> (10,1)  
                gradStep = stepSize *  layer.gradientActivationFunction(layerValues[layerIndex])
                ## e.g layerValues: lV[0]:(781,1) , lV[1]:(100,1), lV[2]:(10,1) 
                newWeights[imageIndex][layerIndex] =  layerValues[layerIndex+1] - gradStep
                ## newWeights[imageIndex][layerIndex]
                ## e.g nW[x][0] (100,1) // nW[x][1] (10,1)
                
            newLayerValues[0] = layerValues[1]
            
            # go back through model using new values
            for layerIndex,layer in enumerate(model):
                newLayerValues[layerIndex+1] =  layer.activationFunction(newLayerValues[layerIndex], newWeights[imageIndex][layerIndex])
            
            newApproxLabels[imageIndex] = newLayerValues[-1]
       
        updateWeights(model, averageArray(newWeights, len(model)) )        
            
    return model

def averageArray(X, size ):
    average = numpy.zeros(size , dtype=object)

    for j in range(size):
        for x in X:  
            average[j] +=  x[j]
    return average/ len(X)

def updateWeights(model, newWeights):
    for layerIndex, layer in enumerate(model):
        print('*** OLD Weights***')
        print(layer.weights())
        layer.setWeights(newWeights[layerIndex])
        print('*** NEW Weights***')
        print(layer.weights())

def testModel(model, images):
    approxLabels = numpy.zeros((len(images)), dtype=object)

    for imageIndex, image in enumerate(images):
        layerOutput = image
        for layer in model:
            layerOutput = layer.activationFunction(layerOutput)
        approxLabels[imageIndex] = layerOutput
    return approxLabels

def compareResults(approxLabels, labels):
    print('******')
    print(numpy.lingalg.norm( approxLabels - labels ))
    print('******')
    
def feed_forward_network(trainingLabels, trainingImages, epochs=2, batchSize=10, learningRate=3.0):      
    model= [ hiddenLayer(100, 784, sigmoid_neuron, 10), hiddenLayer(10,100, sigmoid_neuron) ]
    return batchTraining(epochs, batchSize, model, trainingLabels, trainingImages)        

trainingDataFrame, testDataFrame = getTrainAndTest(mnistDataFrame)
trainingLabels, trainingImages = getLabelsAndImagesInNumpy(trainingDataFrame)



trainedModel = feed_forward_network(trainingLabels, trainingImages)

testLabels, testImages = getLabelsAndImagesInNumpy(testDataFrame)

approxLabels = testModel(trainedModel, testImages)

compareResults(approxLabels, testLabels)