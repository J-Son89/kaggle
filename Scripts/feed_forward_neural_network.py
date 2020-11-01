#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 16:48:00 2020

@author: gad
"""

import pandas as pd
import numpy as numpy
import operator
import math

mnist_df = pd.read_csv('../Data/mnist_train.csv', sep=',')


def perceptron(x,bias=0):
    return 1 if x > bias else 0
#numpy.dot(w,x)
    
def sigmoid_neuron(x, bias=1):
    return 1/(1+ numpy.exp(-x-bias))

def sigmoid_neuronGrad(x):
    return numpy.exp(-x)/numpy.square((1+numpy.exp(-x)))

# Y is a vector of size 10
def l2_loss(Y, FX):
    return numpy.linalg.norm(Y - FX)
    #for i,fx in enumerate(FX):
    #    sum += numpy.square(Y[i] - fx[0])     
    #return sum

def flatten():
    return 1

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

def hiddenLayer(layerSize=100, func=sigmoid_neuron, length=784 ):
    # weights matrix as a dataframe 
    layerWeights = numpy.random.normal(0, 1, size=(layerSize,length))
    bias =0 
    def activationFunction(image): 
        return func(numpy.dot(layerWeights, image ), bias) 
    def setWeights(newWeights):
        nonlocal layerWeights 
        print('olWeights', layerWeights.shape)
        layerWeights = newWeights
        print('newWeights', layerWeights.shape)
        return
    def getWeights():
        nonlocal layerWeights
        return layerWeights
    return [activationFunction, getWeights, setWeights]

def gradientDescent(f, gradF, point, stepSize, threshold):
    # thresholdArray = numpy.full(point.shape, threshold)
    value = f(point)
    newPoint = point - stepSize * gradF(point)
    newValue = f(newPoint)
    
    print('newValue shape', newValue.shape )
    
    #if numpy.linalg.norm(newValue - value) < threshold:
    return value
  #  return gradientDescent(f, gradF, newPoint, stepSize, threshold)

# =============================================================================
def costFunction(approxlabels(weights), labels) :
    return numpy.linalg.norm(labels, trainModel(getModelweights[0]))
    
# =============================================================================
# epoch=iterations 1 for now

def trainModel(model, getModelWeights, setModelWeights, epochs, labels, images):
    approxLabels = numpy.zeros((len(images)), dtype=object)
    
    for i in range(epoch):
        for imageIndex, image in enumerate(images):
            layerOutput = numpy.zeros((len(model)+1), dtype=object)
            layerOutput[0] = numpy.transpose(image)
            for layerIndex, activationFunction in enumerate(model):
                layerOutput[layerIndex+1] = activationFunction(layerOutput[layerIndex])
            approxLabels[imageIndex] = layerOutput[-1]   
            newWeights = gradientDescent(costFunction(approxlabels))
            setModelWeights(newWeights)
    return approxLabels
#sigmoid_neuronGrad

def testModel(model, labels, images):
    approxLabels = numpy.zeros((len(images)), dtype=object)

    for imageIndex, image in enumerate(images):
        layerOutput = image
        for layer in model:
            layerOutput = layer(layerOutput)
        approxLabels[imageIndex] = l2_loss(labels[imageIndex], layerOutput)
        
        
        print('expected',labels[imageIndex])
        print('actual', layerOutput)
        print('loss',l2_loss(labels[imageIndex], layerOutput))
        print('##END##')

    return approxLabels

def feed_forward_network(dataframe, epochs=2, batch_size=100, learning_rate=3.0):
    train_df, test_df = getTrainAndTest(dataframe)
    train_labels, train_images = getLabelsAndImagesInNumpy(train_df)
    test_labels, test_images = getLabelsAndImagesInNumpy(test_df)
    
    layer1, getWeights1, setWeights1 = hiddenLayer(100,sigmoid_neuron, 784)
    #layer2, getWeights2, setWeights2 =  hiddenLayer(10,sigmoid_neuron,100)
    
    model = [layer1]
    getModelWeights = [ getWeights1]
    setModelWeights = [ setWeights1]
    x = trainModel(model, getModelWeights,setModelWeights , epochs, train_labels,train_images) 
    
    #testModel(model, test_labels, test_images)
    return model
        

P = feed_forward_network(mnist_df)