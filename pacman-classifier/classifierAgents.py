# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
import math
from sklearn import tree


'''
This class defines my classifier agent which uses the MultiClassSoftMax class to train
a mutliclass classifier on previous pacMan state training data and then uses this to
predict subsequent moves for PacMan based on its current state feature vectors.
self.classifier defines the classifer object which is used for training and prediction.
'''
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        self.classifier = None

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        self.getClassifier()

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.


    '''
    Prepares the feature data and y class labels. Instantiates a classifier
    using the MultiClassSoftMax class. Convert them into numpy arrays. Use
    oneVsAll method to encode class labels. Augument feature data by combining
    with a vector of ones for the weight bias.
    '''
    def getClassifier(self):

        xTrain = np.array(self.data)
        yTarget = np.array(self.target).reshape(len(xTrain),1)

        ones = np.ones((len(xTrain), 1), dtype=np.float)
        xTrain = np.append(ones, xTrain, axis=1)

        weights = np.random.rand(len(xTrain[0]), len(np.unique(yTarget)))
        softMax = MultiClassSoftMax(weights)

        yTarget = softMax.oneVsAll(yTarget)
        softMax.train(xTrain, yTarget)

        self.classifer = softMax


    '''
    Destructor method to destroy objects between games.
    '''
    def final(self, state):

        self.classifier = None

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    '''
    Get action for pacman using our self.classifier.predict method
    and feature data.
    Args:
        @state: state object containing pacMan state information
    Returns:
        api.makeMove(move, legal)
    '''
    def getAction(self, state):

        features = api.getFeatureVector(state)

        features.insert(0, 1)
        x = np.array(features)
        x = x.reshape(1, len(x))

        prediction = self.classifer.predict(x)

        move = self.convertNumberToMove(prediction[0])
        legal = api.legalActions(state)

        return api.makeMove(move, legal)


'''
MultiClassSoftMax defines our mutliclass classifer using a softmax activation
function. We define self.learning as the learning rate, self.weights as the model
weights, self.epochs as the number of iterations and self.lambdaReg as our lambda
regularlization constant. The classifier uses the following process during training:

    1: First we use a one-hot encoder function on the class labels.
    2: Calculate the net input for every class label.
    3: Calculate probabilties for each class using the softmax function.
    4: Calculate a cost using crossEntropy
    5: Calculate the cost gradient
    6: Calculate new weights using the cost gradient
    7: Iterate, to minimize cost function

To predict:

    1: Calculate net input based on training weights, z.
    2: Get class probabilties using softmax
    3: Take class with maximum probability.

Throughout the comments, the notation refers to k features in the data with n examples and
m class labels.
'''
class MultiClassSoftMax():

    def __init__(self, weights, epochs=1000, learning=0.1, lambdaReg = 1):
        self.weights = weights
        self.epochs = epochs
        self.learning = learning
        self.costLst = []
        self.n = None
        self.lambdaReg = lambdaReg


    '''
    One hot encoding for the m class labels. Takes a vector of n
    class labels and creates a matrix, (nxm) of encoded labels.
    encodes the index position as 1 that represents the class number
    and 0 where the index does not represent the class.
    Args:
        @yTarget: 2d vector, (n, 1), containing class labels.
    Returns:
        matrix of encoded class labels.
    '''
    def oneVsAll(self, yTarget):

        yCount = len(np.unique(yTarget))
        oneHotLst = [[1 if i == y else 0 for i in range(yCount)] for y in yTarget]

        return np.array(oneHotLst)


    '''
    Calculates the softMax function: it computes the probability
    the sample x(i) belongs to class yj given the weights and net input
    z(i). It returns a probability, p(y = j | x(i); wj) for each class
    j = 1, ..., m. It then normalizes the probabilities so they sum
    to one
    Args:
        @z: numpy matrix (nxm) of netInput values, w0x0 + w1x1 + ... + wkxk, for each
        example xi and each class yj.
    Returns:
        zProb: numpy matrix (nxm) of probability values that sample x(i) belongs
        to class yj.
    '''
    def softMax(self, z):

        zProb = np.exp(z)
        normalize = np.sum(zProb, axis=1).T.reshape(len(zProb),1)
        zProb = zProb/normalize

        return zProb


    '''
    Calculates the cross entropy (used to measure difference between two
    probability distributions) between yTarget and yProb. It is used to
    define the cost function.
    Args:
        @yTarget: numpy matrix (nxm) of true class labels.
        @yProb: numpy matrix (nxm) of class probability values for each x(i)
    Returns:
        cEntropy: numeric value entropy between yTarget values and yProb.
    '''
    def crossEntropy(self, yTarget, yProb):

        cEntropy = -(np.sum((yTarget * np.log(yProb))))

        return cEntropy


    '''
    Cost function that we try to minimize. Calculates the cost by finding the
    average cross entropy between true class values, yTarget, and the class
    probabilties, yProb. Uses regularlization to take the complexity of the model
    into account. Complexity(hw) = sum(|wi|**q). Here we set q=2.
    Args:
        @yTarget: numpy matrix (nxm) of true class labels.
        @yProb: numpy matrix (nxm) of class probability values for each x(i)
    Returns:
        cost: numeric cost value
    '''
    def getCost(self, yTarget, yProb):

        cEntropy = self.crossEntropy(yTarget, yProb)
        cost = float(self.n**(-1)) * cEntropy
        cost += (self.lambdaReg*0.5)*np.sum(self.weights * self.weights)

        return cost


    '''
    Computes the gradient of the cost function for gradient descent to
    learn the weights
    Args:
        @yTarget: numpy matrix (nxm) of true class labels.
        @yProb: numpy matrix (nxm) of class probability values for each x(i)
        @xTrain: numpy matrix (nxk) of training data examples.
    Returns:
        costGrad: gradient of cost function based on class labels and
        class proabilities.
    '''
    def costGradient(self, yTarget, yProb, xTrain):

        error = yTarget - yProb
        costGrad = xTrain.T.dot(error)
        costGrad = costGrad/float(-self.n)
        costGrad = costGrad + (self.lambdaReg*self.weights)

        return costGrad


    '''
    Trains MultiClassSoftMax on xTrain examples and yTarget class
    labels. Iterates according to defined number of self.epochs.
    Learns the softmax model weights by iteratively calculating
    net input, acivation softmax function, cost and costGradient.
    Args:
        @xTrain: numpy matrix (nxk) of features.
        @yTarget: numpy matrix (nxm) of true class labels.
    '''
    def train(self, xTrain, yTarget):

        self.n = len(xTrain)

        for i in range(self.epochs):
            z = self.inputFunction(xTrain, self.weights)
            yProb = self.softMax(z)

            cost = self.getCost(yTarget, yProb)
            self.costLst.append(cost)

            gradient = self.costGradient(yTarget, yProb, xTrain)

            weightsNew = self.weights - self.learning*gradient
            self.weights = weightsNew


    '''
    Calculates the net-input function, z = w0x0 + w1x1 + ... + wkxk
    for k feature vectors.
    Args:
        @features: numpy matrix (nxk) of training data.
        @weights: numpy matrix (kxm) of weights.
    Returns:
        netInput: numpy matrix (nxm) of netInput values, w0x0 + w1x1 + ... + wkxk,
        for each example xi and each class yj.
    '''
    def inputFunction(self, features, weights):
        netInput = np.matmul(features, weights)
        return netInput


    '''
    Uses self.weights obtained in classifer training. Calculates netInput,
    class probabilties using softmax function and returns predicted class
    based on class with maximum probability.
    Args:
        @x: numpy array (1, k) feature data with k features.
    Returns:
        yPredict: predicted class label in range 1-k.
    '''
    def predict(self, x):
        z = self.inputFunction(x, self.weights)
        yProb = self.softMax(z)
        yPredict = np.argmax(yProb,axis=1)

        return yPredict
