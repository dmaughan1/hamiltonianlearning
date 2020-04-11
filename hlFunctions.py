# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:14:02 2020

@author: david
"""

import numpy as np
from scipy.linalg import expm

"""
We will have a trueHamiltonian and a learningHamiltonian. We will adjust the entries of the learningHamiltonian using gradient descent to arrive at the trueHamiltonian
"""


def formHamiltonian(myNewWeights, numOfQuantumStates):
    # turn the 1 dim of weights into a 2d hamiltonian matrix
    newHamiltonian = np.zeros((numOfQuantumStates, numOfQuantumStates))
    count = 0
    for i in range(0,numOfQuantumStates):
        for j in range(0,i+1):
            newHamiltonian[i,j] = myNewWeights[count]
            newHamiltonian[j,i] = myNewWeights[count]
            count += 1
    #newHamiltonian = newHamiltonian + newHamiltonian.T - 
    return newHamiltonian

def calculateCircuit(myHamiltonian,myTime):
    # Evolve according to the Schrodinger equation
    # Or you can think of it as a quantum circuit
    return expm(-1j*myTime*myHamiltonian)

def createNumChooseTwoInputData(numOfQuantumStates):
    """
    This functions generates input data
    The data is an evenly mixture of two states
    """
    # Generate the input state you want to evolve
    numOfColumns = int(numOfQuantumStates*(numOfQuantumStates - 1)/2)
    numChooseTwoInputData = np.zeros((numOfQuantumStates,numOfColumns))

    colNum = 0
    spaceBetween = 1
    rowNum = 0
    while colNum < numOfColumns:
        
        numChooseTwoInputData[rowNum][colNum] = 1.0
        numChooseTwoInputData[rowNum + spaceBetween][colNum] = 1.0

        colNum += 1
        spaceBetween += 1

        if (rowNum + spaceBetween) == numOfQuantumStates:
            #print("here")
            spaceBetween = 1
            rowNum += 1
    return numChooseTwoInputData/np.sqrt(2)

def calculateCost(weights, trueWeights, numOfQuantumStates, timeList, inputData, outputData):
    cost = 0.0
    if len(timeList) != len(outputData):
        print("Error! list lengths do not match")
    for index in range(len(outputData)):
        learningHamiltonian = formHamiltonian(weights, numOfQuantumStates)
        learningCircuit = calculateCircuit(learningHamiltonian,timeList[index])
        trueHamiltonian = formHamiltonian(trueWeights, numOfQuantumStates)
        trueCircuit = calculateCircuit(trueHamiltonian, timeList[index])
        trueCircuitConjT = np.conj(trueCircuit).T
        
        # TODO
        # We really just need the diagonal elements of this matrix
        # So I'm wasting (n^2 - n) computations 
        costMatrix = np.dot(trueCircuitConjT,learningCircuit)

        for i_ in range(numOfQuantumStates):
            cost += 1 - abs(costMatrix[i_][i_])**2.0
            #print(1 - abs(costMatrix[i_][i_])**2.0, i_, index)

        predictedStates = learningCircuit @ inputData

        outputDataConjT = np.conj(outputData[index]).T
        costMatrix2 = np.dot(outputDataConjT,predictedStates)

        for i_ in range(numOfQuantumStates):
            cost += 1 - abs(costMatrix2[i_][i_])**2.0


        # Train on uniform superposition
        # This could be added to the input data, it was convient to have here
        uniformSuperposition = (np.ones(numOfQuantumStates).T)/np.sqrt(numOfQuantumStates)

        predictedUniformSuperposition = learningCircuit @ uniformSuperposition
        trueUniformSuperposition = trueCircuit @ uniformSuperposition

        # I didn't bother to normalize the cost for numerical underflow reasons
        cost += 1 - abs(np.dot(np.conj(trueUniformSuperposition).T,predictedUniformSuperposition))**2.0

    return cost
    

def calculateGradient(myCostFunction, learningHamiltonianWeights, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData):
    # TODO: for a final optimization, precalculate the denom instead of doing the division every time
    gradient = np.zeros(len(learningHamiltonianWeights))
    alpha = 1e-10 #This is our numerical derivative stepsize
    for i in range(len(learningHamiltonianWeights)):
        myArray = np.zeros(len(learningHamiltonianWeights))
        myArray[i] = 1.0
        gradient[i] = ( myCostFunction(learningHamiltonianWeights + alpha*myArray, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData) - myCostFunction(learningHamiltonianWeights - alpha*myArray, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData) )/(2.0*alpha)
    return gradient


def spsa(myCostFunction, learningHamiltonianWeights, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData, iters):
    # It wasn't easy to get great results with spsa, but it would be faster
    pertVec = np.random.uniform(0.1, 1.0, learningHamiltonianWeights.shape)
    gradient = np.zeros(len(learningHamiltonianWeights))
    an = .1
    numerator = ( myCostFunction(learningHamiltonianWeights + an*pertVec/np.sqrt(iters + 1), trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData) - myCostFunction(learningHamiltonianWeights - an*pertVec/np.sqrt(iters + 1), trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData) )
    for i in range(len(pertVec)):
        gradient[i] = numerator/(pertVec[i]*2.0*an)
    return gradient     

