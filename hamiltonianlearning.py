# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 20:18:03 2020

@author: david
"""

import numpy as np
from scipy.linalg import expm
import time

from hlFunctions import *

# I used a seed to have reproducible results for checking speed
#np.random.seed(8)

# Choose which hamiltonian you want to learn through the variable 'hamnum'
# 1 - hyperfine splitting of hydrogen
# 2 - cyclobutadiene
# 3 - random hamiltonian
hamnum = 3
if hamnum == 1:
    numOfQuantumStates = 4
    trueHamiltonianWeights = np.array([1.0,0.0,-1.0,0.0,2.0,-1.0,0.0,0.0,0.0,1.0])
elif hamnum == 2:
    numOfQuantumStates = 4
    b = 1.0 
    trueHamiltonianWeights = np.array([0,b,0.0,0.0,b,0.0,b,0.0,b,0.0]) 
else:
    numOfQuantumStates = 3 # the hamiltonian will be this size squared; must be 3 or larger
    trueHamiltonianWeights = np.random.uniform(-1.0,1.0,int(numOfQuantumStates*(numOfQuantumStates+1)/2))

# NOTE: The true hamiltonian is treated as a black box in the code
# It is only used to generate the output data, which would normally be measured in a lab


# Randomly initialize the learning Hamiltonian. We use optimization techniques to get the
# learningHamiltonian as close as possible to the true Hamiltonian
learningHamiltonianWeights = np.random.uniform(-1.0,1.0,int(numOfQuantumStates*(numOfQuantumStates+1)/2))


# for printing
trueHamiltonian = formHamiltonian(trueHamiltonianWeights,numOfQuantumStates)
initLearningHamiltonian = formHamiltonian(learningHamiltonianWeights,numOfQuantumStates)

print("Initial learning Hamiltonian")
print(initLearningHamiltonian)
print("")
print("True Hamiltonian")
print(trueHamiltonian)
print("")

# Define the times you want to train at
# The complexity grows linearly with the length of the timeList
timeList = [3.141592654/4.0,3.141592654/8.0,3.141592654/16.0] #[np.random.uniform(0.0,2.0),np.random.uniform(0.0,2.0)]#

# Define an initial loss. It is quickly overwritten, but we need an inital value to enter the while loop
loss = 1.0

# The maximium number of iterations before you quit 
MAXITERS = 1000

# Current number of iterations of gradient descent
iters = 0

learningRate = .01

# velocities is used in momentum based gradient descent. This makes the code faster
velocities = np.zeros(trueHamiltonianWeights.shape)

dampening = 0.8 #This is a momentum parameter. Setting it to zero, you recover vanila gradient descent. .8 or.9 is reasonable

# Define input and output data
# inputData is a numpy array
# outputData is a list of numpy arrays. The list length is equal to the length of timeList
inputData = createNumChooseTwoInputData(numOfQuantumStates)
outputData = []
for index in range(0,len(timeList)):
    outputData.append( calculateCircuit(trueHamiltonian,timeList[index]) @ inputData)

lossList = [] # store loss at every epoch

# perform gradient descent
comptime = time.time() # to compare speeds
while loss > 1.5e-7 * len(timeList) :#* numOfQuantumStates**2.0: #((np.linalg.norm(gradient) > threshold) or (np.linalg.norm(velocities) > threshold))  and
    gradient = calculateGradient(calculateCost, learningHamiltonianWeights, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData)
    velocities = dampening*velocities + learningRate*gradient
    loss = calculateCost(learningHamiltonianWeights, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData)

    lossList.append(loss)
    learningHamiltonianWeights -= velocities
    iters += 1
    if iters % 10 == 0:
        # check how far from the truth you are
        learningHamiltonian = formHamiltonian(learningHamiltonianWeights,numOfQuantumStates)
        deltaEnergy = learningHamiltonian[0][0] - trueHamiltonian[0][0]
        for i in range(numOfQuantumStates):
            learningHamiltonian[i][i] -= deltaEnergy
            diff = abs(learningHamiltonian - trueHamiltonian)

        maxnorm = np.amax(diff)

        print("Iters: ", iters, " LOSS: ", loss, "Max Norm: ", maxnorm)

    elif iters > MAXITERS:
        print('did not converge')
        break

    # If you can lower the cost by multiplying the hamiltonian by -1, do it
    if iters % 501 == 0:
        if calculateCost(-learningHamiltonianWeights, trueHamiltonianWeights, numOfQuantumStates, timeList, inputData, outputData) < loss:
            learningHamiltonianWeights = -learningHamiltonianWeights



# print final stats
finalLearningHamiltonian = formHamiltonian(learningHamiltonianWeights, numOfQuantumStates)
print("Final learning Hamiltonian")
print(finalLearningHamiltonian)
print("training took: ",time.time()-comptime, " seconds")
print("Final iters: ", iters)
print("Final cost: ", loss)


# plot the iterations vs loss
import matplotlib.pyplot as plt

plt.plot(lossList)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

#### MAX NORM
# To quantify the difference between the hamitonians we have to handle
# the ambigquity of a scalar shift on the diagonal terms.
deltaEnergy = finalLearningHamiltonian[0][0] - trueHamiltonian[0][0]

for i in range(numOfQuantumStates):
    finalLearningHamiltonian[i][i] -= deltaEnergy
diff = abs(finalLearningHamiltonian - trueHamiltonian)

print("Max norm is: ", np.amax(diff))
