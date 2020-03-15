import numpy as np
from scipy.linalg import expm
import time

from hlFunctions import *

# In case you want to save results
#timestr = time.strftime("%Y%m%d-%H%M")
#myFile = open(timestr + "hamiltonianLearningTest.txt", "w")
np.random.seed(8)

# Define variables
numOfQuantumStates = 4 # Should be equal to 4 if you want to simulate hyperfine or cyclobutadiene
symmetricBasis = generateSymmetricBasis(numOfQuantumStates)

# Choose which hamiltonian you want to attempt to learn. We have included
# the hamiltonians for hyperfine splitting, the cyclobutadiene molecule
# and a totally random hamiltonian

# Define the true hamiltonian, which is treated as a black box in the code
# It is only used to generate the output data

""" PREDEFINED HAMILTONIANS
# Hamiltonian for hyperfine ### numOfQuantumStates must equal 4
#trueHamiltonianWeights = np.array([1.0,0.0,-1.0,0.0,2.0,-1.0,0.0,0.0,0.0,1.0]) 

# Hamiltonian for cyclobutadiene molecule ### numOfQuantumStates must equal 4
#b = 1.0 # This value and hamiltonian comes from stack exchange post Dave put in our Slack
#trueHamiltonianWeights = np.array([0,b,0.0,0.0,b,0.0,b,0.0,b,0.0]) 
"""
# Random Hamiltonian
trueHamiltonianWeights = np.random.uniform(-1.0,1.0,int(numOfQuantumStates*(numOfQuantumStates+1)/2))

# Randomly initialize the learning Hamiltonian. We use optimization techniques to try get the learningHamiltonian as
# close as possible to the true Hamiltonian
learningHamiltonianWeights = np.random.uniform(-1.0,1.0,int(numOfQuantumStates*(numOfQuantumStates+1)/2))
##learningHamiltonianWeights = np.random.uniform(-1.0,1.0,int(numOfQuantumStates*(numOfQuantumStates+1)/2))

#myFile.write("True Weights: " + str(trueHamiltonianWeights) +"\n")
#myFile.write("Learning Weights: " + str(learningHamiltonianWeights) +"\n")

trueHamiltonian = formHamiltonian(trueHamiltonianWeights,symmetricBasis,numOfQuantumStates)
initLearningHamiltonian = formHamiltonian(learningHamiltonianWeights,symmetricBasis,numOfQuantumStates)

print("Initial learning Hamiltonian")
print(initLearningHamiltonian)
print("")
print("True Hamiltonian")
print(trueHamiltonian)
print("")

# Define the times you want to train at
# The complexity grows linearly with the length of the timeList
timeList = [3.141592654/4.0,3.141592654/8.0,3.141592654/16.0] #[np.random.uniform(0.0,2.0),np.random.uniform(0.0,2.0)]#
#myFile.write("Time List: " + str(timeList) + "\n")
# Define an initial loss. It is quickly overwritten, but we need an inital value to enter the while loop
loss = 1.0

# The maximium number of iterations before you quit 
MAXITERS = 4000

# Current number of iterations of gradient descent
iters = 0

learningRate = .01

# Some initial data info
#print("Initial Weights: ", learningHamiltonianWeights)
#print("Desired Weights: ", trueHamiltonianWeights)

# velocities is used in momentum based gradient descent. This makes the code faster
velocities = np.zeros(trueHamiltonianWeights.shape)

threshold = 2e-6 #This convergence threshold is a little finicky. Depends on the problem you are solving
dampening = 0.8 #This is a momentum parameter. Setting it to zero, you recover vanila gradient descent. .8 or.9 is reasonable

# Define input and output data
# inputData is a numpy array
# outputData is a list of numpy arrays. The list length is equal to the length on timeList
inputData = createNumChooseTwoInputData(numOfQuantumStates)
outputData = []
for index in range(0,len(timeList)):
    outputData.append( calculateCircuit(trueHamiltonian,timeList[index]) @ inputData)

# Set the initial gradient to something large enough to enter the while loop
gradient = np.ones(trueHamiltonianWeights.shape)

# perform gradient descent
lossList = []
while loss > 1.5e-11 * len(timeList) * numOfQuantumStates**2.0: #((np.linalg.norm(gradient) > threshold) or (np.linalg.norm(velocities) > threshold))  and
    gradient = calculateGradient(calculateCost, learningHamiltonianWeights, trueHamiltonianWeights, symmetricBasis, numOfQuantumStates, timeList, inputData, outputData)
    velocities = dampening*velocities + learningRate*gradient
    loss = calculateCost(learningHamiltonianWeights, trueHamiltonianWeights, symmetricBasis, numOfQuantumStates, timeList, inputData, outputData)

    lossList.append(loss)
    learningHamiltonianWeights -= velocities
    iters += 1
    if iters % 10 == 0:
        learningHamiltonian = formHamiltonian(learningHamiltonianWeights,symmetricBasis,numOfQuantumStates)
        deltaEnergy = learningHamiltonian[0][0] - trueHamiltonian[0][0]
        for i in range(numOfQuantumStates):
            learningHamiltonian[i][i] -= deltaEnergy
            diff = abs(learningHamiltonian - trueHamiltonian)

        maxnorm = np.amax(diff)

        print("Iters: ", iters, " LOSS: ", loss, "Max Norm: ", maxnorm)
        #print("Weights")
        #print(learningHamiltonianWeights)
        #print("")
    elif iters > MAXITERS:
        print("normVel:",np.linalg.norm(velocities), "normGrad:", np.linalg.norm(gradient), "COST:",loss)
        #print(learningHamiltonianWeights)
        break
    #""" commented out for SPSA
    # If you can lower the cost by multiplying the hamiltonian by -1, do it
    if iters % 501 == 0:
        if calculateCost(-learningHamiltonianWeights, trueHamiltonianWeights, symmetricBasis, numOfQuantumStates, timeList, inputData, outputData) < loss:
            learningHamiltonianWeights = -learningHamiltonianWeights
    if np.linalg.norm(velocities) < threshold and np.linalg.norm(gradient) < threshold:
        print("Did not converge!!!!!!!")
        break
   # """

    


# print final stats
finalLearningHamiltonian = formHamiltonian(learningHamiltonianWeights, symmetricBasis, numOfQuantumStates)

print("Final iters: ", iters)
print("Final cost: ", loss)
print("Final learning Hamiltonian")
print(finalLearningHamiltonian)

#myFile.write("Final iters: " + str(iters) + "\n")
#myFile.write("Final cost: " + str(loss) + "\n")
#myFile.write("final learning weights: " + str(learningHamiltonianWeights)) 

#myFile.close()

# plot the iterations vs loss
import matplotlib.pyplot as plt

plt.plot(lossList)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#### MAX NORM

# To quantify the difference between the hamitonians we have to handle
# the ambigquity of a scalar shift on the diagonal terms.
deltaEnergy = finalLearningHamiltonian[0][0] - trueHamiltonian[0][0]

for i in range(numOfQuantumStates):
    finalLearningHamiltonian[i][i] -= deltaEnergy
diff = abs(finalLearningHamiltonian - trueHamiltonian)

print("Max norm is: ", np.amax(diff))
