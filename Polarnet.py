from random import uniform
import math
import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


trainingIterations = int(input("how many times to train"))


n = 30 #medial neurons
inp = 3 #input neurons
out = 2 #output neurons


inputs = [0.0, 0.0, 1.0]
synone = np.random.uniform(-1.0, 1.0, (inp, n))
syntwo = np.random.uniform(-1.0, 1.0, (n, out))
medin = np.zeros(n)
medout = np.zeros(n)
output = np.zeros(out)      #medin/out in/output of medial layer
error = np.zeros(out)
sigma = np.zeros(n)
sigmoid = np.zeros(n)
rate = 0.005
errorhistory = []


def selectRandomPoint(inputs):
    r = uniform(0.0, 1.0)
    theta = uniform(0.0, 1.0) 

    inputs[0] = r
    inputs[1] = theta
    return inputs


def convertCoordinates(n, inp, out, inputs):
    for i in range(n):
        medin[i] = 0
        for j in range(inp):
          medin[i] = medin[i] + synone[j, i] * inputs[j]
        medout[i] = math.tanh(medin[i])

    for i in range(out):
        output[i] = 0 
        for j in range(n):
            output[i] = output[i] + syntwo[j, i] * medout[j]
        output[i] = math.tanh(output[i])


def calculateTrueCoordinates(inputs):
    r = inputs[0]
    theta = math.radians(inputs[1] * 360)
    true_x = r * math.cos(theta)
    true_y = r * math.sin(theta)

    error[0] = true_x - output[0]
    error[1] = true_y - output[1]

    totalloss = (error[0] ** 2 + error[1] ** 2) / 2
    errorhistory.append(totalloss)



#back propagation
def backPropagation(n, out, rate, synone, syntwo, inputs, error, medout):
    #last synapses first
    for i in range(out): 
        for j in range(n):
         syntwo[j, i] = syntwo[j, i] + rate *  medout[j] * error[i]

    #medial layer
    for i in range(n):
        sigma[i] = 0
        for j in range(out):
            sigma[i] = sigma[i] + error[j] * syntwo[i, j]
        sigmoid[i] = 1 - (medout[i]) ** 2

    #first synapses
    for i in range(inp):
        for j in range(n):
            delta = rate * sigmoid[j] * sigma[j] * inputs[i]
            synone[i, j] = synone[i, j] + delta


def graphError(errorhistory):
    plt.figure(figsize=(10, 5))
    plt.plot(errorhistory, label="Training Loss (Error)", color="black", alpha=0.6)


    plt.title("Neural Network Learning Progress")
    plt.xlabel("Training Iterations")
    plt.ylabel("Error (Loss)")
    plt.grid(True)
    plt.legend()
    plt.show()

#main loop
for i in range(trainingIterations):
    selectRandomPoint(inputs)
    convertCoordinates(n, inp, out, inputs)
    calculateTrueCoordinates(inputs)
    backPropagation(n, out, rate, synone, syntwo, inputs, error, medout)
graphError(errorhistory)
usrInput = 0
#CLI
while not usrInput == "q":
    usrInput = input("input radius, or q to quit")
    inputs[0] = float(usrInput)
    usrInput = input("input angle, or q to quit")
    inputs[1] = float(usrInput)

    convertCoordinates(n, inp, out, inputs)
    for i in range(out):
        print(output[i])

    calculateTrueCoordinates(inputs)
    for i in range(out):
        print(error[i])

