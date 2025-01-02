#Finds the convergence of a scatter plot data
#It shows the line with the least error

from numpy import *

def computeError(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        #x value
        x = points[i, 0]
        #y value
        y = points[i, 1]
        #Get difference. square to get positive value. Add to total
        totalError += (y - (m * x + b)) **2

    #Get average
    return totalError / float(len(points))

def gradientDescentRunner(points, initialB, initialM, learningRate, numIterations):
    #starting b and m
    b = initialB
    m = initialM

    #gradient descent updates b and m for a more accurate b and m
    for i in range(numIterations):
        b, m = stepGradient(b, m, array(points), learningRate)
    return [b, m]

def stepGradient(currentB, currentM, points, learningRate):
    #starting points
    gradientB = 0
    gradientM = 0
    N = float(len(points))

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]

        #Compute partial derivative of error
        #Gives direction on where to go to find the lowest point in the bowl
        gradientB += -(2/N) * (y - ((currentM + x) + currentB))
        gradientM += -(2/N) * x * (y - ((currentM * x) + currentB))

    newB = currentB - (learningRate * gradientB)
    newM = currentM - (learningRate * gradientM)

    return [newB, newM]

def run():
    #Collect data
    points = genfromtxt("data.csv", delimiter = ",")
    #Define parameters
    #determine how fast to update b and m value
    learningRate = 0.0001
    #y = mx + b
    initialB = 0
    initialM = 0
    numIterations = 1000

    #Train model
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initialB, initialM, computeError(initialB, initialM, points)))
    [b, m] = gradientDescentRunner(points, initialB, initialM, learningRate, numIterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(numIterations, b, m, computeError(b, m, points)))

if __name__ == '__main__':
    run()

