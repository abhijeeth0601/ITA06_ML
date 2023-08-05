import numpy as np
x = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
X = x/np.amax(x, axis=0)
Y = y/100

# Sigmoid Function


def sigmoid(x):
    return 1/(1+np.exp(-x))

# Dericativve of SigmoidFunction


def derivatives_sigmoid(x):
    return x * (1-x)


# Variable of sigmoid function
epoch = 5000  # training iterations
lr = 0.1     # setting learning rate
inputlayer_neurons = 2
hiddenlayer_nerurons = 3
output_neurons = 1

# weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_nerurons))
bh = np.random.uniform(size=(1, hiddenlayer_nerurons))
wout = np.random.uniform(size=(hiddenlayer_nerurons, output_neurons))
bout = np.random.uniform(size=(1, output_neurons))

# draws a random range of numbers uniformly of dim x*y
for i in range(epoch):

    # forword propogation
    hinpl = np.dot(X, wh)
    hinp = hinpl + bh
    hlayer_act = sigmoid(hinp)
    outinpl = np.dot(hlayer_act, wout)
    outinp = outinpl + bout
    output = sigmoid(outinp)

    # back propagation
    EO = y-output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)

    # how much hidden layer wts contributed to error
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad
    # dotproduct of nextlayererror and currentlayerop
    wout += hlayer_act.T.dot(d_output) * lr
    wh += X. T. dot(d_hiddenlayer)*lr
print("Input: \n" + str(X))
print("Actual Output: \n" + str(y))
print("Predicted Output:\n", output)
