# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt


#
# n_input=4
# n_hidden=10
# n_output=1
# Learningrate=0.1
# epochs=10000
#
# X= tf.placeholder(tf.float32)
# Y=tf.placeholder(tf.float32)
#
# Weights1= tf.Variable(tf.random_uniform([n_input,n_hidden], -1.0,1.0))
# Weights2= tf.Variable(tf.random_uniform([n_hidden,n_output], -1.0,1.0))
#
# bias1= tf.Variable(tf.zeros([n_hidden]), name="bias1")
# bias2= tf.Variable(tf.zeros([n_output]), name="bias2")
#
# Level2= tf.sigmoid(tf.matmul(X,Weights1)+bias1)
# OutputLayer = tf.sigmoid((tf.matmul(Level2,Weights2)+bias2))
#
# cost = tf.reduce_mean(-Y*tf.log(OutputLayer) - (1-Y)*tf.log(1-OutputLayer))
# optimizer= tf.train.GradientDescentOptimizer(Learningrate).minimize(cost)
#
# init = tf.initialize_all_variables()
#
# with tf.Session() as session:
#     session.run(init)
#     for step in range(epochs):
#         session.run(optimizer,feed_dict={X:inputnp,Y:outputnp})
#
#         if step%1000==0:
#             print (session.run(cost,feed_dict={X:inputnp,Y:outputnp}))
#     answer = tf.equal(tf.floor(OutputLayer),Y)
#     accuracy = tf.reduce_mean(tf.cast(answer,"float"))
#
#     print(session.run([OutputLayer],feed_dict={X:inputnp,Y:outputnp}))

from numpy import exp, array, random, dot
import json

class NeuronLayer():
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2 * random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1.synaptic_weights += layer1_adjustment
            self.layer2.synaptic_weights += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1.synaptic_weights))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2.synaptic_weights))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print("    Layer 1 (10 neurons, each with 4 inputs): ")
        print(self.layer1.synaptic_weights)
        print("    Layer 2 (1 neuron, with 4 inputs):")
        print(self.layer2.synaptic_weights)


if __name__ == "__main__":
    # Seed the random number generator
    random.seed(1)
    with open('cleanData.json', 'r') as f:
        cleanData = json.load(f)

    inputarray = []
    outputarray = []

    for i in cleanData:
        inputarray.append([i[0], i[1], i[2], i[3]])
        outputarray.append([i[4]])
    inputnp = array(inputarray)
    outputnp = array(outputarray)
    print(inputarray)
    print(outputarray)
    with open('cleanData.json', 'r') as f:
        cleanData = json.load(f)

    inputarray=[]
    outputarray=[]

    for i in cleanData:
        inputarray.append([i[0],i[1],i[2],i[3]])
        outputarray.append([i[4]])
    inputnp=array(inputarray)
    outputnp=array(outputarray);

    # Create layer 1 (4 neurons, each with 3 inputs)
    layer1 = NeuronLayer(5, 4)

    # Create layer 2 (a single neuron with 4 inputs)
    layer2 = NeuronLayer(1, 5)

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    print("Stage 1) Random starting synaptic weights: "+str(neural_network.print_weights()))

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    training_set_inputs = inputnp
    training_set_outputs =outputnp

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 10)

    print("Stage 2) New synaptic weights after training: "+str(neural_network.print_weights()))

    # Test the neural network with a new situation.
    print("Stage 3) Considering a new situation [19, 10, 3, 5] -> ?: ")
    hidden_state, output = neural_network.think(array([24, 14, 3, 25]))
    print(output)