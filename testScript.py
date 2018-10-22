import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
with open('cleanData.json', 'r') as f:
    cleanData = json.load(f)

inputarray=[]
outputarray=[]

for i in cleanData:
    inputarray.append([i[0],i[1],i[2],i[3]])
    outputarray.append([i[4]])
inputnp=np.array(inputarray)
outputnp=np.array(outputarray);
print(inputarray)
print(outputarray)

n_input=4
n_hidden=10
n_output=1
Learningrate=0.1
epochs=10000

X= tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

Weights1= tf.Variable(tf.random_uniform([n_input,n_hidden], -1.0,1.0))
Weights2= tf.Variable(tf.random_uniform([n_hidden,n_output], -1.0,1.0))

bias1= tf.Variable(tf.zeros([n_hidden]), name="bias1")
bias2= tf.Variable(tf.zeros([n_output]), name="bias2")

Level2= tf.sigmoid(tf.matmul(X,Weights1)+bias1)
OutputLayer = tf.sigmoid((tf.matmul(Level2,Weights2)+bias2))

cost = tf.reduce_mean(-Y*tf.log(OutputLayer) - (1-Y)*tf.log(1-OutputLayer))
optimizer= tf.train.GradientDescentOptimizer(Learningrate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(init)
    for step in range(epochs):
        session.run(optimizer,feed_dict={X:inputnp,Y:outputnp})

        if step%1000==0:
            print (session.run(cost,feed_dict={X:inputnp,Y:outputnp}))
    answer = tf.equal(tf.floor(OutputLayer),Y)
    accuracy = tf.reduce_mean(tf.cast(answer,"float"))

    print(session.run([OutputLayer],feed_dict={X:inputnp,Y:outputnp}))