import numpy as np 
import matplotlib.pyplot as plt
import pylab 
import neural_network
import pickle   #the built-in pickle module is unable to serialize(lambda functions, nested functions, and functions defined at the command line) 
#import cloud.serialization.cloudpickle

'''
# ---------------this is a test part

print(data_list[0])

all_values = data_list[1].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array,cmap='Greys',interpolation='None')
pylab.show()
#----------------------------
'''

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
#output nodes is 10 (for MNIST)
output_nodes = 10

# learning rate is 0.3
learning_rate = 0.1

# create instance of neural network
n = neural_network.neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load the mnist trainning data(CSV file) into a list
trainning_data_file = open("./DataSet/MNIST/mnist_train.csv",'r')
trainning_data_list = trainning_data_file.readlines()
trainning_data_file.close()

### train the neural network
epochs = 7
# go through all records in the trainning data set
for e in range(epochs):
	for record in trainning_data_list:
		all_values = record.split(',')
		# rescale the input: values range from 0 to 255 to mach smaller range 0.01 to 0.99
	    # chosen 0.01 because 0 can artificially kill weight updates
	    # chosen 0.99 because we should avoid the outputs to reach 1.0
		inputs = np.asfarray(all_values[1:])/255.0*0.99 + 0.01
	    # create the target output values 
		targets = np.zeros(output_nodes)+0.01
		targets[int(all_values[0])] = 0.99
		n.train(inputs,targets)
		pass
	pass 	
### test the network
# load the mnist test data(CSV file) into a list
test_data_file = open("./DataSet/MNIST/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list:
	all_values = record.split(',')
	correct_label = int(all_values[0])
	#print("correct label is :",correct_label)
	inputs = np.asfarray(all_values[1:])/255.0*0.99+0.01
	outputs = n.query(inputs)
    #the index of the highest value corresponds to the label
	label = np.argmax(outputs)
	#print("the network answer is:", label)

    #append correct or incorrect to list
	if(label==correct_label):
		scorecard.append(1)
	else:
		scorecard.append(0)
		pass
	pass
#print(scorecard)

### calculate the performance score
scorecard_array = np.asarray(scorecard)
performance = scorecard_array.sum()/float(scorecard_array.size)
print("performance=", performance)

if(performance >= 0.95):
	save_nn = open('saved_neuralnetwork.plk','wb')
	pickle.dump(n,save_nn)
	save_nn.close()
	print('The neural network which has a good performance has been saved.....')


