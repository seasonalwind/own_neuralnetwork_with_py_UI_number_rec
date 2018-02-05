import pickle
import neural_network
import numpy as np

#load the neural_network
pickle_file = open('saved_neuralnetwork.plk','rb')
savedNN = pickle.load(pickle_file)
pickle_file.close()

#using loaded nn
# load the mnist trainning data(CSV file) into a list
test_data_file = open("./DataSet/MNIST/mnist_test_10.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

print(test_data_list[1])
print('*'*50)
all_values = test_data_list[1].split(',')
inputs =  np.asfarray(all_values[1:])/255.0*0.99 + 0.01

outputs = savedNN.query(inputs)
#the index of the highest value corresponds to the label
label = np.argmax(outputs)
print("the network answer is:", label)

 
