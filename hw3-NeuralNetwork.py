import numpy as np
import sys
import csv
import math
import time

def read_input():
	content = []
	with open(sys.argv[1], "r") as f:
		reader = csv.reader(f)
		for line in reader:
			content.append([int(x) for x in line])
		train_image = np.array(content)

	content = []
	with open(sys.argv[2], "r") as f:
		reader = csv.reader(f)
		for line in reader:
			content.append([int(x) for x in line])
		train_label = np.array(content)

		train_label_transpose = train_label.T
		train_label_prime = np.zeros((10, train_label_transpose.shape[1]))
		for i in range(train_label_transpose.shape[1]):
			train_label_prime[ train_label_transpose[0][i] ][i] = 1

	content = []
	with open(sys.argv[3], "r") as f:
		reader = csv.reader(f)
		for line in reader:
			content.append([int(x) for x in line])
		test_image = np.array(content)

	return train_image.T, train_label_prime, test_image.T

def write_output(predict):
	content = predict.tolist()
	data = []
	for item in content:
		data.append([item])
	with open('test_predictions.csv','w') as f:
		writer = csv.writer(f)
		writer.writerows(data)

def sigmoid(x):
	z = 1/(1 + np.exp(-x))
	return z

def sigmoid_derivative(x):
	z = np.exp(-x) / ((1+np.exp(-x))**2)
	return z

def add_B(M, B):
	if M.shape[0] == B.shape[0]:
		row = M.shape[0]
		col = M.shape[1]
		for i in range(row):
			for j in range(col):
				M[i][j] = M[i][j] + B[i][0]
		return M
	else:
		print("Error in add_B")
		return False

def softmax(X):
	output_out = np.zeros(X.shape)
	for i in range(X.shape[1]):
		piece = X[:,i] #1d array size 10
		soft = np.exp(piece) / np.sum(np.exp(piece)) #softmax
		for j in range(X.shape[0]):
			output_out[j][i] = soft[j]
	return output_out

def test_accuracy(predict):
	content = []
	with open("test_label.csv", "r") as f:
		reader = csv.reader(f)
		for line in reader:
			content.append([int(x) for x in line])
		test_label = np.array(content)
		truth = test_label.T

	total = truth.shape[1]
	correct = 0
	for i in range(total):
		if predict[i] == truth[0][i]:
			correct = correct + 1.0

	accuracy = 100*correct/total
	print("corrrect:", correct)
	print("total:", total)
	print("accuracy:", accuracy)


if __name__ == "__main__": 

	VERBOSE = False

	start = time.time()

	train_image, train_label, test_image = read_input() #(784, 60000) (10, 60000) (784, 10000)

	if VERBOSE:
		print(train_image.shape)
		print(train_label.shape)
		print(test_image.shape)

	train_image = train_image/255
	test_image = test_image/255

	epoch = 20 #2000
	batch_size = 100
	iteration = int(train_image.shape[1]/batch_size)
	learning_rate = 1

	hidden_layer1_size = 128
	hidden_layer2_size = 64
	output_layer_size = 10

	W1 = np.random.randn(hidden_layer1_size, 784)*0.1
	B1 = np.zeros((hidden_layer1_size, 1))

	W2 = np.random.randn(hidden_layer2_size, hidden_layer1_size)*0.1
	B2 = np.zeros((hidden_layer2_size, 1))

	W3 = np.random.randn(output_layer_size, hidden_layer2_size)*0.1
	B3 = np.zeros((output_layer_size, 1))

	for i in range(epoch): 
		for j in range(iteration):

			temp_train = train_image[:, j*batch_size:(j+1)*batch_size]
			temp_label = train_label[:, j*batch_size:(j+1)*batch_size]

			#forward pass
			L1_in = add_B(np.matmul(W1, temp_train), B1)
			L1_out = sigmoid(L1_in) #hidden_layer1_size, batch_size

			L2_in = add_B(np.matmul(W2, L1_out), B2)
			L2_out = sigmoid(L2_in) #hidden_layer2_size, batch_size

			output_in = add_B(np.matmul(W3, L2_out), B3) 
			output_out = softmax(output_in) # output_layer_size(10), batch_size(60000) 

			#compute loss
			loss = -(1./batch_size) * np.sum(temp_label * np.log(output_out))

			#back propagate
			d3 = output_out - temp_label #(10, batch_size)
			dw3 = (1./batch_size) * np.matmul(output_out - temp_label, L2_out.T) #(10, hidden_layer2_size)
			db3 = (1./batch_size) * np.sum(output_out - temp_label, axis=1, keepdims=True) #sum rows produce a 1d(image_size) array and average

			d2 = np.matmul(W3.T, d3) #(hidden_layer2_size, batch_size)
			d2_prime = d2 * sigmoid_derivative(L2_in) #f'(Ij)*delta 
			dw2 = (1./batch_size) * np.matmul(d2_prime, L1_out.T) #oi*f'(Ij)*delta  (hidden_layer2_size, hidden_layer1_size)
			db2 = (1./batch_size) * np.sum(d2_prime, axis=1, keepdims=True)

			d1 = np.matmul(W2.T, d2)
			d1_prime = d1 * sigmoid_derivative(L1_in) #(hidden_layer1_size, batch_size)
			dw1 = (1./batch_size) * np.matmul(d1_prime, temp_train.T) #(hidden_layer1_size, 784)
			db1 = (1./batch_size) * np.sum(d1_prime, axis=1, keepdims=True)

			W3 = W3 - learning_rate*dw3
			W2 = W2 - learning_rate*dw2
			W1 = W1 - learning_rate*dw1

			B3 = B3 - learning_rate*db3
			B2 = B2 - learning_rate*db2
			B1 = B1 - learning_rate*db1

		if VERBOSE:
			print("epoch", i, "loss", loss)


	#forward pass
	L1_in = add_B(np.matmul(W1, test_image), B1)
	L1_out = sigmoid(L1_in) #hidden_layer1_size, batch_size

	L2_in = add_B(np.matmul(W2, L1_out), B2)
	L2_out = sigmoid(L2_in) #hidden_layer2_size, batch_size

	output_in = add_B(np.matmul(W3, L2_out), B3) 
	output_out = softmax(output_in) # output_layer_size(10), batch_size

	predict = np.argmax(output_out, axis=0) #choose maximum index of each column and form 1d array

	if VERBOSE:
		test_accuracy(predict)

	write_output(predict)

	end = time.time()
	cost = end - start
	print("total time:", cost)
	print("Average time for 1 epoch", cost/10)