#Function to load the data
import gzip 
import numpy as np 

def open_images(filename):
	with gzip.open(filename, "rb") as file:
		data = file.read()
		return np.frombuffer(data, dtype=np.uint8, offset=16)\
			.reshape(-1, 28, 28)\
    		.astype(np.float32)

def open_labels(filename):
    with gzip.open(filename, "rb") as file:
    	data = file.read()
    	return np.frombuffer(data, dtype=np.uint8, offset=8)\

#loads datas form directory
x_train = open_images("data/fashion/train-images-idx3-ubyte.gz")


y_train = open_labels("data/fashion/train-labels-idx1-ubyte.gz")

# t-shirt has number 0, here we just want to see if 
#we have a t-shirt or not


y_train = y_train == 0


x_test = open_images("data/fashion/t10k-images-idx3-ubyte.gz")

y_test = open_labels("data/fashion/t10k-labels-idx1-ubyte.gz")

y_test = y_test == 0


#definition of model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy", metrics=["accuracy"])

#reshape
x_train.reshape(60000, 784)


#model train settings
model.fit(
	x_train.reshape(60000, 784),
	y_train,
	epochs=10,
	batch_size=1000)

print(model.evaluate(x_test.reshape(10000, 784), y_test))