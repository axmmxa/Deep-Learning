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
print(x_train.shape)

y_train = open_labels("data/fashion/train-labels-idx1-ubyte.gz")
print(y_train.shape)
# t-shirt has number 0, here we just want to see if 
#we have a t-shirt or not
print(y_train)

y_train = y_train == 0
print(y_train)

#definition of model
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(100, activation="sigmoid", input_shape=(784,)))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="sgd", loss="binary_crossentropy")

#reshape
x_train.reshape(60000, 784)

#model train settings
model.fit(
	x_train.reshape(60000, 784),
	y_train,
	epochs=10,
	batch_size=1000)

#do a prediction for the first picture
#if you want to change the number of the picture change x_train[1]
y_train_pred = model.predict(x_train.reshape(60000, 784))


#round datas to 0 or 1, reshape them to compare 
#datas with y_train
import numpy as np
np.round(y_train_pred).reshape(60000)

print(y_train)

print(np.round(y_train_pred).reshape(60000) == y_train)


#calculate the average
print(np.mean(np.round(y_train_pred).reshape(-1) == y_train))







