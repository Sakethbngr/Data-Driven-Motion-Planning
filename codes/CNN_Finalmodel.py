import pandas as pd
import os
import dataprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv1D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


df_train = pd.read_csv('Training set/Data_illustrated_CSV.csv')

#generating combined test set

#read the path
file_path = "Test_set"
#list all the files from the directory
file_list = os.listdir(file_path)
dframe = pd.DataFrame()
#append all files together
for file in file_list:
    # print(file)
    df_temp = pd.read_csv(file_path + '/' +file, header = None)
    test_dframe = dframe.append(df_temp, ignore_index = None)

# print(test_dframe)


#Data Processing

X_train, Y_train, training_set_processed = dataprocessing.preprocessing(df_train)
X_test_total, Y_test_total, test_set_processed_total = dataprocessing.preprocessing(test_dframe)


#Convoluted Neural Networks
np.random.seed(42)
tf.random.set_seed(42)


model = Sequential()
model.add(Conv1D(16, 2, activation="relu", input_shape=(20,1)))
model.add(Flatten())
model.add(Dense(64, activation="relu", kernel_regularizer = regularizers.l1()))
model.add(Dense(32,  activation="relu"))
model.add(Dense(16,  activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))

model.compile(loss="mae", optimizer= Adam(0.0001))
model.summary()
history = model.fit(X_train, Y_train, batch_size=28,epochs=100, verbose=1, validation_split=0.1)
ypred_train = model.predict(X_train)
ypred = model.predict(X_test_total)
p = model.predict(X_train)

mean_CNN = mean_absolute_error(Y_test_total, ypred)
mean_CNN_1 = mean_absolute_error(Y_train, p)

print("E_in_train:")
print(mean_CNN_1)
print("E_in_test:")
print(mean_CNN)

plt.plot(history.history['loss'])
plt.xlabel('Iterations')
plt.ylabel('E_in')
plt.title('Ein after Regularization')
plt.show()


loss=history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.plot(epochs,loss,'g',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title('Training loss vs Validation Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()