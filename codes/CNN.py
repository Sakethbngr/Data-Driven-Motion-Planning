

import pandas as pd
import dataprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.layers import Conv1D
from keras.optimizers import Adam
import numpy as np 
import tensorflow as tf

df_train = pd.read_csv('Training set/Data_illustrated_CSV.csv')
df_test = pd.read_csv('Test_set/Aug14_Box_5.csv', header = None) #using the file Aug15_box5.csv to test the model

X_train, Y_train, training_set_processed = dataprocessing.preprocessing(df_train)
X_test, Y_test, test_set_processed = dataprocessing.preprocessing(df_test)


#Convoluted Neural Networks


np.random.seed(42)
tf.random.set_seed(42)

model = Sequential()
model.add(Conv1D(16, 2, activation="relu", input_shape=(20,1)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(32,  activation="relu"))
model.add(Dense(16,  activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(4, activation="relu"))
model.add(Dense(2, activation="relu"))

model.compile(loss="mae", optimizer= Adam(0.0001))
model.summary()
model.fit(X_train, Y_train, batch_size=28,epochs=100, verbose=1)
ypred_train = model.predict(X_train)
ypred = model.predict(X_test)

mean_CNN = mean_absolute_error(Y_test, ypred)
print("Error:")
print(mean_CNN)