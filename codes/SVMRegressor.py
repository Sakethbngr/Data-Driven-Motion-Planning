import pandas as pd
import dataprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


df_train = pd.read_csv('Training set/Data_illustrated_CSV.csv')
df_test = pd.read_csv('Test_set/Aug14_Box_5.csv', header = None) #using the file Aug15_box5.csv to test the model


X_train, Y_train, training_set_processed = dataprocessing.preprocessing(df_train)
X_test, Y_test, test_set_processed = dataprocessing.preprocessing(df_test)

### Epsilon SVM regressor




regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, Y_train.iloc[:,0])

pred_SVM = regr.predict(X_test)


mean_SVM = mean_absolute_error(Y_test.iloc[:,0], pred_SVM)
print("Error:")
print(mean_SVM)