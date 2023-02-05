import pandas as pd
import dataprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df_train = pd.read_csv('Training set/Data_illustrated_CSV.csv')
df_test = pd.read_csv('Test_set/Aug14_Box_5.csv', header = None) #using the file Aug15_box5.csv to test the model


X_train, Y_train, training_set_processed = dataprocessing.preprocessing(df_train)
X_test, Y_test, test_set_processed = dataprocessing.preprocessing(df_test)

reg = LinearRegression().fit(X_train, Y_train)
pred_lin = reg.predict(X_test)

mean_lin = mean_absolute_error(Y_test, pred_lin)
print("Error:")
print(mean_lin)