import pandas as pd
import dataprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor



df_train = pd.read_csv('Training set/Data_illustrated_CSV.csv')
df_test = pd.read_csv('Test_set/Aug14_Box_5.csv', header = None) #using the file Aug15_box5.csv to test the model


X_train, Y_train, training_set_processed = dataprocessing.preprocessing(df_train)
X_test, Y_test, test_set_processed = dataprocessing.preprocessing(df_test)




##### Gradient Boosting regressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0)
gbrt.fit(X_train, Y_train.iloc[:,0])

pred_gbr = gbrt.predict(X_test)

mean_gbr = mean_absolute_error(Y_test.iloc[:,0], pred_gbr)

print("Error:")
print(mean_gbr)