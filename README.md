# Data-Driven-Motion-Planning
  
## Objective

The goal of this project is to develop a machine learning model for a four-wheeled toy car with non-holonomic constraints, as it navigates through corridor and box environments using real-time sensor data collected through laser scanning. This approach is referred to as "Data-Driven Motion Planning."



## Dependencies Used
1. Numpy
2. Matplotlib
3. Pandas
4. scikit-learn
5. TensorFlow


## Datasets:
1. Training set: https://drive.google.com/drive/folders/1IgiPMaMyktjIa9qH-5qqSjwFuew9TiPW
2. Test set: https://drive.google.com/drive/folders/1IhVbX1VwAQf4WzamN8m81sNQTABDWw5y

## Pipelines Evaluated:

1. Linear regression (Mean Absolute Error: 0.24189557811305168)
2. Support Vector Regression (Mean Absolute Error: 0.26905703357030397)
3. Gradient Boosting Regresson (Mean Absolute Error: 0.20074039777749236)
4. Convoluted Neural Networks (Mean Absolute Error: 0.18509529328224894)

## Final Model Used:

Convoluted Neural Networks

## Results (After Regularization):

![alt text](https://github.com/Sakethbngr/Data-Driven-Motion-Planning/blob/712ce0f74ad49a4099204734bcd5de5d1515d4e3/Results/Ein.png?raw=true)
![alt text](https://github.com/Sakethbngr/Data-Driven-Motion-Planning/blob/dd90bbe217781af9fbf582951a6edcae79cddc51/Results/after_l1.png?raw=true)





