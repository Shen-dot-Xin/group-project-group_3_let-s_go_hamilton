## Question 2

> Fit a model using data from 1950:2010, and predict drivers that come in second place between 2011 and 2017. From your fitted model:
> - describe your model, and explain how you selected the features that were selected
> - provide statistics that show how good your model is at predicting, and how well it performed predicting second places in races between 2011 and 2017?
> - the most important variable in Q1 is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in Q1. How different are they? Why are they different?

#### Model selection:
> - describe your model, and explain how you selected the features that were selected

Our best model is the Random Forest Regression model with 0.69 R^2 , 92% overall accuracy (both 0 and 1) and 50% actual accuracy (only predicting 1).The model selection period is quite hard.

1.	At the beginning, we use the Random Forest Classifier model (see part_1). To predict whether the driver comes in the second place, we use binary prediction output (second place = 1, otherwise = 0). Although the model provides unusually high accuracy (0.95), the actual prediction power of the model is low. The binary learning output makes the model to predict all the result as 0 and still get a high accuracy, the other measurement scores are quite low (almost all 0). This model actually overlooks many important informations when it consider all non-second place data as the same 0.
 ![binary_metrics](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Binary_Metrics.png)
2.	So, we change the binary classifier model into non-binary classifier model (the prediction output is the position), the accuracy decreases to 0.262, which is quite low, but the prediction result is reasonable. If we randomly guess the ranking, the possibility will be 0.05, and due to the lack of data, the accuracy cannot be very high.
 ![non-binary_metrics](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Non-Binary_Metrics.png)
3.	However, we are still unsatisfied with the result and try to figure out better models. In this case, we use Random Forest Regression model as alternative. For Regression model, we change the output from “finishposition” into “driverRacePoints”. The R^2 is pretty high (70%), yet, the new issue is that the regression does not directly provide the prediction accuracy. In this case, we need to manual calculate the accuracy. Manually, we consider drivers who have the predicted “driverRacePoints” around 18 (16,17,18, and 19) as in the second place, because the model tend to predict a lower points than reality. Finally, the accuracy on predicting both 0 and 1 is 92%, but the accuracy on predicting 1 (the prediction second palce/ real second palce) is 50%. Finally, we believe this model is good enough for prediction.
![overall_accuracy](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Overall_Accuracy.png)
![actual_accuracy](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Actual_accuracy_as1.png)

#### Features Selection:

Many feature selection processes are similar to those in Q1:
1.	The performance of the driver in the whole season will affect the result, so we include “driverStPosition” and “driverSeasonWins”.
2.	The performance of the constructor will affect the result, so we include “constStPosition” and “constSeasonWins”
3.	The performance of the driver in this race will affect the result, so we include “fastestLapRank”, “gridPosition”, “fastestLap”, and “fastestLapSpeed”.
4.	The performance of the driver in the previous three races will affect the result, so we include “finishPositionRM3”, “finishPositionRM2”, and “finishPositionRM1”
The importance of features is listed below:
 ![non-binary_metrics](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Features_Importance.png)
 
However, since we select regression model instead of classifier model, some of the key features in Q2 model are different from those in Q1.
1.	We drop those binary variables in Q2, since the binary variables are not very powerful in the regression model.
2.	We add more features into the model.

#### Model statistics
> - provide statistics that show how good your model is at predicting, and how well it performed predicting second places in races between 2011 and 2017?

As shown below, the random forest binary classifier model and random forest non-binary classifier model have poor actual prediction power:
Binary:

![binary_metrics](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Binary_Metrics.png)
 
Non-binary:
 ![non-binary_metrics](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Non-Binary_Metrics.png)
In this case, we choose random forest regression model. After parameter adjustment, we come up with our optimal model with best prediction power (parameters: "n_estimators": 1000, "max_depth": 5, "random_state": 10), the performance of its predicting second places in races between 2011 and 2017 is as below:
  ![mlflow](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_MLflow_result.png)
The predicting performance between 2011 and 2017 is as below, we calculate it via Superset(http://ec2-3-84-157-243.compute-1.amazonaws.com:8088/r/36; http://ec2-3-84-157-243.compute-1.amazonaws.com:8088/r/37）:
![overall_accuracy](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Overall_Accuracy.png)
![actual_accuracy](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q2_Actual_accuracy_as1.png)

As shown in the figures, the accuracy on 1 keep increasing, and the average accuracy is almost 50%. A 50% accuracy is quite high for prediciting a specific ranking in the race. We are statisfied with the model.

#### Feature Importance
> - the most important variable in (1) is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in (1). How different are they? Why are they different?
According to our model, the top 5 important features are as below:
 
The marginal effect of the “driverStPosition” is:
Before dropping:
 
After dropping:
 

Both
 
Compare with most important variable in Q1, the difference are:

