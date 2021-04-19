
## Model Description 

As per the question required, I choose data from between 1950 to 2010 and looked at all races data, explored a few features and conducted some exploratory analysis. I decided to use Logistic Regression model after trying a few higher-order Classifier models. 

The Logistic model was given a Binary column of whether or not a driver ended up in second position as Y and a list of six features to model based on the features are explained in further sections of the document. 

## Feature Selection and reasoning

| Feature            	| Description and Reason                                                                                                                                                                                                                                                                                   	|
|--------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| gridPosition           	| This features acts as a IID for each record, choosen just for convenience and when running trials with other tree based models.                                                                                                                                                                  	|
| driverRacePoints   	| This feature says the points a driver got in a race, it is directly proportional to the position a driver finishes in and specifically for our prediction of second position which has 18 points.                                                                                                        	|
| driverSeasonPoints 	| This feature shows points in the season thus far for the Driver in a cumulative fashion, it was chosen since it may help determine if a second position is going to be a one off performance or one with some repeatability.                                                                             	|
| drivSecPosRM1      	| A binary column that says if a driver finished second or not in the previous race! This might massively determine the nature and the car quality and consistency                                                                                                                                         	|
| drivSecPosRM2      	| A binary column that says if a driver finished second or not in the last-before race! This might massively determine the nature and the car quality and consistency                                                                                                                                      	|
| drivSecPosRM3      	| A binary column that says if a driver finished second or not in the two-races-before one! This might massively determine the nature and the car quality and consistency                                                                                                                                  	|


## Model Fit Statistics

### ROC Curve
ROC is a probability curve and AUC represents degree or measure of separability. ROC tells how much model is capable of distinguishing between classes. Higher the AUC, better the model is at predicting if a driver will end up in second position or not. The fitted Model perfoemd quite wel the are under ROC curve for the model was 0.950 [Source](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/)

![Q1_ROC_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_roc_curve.png)

### Precision and Recall Curve
The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate.[Sklearn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

![Q1_precision_recall_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_precision_recall_curve.png)

### beta Coefficients


The beta coefficient is the degree of change in the outcome variable for every 1-unit of change in the predictor variable. The t-test assesses whether the beta coefficient is significantly different from zero.

![Q1_beta_coefficients](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_beta_coefficients.png)


## Model Feature Importance 

| Features           	| Importance     	|
|--------------------	|----------------	|
| drivSecPosRM3      	| 0.703767601    	|
| driverRacePoints   	| 0.37644337     	|
| drivSecPosRM2      	| 0.2545058784   	|
| drivSecPosRM1      	| 0.2460997724   	|
| driverSeasonPoints 	| -0.01029259517 	|
| gridPosition       	| -0.09021629253 	|


#### Most Important Variable 

The most important variable from the above Matrix is actually the **driverRacePoints** since it was able to predict the log-odds most effectively. Another features shows up high in the coefficients i.e **drivSecPosRM3** has a marginally high p-value (0.067) and hence not so significant and we chose not to infer conclusions from it.

## Marginal Effects & Story

Marginal effects for continuous variables measure the instantaneous rate of change for any variable i.e for a unit change in the feature(Independent variable) what is the likelihood of increasing probability of the outcome to be 1.

| Features           	| Coefficients   	| Marginal Effect Interpretation                                                                                                                                                           	|
|--------------------	|----------------	|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| drivSecPosRM3      	| 0.703767601    	| For every unit increase in the binary value of second position in 3rd previous race i.e from 0 to 1, the logit or the log odds of the driver coming in second position increases by 0.70 	|
| driverRacePoints   	| 0.37644337     	| For every unit increase in driverRacePoints, the logit or the log odds of the driver coming in second position increases by 0.37                                                         	|
| drivSecPosRM2      	| 0.2545058784   	| For every unit increase in the binary value of second position in last-before race i.e from 0 to 1, the logit or the log odds of the driver coming in second position increases by 0.257 	|
| drivSecPosRM1      	| 0.2460997724   	| For every unit increase in the binary value of second position in last race i.e from 0 to 1, the logit or the log odds of the driver coming in second position increases by 0.246        	|
| driverSeasonPoints 	| -0.01029259517 	| For every unit increase in driverSeasonPoints, the logit or the log odds of the driver coming in second position decreases by -0.010                                                     	|
| gridPosition       	| -0.09021629253 	| For every unit increase in gridPosition  the logit or the log odds of the driver coming in second position decreases by 0.09                                                             	|



The story that one can knit together from above feature importances, marginal effects/coefficients is that a driver ending up in second position or not is determined by the above features and we can say that if a driver dis end up in second position or not in each of the last 2 races determine or increase his log-odds of taking second position in the next race by 0.25 units! which is very significant and the same is a driver goes behind in the grid position then it decreases the log-odds by 0.10 units

## Is it an Explanation or Simply an Association?

It is an explanation, given if we say that we have a good overview of all variables that are relevant in the context of F1 race, we are arguing for the logical relation between each and using them in our Models with reasoning. Shalit's point about measurement error and also the fact that explanatory (data) models are - and must be - simplifications of observed reality by construction. In this project we can say that we have a lot of features and information and hence we do have a good overview of how variables and combination of varaibles are interacting with each other. Hence when we model based on that we can say that the Model coefficients are more than simply association and are indeed explanatory. 

