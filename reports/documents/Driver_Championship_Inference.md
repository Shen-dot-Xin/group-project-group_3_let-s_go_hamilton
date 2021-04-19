
## Model Description 


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

![Q1_ROC_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_ROC_curve.png)

![Q1_precision_recall_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_preceision_recall.png)

### Model Feature Importance 

| Features           	| Importance     	|
|--------------------	|----------------	|
| drivSecPosRM3      	| 0.703767601    	|
| driverRacePoints   	| 0.37644337     	|
| drivSecPosRM2      	| 0.2545058784   	|
| drivSecPosRM1      	| 0.2460997724   	|
| driverSeasonPoints 	| -0.01029259517 	|
| gridPosition       	| -0.09021629253 	|


## Most Important Variable 

## Marginal Effects & Story
Marginal effects for continuous variables measure the instantaneous rate of change for any variable i.e for a unit change in the feature(Independent variable) what is the likelihood of increasing probability of the outcome to be 1.



## Is it an Explanation or Simply an Association?
