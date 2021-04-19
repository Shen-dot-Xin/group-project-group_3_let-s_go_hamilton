
## Model Description 


## Feature Selection and reasoning

| Feature            	| Description and Reason                                                                                                                                                                                                                                                                                   	|
|--------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------	|
| resultId           	| This features acts as a IID for each record, choosen just for convenience and when running trials with other tree based models.                                                                                                                                                                          	|
| raceYear           	| This feature should help probably show the growing or receeding pattern in ones performance                                                                                                                                                                                                              	|
| driverRacePoints   	| This feature says the points a driver got in a race, it is directly proportional to the position a driver finishes in and specifically for our prediction of second position which has 18 points.                                                                                                        	|
| driverSeasonPoints 	| This feature shows points in the season thus far for the Driver in a cumulative fashion, it was chosen since it may help determine if a second position is going to be a one off performance or one with some repeatability.                                                                             	|
| driverSeasonWins   	| This feature shows wins in the season thus far for the Driver in a cumulative fashion, it was chosen since it may help determine if a win is a one off performance or one with some repeatability.                                                                                                       	|
| constSeasonPoints  	| This feature shows points in the season thus far for the Constructor in a cumulative fashion, it was chosen since it may help determine the probability of a team member getting a second position which gets them 18 points.                                                                            	|
| constSeasonWins    	| This feature shows wins in the season thus far for the Constructor in a cumulative fashion, it was chosen since it may help determine the probability of a team member getting a second position like Mercedes won so many titles and Bottas often came second right behind the champion Lewis Hamilton. 	|
| drivSecPosRM1      	| A binary column that says if a driver finished second or not in the previous race! This might massively determine the nature and the car quality and consistency                                                                                                                                         	|
| drivSecPosRM2      	| A binary column that says if a driver finished second or not in the last-before race! This might massively determine the nature and the car quality and consistency                                                                                                                                      	|
| drivSecPosRM3      	| A binary column that says if a driver finished second or not in the two-races-before one! This might massively determine the nature and the car quality and consistency                                                                                                                                  	|


## Model Fit Statistics

![Q1_ROC_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_ROC_curve.png)

![Q1_precision_recall_curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Q1_preceision_recall.png)

### Model Feature Importance 

| Feature       	    |Feature Coefficients|
|--------------------	|------------------	|
| drivSecPosRM3       | 0.9107840875      |
| drivSecPosRM2      	| 0.6872625734     	|
| drivSecPosRM1      	| 0.4855982405     	|
| driverRacePoints   	| 0.4443269082     	|
| constSeasonWins    	| 0.02519453535    	|
| resultId           	| 9.97E-08         	|
| driverSeasonPoints 	| -0.0001155720202 	|
| raceYear           	| -0.0006878917667 	|
| constSeasonPoints  	| -0.001969819308  	|
| driverSeasonWins   	| -0.2049255411    	|



## Most Important Variable 

## Marginal Effects & Story
Marginal effects for continuous variables measure the instantaneous rate of change for any variable

## Is it an Explanation or Simply an Association?
