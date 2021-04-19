
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

## Most Important Variable 

## Marginal Effects & Story

## Is it an Explanation or Simply an Association?
