## Constructor Championship Inference

#### Background
Between year 1958 and 2010, Formula One constructors have been racing every year for the constructor's championship. In this period 202 teams and 610 drivers have raced in 775 races. Over time, some teams quit, while others joined. In the year of 2021, 10 teams are racing for constructor championship, among which only the team Ferrari raced every single race in the history of Formula One. Besides, some teams faced frequent change of ownership while others rebranded themselves periodically for various reasons, which all give the history of Formula One teams a lot of complications. 

Formula One rules have been constantly changing as well. From more than 10 racers per season for some team in the early years to the current day regulation that specifies two racers per team, the rules have evolved itself to make Formula One a spectacular sport to watch.  

#### Model
Given the history of constructor competition, the team is interested in understanding the key contributing factors to a constructor's success. Using a logistic regression model with two selected features, the team tries to come up with model coefficients that make the model best fit the championship data. 

The two features selected to explain the constructor championship in the logistic regression model are:
|Features|Description| Type| Coefficients|
|--------|-----------|-----|-------------|
|race_count|the number of races that the team drivers completed in the current season|numerical|5.93|
|avg_point|the average driver points per race the constructor obtained in the last season|numerical|2.75|

#### Feature Selection
`race_count` is the more important feature as compared to `avg_point` because it has a larger positive coefficient. The rest of the features in the features pool include circuits, constructor ranking, etc. They are discarded by applying Lasso regression with a regularization that reduces their coeffcients to 0. 

![Other Features](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/distribution_normalizedfeature.png)

#### Model Performance
The model handles the data relatively well. 

![ROC Curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/ROC-Curve.png)
![Precision Recall Curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Precision-Recall.png)

After the fitting the data, the area under ROC curve reaches 0.928. The F score is 0.945, the precision is 0.945 and the recall is 0.95. 

#### Marginal Effect of Features
![Marginal Effect - surface](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/marginaleffect_3dsurface.png)

The more race a team finishes, the more points the team get.
The better a team performs last season, the better it would perform this season.

![race_count](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/racecount_by_championship.png)
![lag1_avg_point](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/lag1driverpoint_by_championship.png)

#### Interpretation
The number of race completed may have some explanatory power: 
1. For the top tier constructors, their drivers tend to out-perform their peers and complete a race with some points. Although this is not always true, not completing a race, be it a car retirement or a crash, means no point for the team. While completing a race never equals winning the race, it does indicate for the top teams, that their drivers score some decent points. 

The average driver points per race the constructor obtained in the last season, indicating the constructor's last season performance, does not neccessarily explain a championship.
