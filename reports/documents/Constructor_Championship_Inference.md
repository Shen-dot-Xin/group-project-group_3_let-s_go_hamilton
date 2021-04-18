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
`race_count` is the more important feature as compared to `avg_point` because it has a larger positive coefficient. The boxplots show that the distribution of the two features broken down by championship and indicate that between the groups the distributions are quite different. 
![race_count](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/racecount_by_championship.png)
![lag1_avg_point](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/lag1driverpoint_by_championship.png)

The rest of the features in the features pool include circuits, constructor ranking, etc. They are discarded by applying Lasso regression with a regularization that reduces their coeffcients to 0. 

<p align="center">
  <img width="460" height="460" src="https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/distribution_normalizedfeature.png">
</p>

#### Model Performance
The model handles the data relatively well. 

![ROC Curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/ROC-Curve.png)
![Precision Recall Curve](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Precision-Recall.png)

After the fitting the data, the area under ROC curve reaches 0.928. The F score is 0.945, the precision is 0.945 and the recall is 0.95. 

#### Marginal Effect of Features
<p align="center">
  <img width="500" height="460" src="https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/marginaleffect_3dsurface.png">
</p>

Generally, the more race a team finishes, the better a team performs last season, the larger the probability it will win the championship for this season. Here, the normalized data for `race_count` and `avg_point` are mapped on the axis. That is to say, the marginal effect of the features for any team depends on its comparison with other teams. The marginal increase in probability is largest when `race_count` and `avg_point` are at their largest as compared to the performance of other constructors in the season. 

#### Interpretation
The number of race completed may have some explanatory power. For the top tier constructors, their drivers tend to out-perform their peers and complete a race with some points. Although this is not always true, not completing a race, be it a car retirement or a crash, means no point for the team. While completing a race never equals winning the race, it does indicate for the top teams, that their drivers score some decent points. 

However, when competitions are fierce and the top tier constructors are fighting a close battle, race completion does not neccessarily have any significance.

The average driver points per race the constructor obtained in the last season, indicating the constructor's last season performance, explains a championship in a limited way. While advantage of a constructor over the others could carry through from year to year, it is not neccessarily so, given a lot of external factors such as rule change, weather, accidents, etc., which are beyond control. 

![Championship](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Championship_fitted.png)
![Fitted](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/blob/main/reports/figures/Fit_by_Championship.png)

In fact, giving weights to last year performance result in a better of the model over constructors with consecutive and multiple wins, such as Ferrari, Williams and Mclaren. The three are also among the teams with the least rebranding and name-changing drama, which could indicate some consistancy in their performance. 
