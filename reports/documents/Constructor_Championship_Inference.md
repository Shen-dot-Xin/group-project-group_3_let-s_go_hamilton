### Constructor Championship Inference

Between year 1958 and 2010, Formula One constructors have been racing every year for the constructor's championship. In this period 202 teams and 610 drivers have raced in 775 races. Over time, some teams quit, while others joined. In the year of 2021, 10 teams are racing for constructor championship, among which only the team Ferrari raced every single race in the history of Formula One. Besides, some teams faced frequent change of ownership while others rebranded themselves periodically for various reasons, which all give the history of Formula One teams a lot of complications. 

Formula One rules have been constantly changing as well. From more than 10 racers per season for some team in the early years to the current day regulation that specifies two racers per team, the rules have evolved itself to make Formula One a spectacular sport to watch.  

Given the history of constructor competition, the team is interested in understanding the key contributing factors to a constructor's success. Using a logistic regression model with two selected features, the team tries to come up with model coefficients that make the model best fit the championship data. 

The two features selected to explain the constructor championship in the logistic regression model are:
|Features|Description| Type| Coefficients|
|--------|-----------|-----|-------------|
|race_count|the number of races that the team drivers completed in the current season|numerical|5.93|
|avg_point|the average driver points per race the constructor obtained in the last season|numerical|2.75|

The rest of the features in the features pool include circuits, constructor ranking, etc. They are discarded by applying Lasso regression with a regularization that reduces their coeffcients to 0. 

The model handles the data relatively well. 

![ROC curve]
![alt text](https://github.com/QMSS-GR5069-Spring2021/group-project-group_3_let-s_go_hamilton/reports/figures/ROC-Curve.png)

1. Area under ROC curve
2. F measure
3. Confusion Matrix

- What is the most important features? How is it determined?
1. the number of races is more important than the average, because it has a larger positive coefficients

- Provide some marginal effects for the variable you identified and interpret it in the context of f1 racing (story telling)
It is very 

The more race a team finishes, the more points the team get.
The better a team performs last season, the better it would perform this season.

For teams which have won a championship, how is the breakdown of true and false prediction?
What teams are predicted to win the championship, but actually did not?

- Does it make sense to think of it as an explanation for why a constructor wins a season? Or is it simply association?

The number of race completed may have some explanatory power: 
1. For the top tier constructors, their drivers tend to out-perform their peers and complete a race with some points. Although this is not always true, not completing a race, be it a car retirement or a crash, means no point for the team. While completing a race never equals winning the race, it does indicate for the top teams, that their drivers score some decent points. 

The average driver points per race the constructor obtained in the last season, indicating the constructor's last season performance, does not neccessarily explain a championship.
1. Last year is last year, this year is this year. Almost every year, FIA make rule changes or regulates the car design, etc., which are sometime designed to deprive some teams of their advantages.  
2. However, the points a constructor win last year do indicates the constructor's performance, which encompass numerous aspects contributing to its success, including the quality of the car, the strategy, the choice of driver lineups, availability of funding, etc. 
