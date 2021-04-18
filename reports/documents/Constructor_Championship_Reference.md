### Constructor Championship Inference

- Describe the model, explain why each feature was selected
- Provide statistic that show how well this model fit the data
- What is the most important features? How is it determined?
- Provide some marginal effects for the variable you identified and interpret it in the context of f1 racing (story telling)
- Does it make sense to think of it as an explanation for why a constructor wins a season? Or is it simply association?

- Describe the model, explain why each feature was selected

The model selected is a logistic regression model with two features: 
1. the number of races that the drivers completed in the current season  
2. the average driver points per race the constructor obtained in the last season

The other features are discarded by applying Lasso regression with a regularization that reduces the coeffcients of unnecessary features to 0. 

- Provide statistic that show how well this model fit the data

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
