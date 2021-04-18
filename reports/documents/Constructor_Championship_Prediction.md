### Constructor_Championship_Prediction
1. Describe your model, and explain how you selected the features that were selected
2. Provide statistics that show how good your model is at predicting, and how well it performed predicting constructors
success between 2011 and 2017
3. The most important variable in (3) is bound to also be included in your predictive model. Provide marginal effects or
some metric of importance for this variable and make an explicit comparison of this value with the values that you
obtained in (3). How different are they? Why are they different?


### 1. Describe your model, and explain how you selected the features that were selected
  I firstly compared the Support Vector Classifier (SVC) and Logistic Regression Models using training data (1950-2010) and test data (2011-2017). The mean cross-validation score of both model are 0.94, while the test set score of logistic regression model is 0.94, higher than that of SVC model(0.91), both values are generated from best parameters. Therefore, I chose logistic regression model to do the regression. And the features I used were normalized in order to improve the accuracy of my model.
  
  The features I used in regression including: 
  - 'avg_fastestspeed': The mean value of selected drivers' fastest speed in this season. This can represent the strategy that the drivers would use to win the race.
  - 'avg_fastestlap': The mean value of selected drivers' fastest lap in this season. It is also a kind of strategy to win the race.
  - 'race_count': Finished races in this season. More races completed means more meaningful points would possibly be counted.
  - 'engineproblem': The number of engine problems occured in this season, which shows the reliability of the constructor's technique.
  - 'lag1_ptc': Whether the constructor participated in last season, representing whether the constructor is experienced in participating in F1 competition.
  - 'lag1_avg': The average points the constructor won in last season, representing the constructor's performance in last season.
  - 'lag1_pst': The average positions the constructor had in last season, showing another dimension of constructor's performance.
  - 'lag2_pst': The average positions the constructor had in the year before last, showing another dimension of constructor's performance.


### 2. Provide statistics that show how good your model is at predicting, and how well it performed predicting constructors success between 2011 and 2017
   - Accuracy of my model: 0.955
   - False positive rate: 0.589
   - test area under ROC curve: 0.976
   - true positive rate: 0.955
   - Test set score: 0.94 
   - The importance of each feature is as follows: "race_count" is 5.46, "lag1_avg" is 2.26, "avg_fastestspeed" is 1.48, "lag2_pst" is 1.10, “engineproblem” is -0.73, "avg_fastestlap" is -2.08, “lag1_pst” is -2.17 and "lag1_ptc" equals 0.
  The top 3 important features in my model is race_count, "lag1_avg" and "lag1_pst". From these features, there are several comclusions that could be given. 
  1) The constructor's average performance in last season, including average points and positions. This means that if a constructor is good at selecting great drivers in last season, the possiblity that it win this season would increase. 
  2) The more races the constructor's drivers completed in this season, the more points would be counted, and then the possibility of the constructor win this season will grow up as well.


### 3. The most important variable in (3) is bound to also be included in your predictive model. Provide marginal effects or some metric of importance for this variable and make an explicit comparison of this value with the values that you obtained in (3). How different are they? Why are they different?
   - Marginal effects of "lag1_avg" and "race_count"
    
   - Comparing variables in question (3) and (4)
    The two common features of Models in question (3) and (4) are "lag1_avg" and "race_count". In model (3), the importance of these two features are 2.72 for "lag1_avg" and 3.42 for "race_count". In model (4), the importance of "lag1_avg" goes down to 2.26. It may be beacuse of more features representing performance in last season were introduced in model (4), for instance, "lag1_pst" could also show the perfomance of the drivers selected by the constructor in last season, and this feature is important in my model as well. As for "race_count", the importance of it raise up in question (4) than that of question (3). I suppose the difference is due to the selection of other variables. "avg_fastestlap", "lag1_pst" and "engineproblem" are key features in determining whether the constructor would lose the season. Since these variables with negative importance were introduced in Model (4), the importance of "race_count" may be increased along with it.
    

