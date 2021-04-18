### Constructor_Championship_Prediction
- Describe your model, and explain how you selected the features that were selected
- Provide statistics that show how good your model is at predicting, and how well it performed predicting constructors
success between 2011 and 2017
- The most important variable in (3) is bound to also be included in your predictive model. Provide marginal effects or
some metric of importance for this variable and make an explicit comparison of this value with the values that you
obtained in (3). How different are they? Why are they different?


### Describe your model, and explain how you selected the features that were selected
  I firstly compared the Support Vector Classifier (SVC) and Logistic Regression Models using training data (1950-2010) and test data (2011-2017). The mean cross-validation score of both model are 0.94, while the test set score of logistic regression model is 0.95, higher than that of SVC model(0.91), both values are generated from best parameters. Therefore, I chose logistic regression model to do the regression.
  The features I used in regression including: 
  - 'avg_fastestspeed': The mean value of selected drivers' fastest speed in this season. This can represent the strategy that the drivers would use to win the race.
  - 'avg_fastestlap': The mean value of selected drivers' fastest lap in this season. It is also a kind of strategy to win the race.
  - 'race_count': Finished races in this season. More races completed means more meaningful points would possibly be counted.
  - 'engineproblem': The number of engine problems occured in this season, which shows the reliability of the constructor's technique.
  - 'lag1_ptc': Whether the constructor participated in last season, representing whether the constructor is experienced in participating in F1 competition.
  - 'lag1_avg': The average points the constructor won in last season, representing the constructor's performance in last season.
  - 'lag1_pst': The average positions the constructor had in last season, showing another dimension of constructor's performance.
  - 'lag2_pst': The average positions the constructor had in the year before last, showing another dimension of constructor's performance.


### Provide statistics that show how good your model is at predicting, and how well it performed predicting constructors success between 2011 and 2017
