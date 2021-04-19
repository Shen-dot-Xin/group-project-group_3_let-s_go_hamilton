# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## RF_binary_classifier_driver_results Prediction Model_part_1
# MAGIC 
# MAGIC At the beginning, we use the Random Forest Classifier model (see part_1). To predict whether the driver comes in the second place, we use binary prediction output (second place = 1, otherwise = 0). Although the model provides unusually high accuracy (0.95), the actual prediction power of the model is low. The binary learning output makes the model to predict all the result as 0 and still get a high accuracy, the other measurement scores are quite low (almost all 0).

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat,when
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import tempfile
from pyspark.sql.types import DoubleType

import matplotlib.pyplot as plt

from numpy import savetxt

dbutils.library.installPyPI("mlflow", "1.14.0")
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score,f1_score, balanced_accuracy_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

# MAGIC %md
# MAGIC Data Cleansing and Features selection

# COMMAND ----------

#Reading Drivers data
drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True, inferSchema=True)

#Reading Races data
races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True, inferSchema=True)

#Reading Constructors data
constructors = spark.read.csv('s3://columbia-gr5069-main/raw/constructors.csv', header=True, inferSchema=True)


# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driverRaceDF= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driverRaceDF = driverRaceDF.drop('_c0')


#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Dropping the pitstops data since this data is only available post 2011 races
driverRaceDF = driverRaceDF.drop("totPitstopDur","avgPitstopDur","countPitstops","firstPitstopLap","raceDate")

# COMMAND ----------

# Dropping similar columns to target variables
driverRaceDF = driverRaceDF.drop("positionOrder","driverRacePoints","drivSecPosCat","raceLaps","finishPosition","driverSeasonPoints")
#driverRaceDF = driverRaceDF.withColumn('drivSecPosCat', driverRaceDF['drivSecPosCat'].cast(DoubleType()))

# COMMAND ----------

#Splitting the data frame into Train and Test data based on the requirements of the Project
driverRaceTrainDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)

#Test data according to the question
driverRaceTestDF = driverRaceDF.filter(driverRaceDF.raceYear > 2010)
driverRaceTestDF = driverRaceTestDF.filter(driverRaceTestDF.raceYear < 2018)

# COMMAND ----------

# Converting the Spark Dataframes into Pandas Dataframes
driverRaceTrainDF =driverRaceTrainDF.select("*").toPandas()
driverRaceTestDF =driverRaceTestDF.select("*").toPandas()

# COMMAND ----------

X_train = driverRaceTrainDF.drop(['drivSecPos'], axis=1)
X_test = driverRaceTestDF.drop(['drivSecPos'], axis=1)
y_train = driverRaceTrainDF['drivSecPos']
y_test = driverRaceTestDF['drivSecPos']

# COMMAND ----------

 with mlflow.start_run():
  
    # Set the model parameters. 
    n_estimators = 1000
    max_depth = 5
    max_features = 5

    # Create and train model.
    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)

    
    #accuracy_score,f1_score, balanced_accuracy_score,precision_score,recall_score,roc_auc_score
    # Create metrics
    accuracy_score = accuracy_score(y_test, predictions)
    balanced_accuracy_score = balanced_accuracy_score(y_test, predictions)
    f1_score = f1_score(y_test, predictions)
    precision_score = precision_score(y_test, predictions)
    recall_score = recall_score(y_test, predictions)
    roc_auc_score = roc_auc_score(y_test, predictions)
    print("  accuracy_score: {}".format(accuracy_score))
    print("  balanced_accuracy_score: {}".format(balanced_accuracy_score))
    print("  f1_score: {}".format(f1_score))
    print("  precision_score: {}".format(precision_score))
    print("  recall_score: {}".format(recall_score))
    print("  roc_auc_score: {}".format(roc_auc_score))
    

    # Log metrics
    mlflow.log_metric("accuracy_score", accuracy_score)
    mlflow.log_metric("balanced_accuracy_score", balanced_accuracy_score)  
    mlflow.log_metric("f1_score", f1_score) 
    mlflow.log_metric("precision_score", precision_score) 
    mlflow.log_metric("recall_score", recall_score)
    mlflow.log_metric("roc_auc_score", roc_auc_score)

    # Create feature importance
    importance = pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), 
                                columns=["Feature", "Importance"]
                                ).sort_values("Importance", ascending=False)

    # Log importances using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    temp_name = temp.name
    try:
      importance.to_csv(temp_name, index=False)
      mlflow.log_artifact(temp_name, "feature-importance.csv")
    finally:
      temp.close() # Delete the temp file

    # Create plot
    fig, ax = plt.subplots()

    sns.residplot(predictions, y_test, lowess=True)
    plt.xlabel("Predicted Driver Position")
    plt.ylabel("Residual")
    plt.title("Residual Plot")

    # Log residuals using a temporary file
    temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
    temp_name = temp.name
    try:
      fig.savefig(temp_name)
      mlflow.log_artifact(temp_name, "residuals.png")
    finally:
      temp.close() # Delete the temp file

    display(fig)

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

# MAGIC %md
# MAGIC Thus, we change to the random forest non-binary classifier model.