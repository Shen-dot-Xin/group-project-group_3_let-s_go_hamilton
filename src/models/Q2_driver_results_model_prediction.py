# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## driver_results Prediction Model
# MAGIC 
# MAGIC This file is for creating a Model to Predict (Q2) if a Driver will end up in second position.

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
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

from sklearn.ensemble import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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
driverRaceDF = driverRaceDF.drop("positionOrder","driverRacePoints")
driverRaceDF = driverRaceDF.withColumn('drivSecPosCat', driverRaceDF['drivSecPosCat'].cast(DoubleType()))

# COMMAND ----------

## Transforming a selection of features into a vector using VectorAssembler.
vecAssembler = VectorAssembler(inputCols = ['resultId', 'raceYear','constructorId','raceId','driverStPosition','gridPosition'
                                             'driverSeasonPoints', 'driverSeasonWins',
                                            'constSeasonPoints', 'constSeasonWins', 
                                            'drivSecPosRM1','drivSecPosRM2','drivSecPosRM3'], outputCol = "vectorized_features")
driverRaceDF = vecAssembler.transform(driverRaceDF)

# COMMAND ----------

#Splitting the data frame into Train and Test data based on the requirements of the Project
driverRaceTrainDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)
driverRaceTestDF = driverRaceDF.filter(driverRaceDF.raceYear > 2010)

# COMMAND ----------

# Converting the Spark Dataframes into Pandas Dataframes
driverRaceTrainDF =driverRaceTrainDF.select("*").toPandas()
driverRaceTestDF =driverRaceTestDF.select("*").toPandas()

# COMMAND ----------

X_train = driverRaceTrainDF.drop(['finishPosition'], axis=1)
X_test = driverRaceTestDF.drop(['finishPosition'], axis=1)
y_train = driverRaceTrainDF['finishPosition']
y_test = driverRaceTestDF['finishPosition']

# COMMAND ----------

## Run 1 - Run Params: {n_estimators = 100; max_depth = 6; max_features = 3}

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test)
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  mae = mean_absolute_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)
  print("  mse: {}".format(mse))
  print("  mae: {}".format(mae))
  print("  R2: {}".format(r2))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  mlflow.log_metric("mae", mae)  
  mlflow.log_metric("r2", r2) 
  
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
  plt.xlabel("Predicted values for Price ($)")
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

