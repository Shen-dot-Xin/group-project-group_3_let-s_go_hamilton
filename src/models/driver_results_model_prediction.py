# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## driver_results Prediction Model
# MAGIC 
# MAGIC This file is for creating a Model to Predict (Q2) if a Driver will end up in second position.

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

driverRaceDF = driverRaceDF.withColumn("finishPosition", when(driverRaceDF["finishPosition"] == 999, 20).otherwise(driverRaceDF["finishPosition"]))
driverRaceDF = driverRaceDF.withColumn("finishPositionRM1", when(driverRaceDF["finishPositionRM1"] == 999, 20).otherwise(driverRaceDF["finishPositionRM1"]))  
driverRaceDF = driverRaceDF.withColumn("finishPositionRM2", when(driverRaceDF["finishPositionRM2"] == 999, 20).otherwise(driverRaceDF["finishPositionRM2"]))  
driverRaceDF = driverRaceDF.withColumn("finishPositionRM3", when(driverRaceDF["finishPositionRM3"] == 999, 20).otherwise(driverRaceDF["finishPositionRM3"]))
#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Dropping the pitstops data since this data is only available post 2011 races
driverRaceDF = driverRaceDF.drop("totPitstopDur","avgPitstopDur","countPitstops","firstPitstopLap","raceDate")

# COMMAND ----------

# Dropping similar columns to target variables
driverRaceDF = driverRaceDF.drop("positionOrder","driverRacePoints","drivSecPosCat","raceLaps","driverSeasonPoints","drivSecPos")
#,"drivSecPosRM3","drivSecPosRM2","drivSecPosRM1"
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

X_train = driverRaceTrainDF.drop(['finishPosition'], axis=1)
X_test = driverRaceTestDF.drop(['finishPosition'], axis=1)
y_train = driverRaceTrainDF['finishPosition']
y_test = driverRaceTestDF['finishPosition']

# COMMAND ----------

 with mlflow.start_run():
  
    # Set the model parameters. 
    n_estimators = 1000
    max_depth = 5
    max_features = 10

    # Create and train model.
    rf = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
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

#score for classifier model
score = rf.score(X_test, y_test)
log_p = rf.predict_proba(X_test)
print(score)
print(log_p)

# COMMAND ----------

# MAGIC %md
# MAGIC Model Testing (to select the best parameters)

# COMMAND ----------

from sklearn.ensemble import 


# COMMAND ----------


# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

def rf_reg(run_name, params, X_train, X_test, y_train, y_test):
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
  with mlflow.start_run(run_name=run_name):
  
    # Set the model parameters. 
    #n_estimators = 1000
    #max_depth = 5
    #max_features = 5

    # Create and train model.
    rf = RandomForestRegressor(**params)
    #rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = predict_log_proba(y_test, predictions)
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



# COMMAND ----------

params = {
  "n_estimators": 100,
  "max_depth": 5,
  "random_state": 42}

rf_reg("Second Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Data of the Best Model

# COMMAND ----------

 with mlflow.start_run():
  
    # Set the model parameters. 
    n_estimators = 1000
    max_depth = 5
    max_features = 10

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

predictions = rf.predict(X_test)
driverRaceDFPred = X_test
driverRaceDFPred['predictions'] = pd.Series(predictions, index=driverRaceDFPred.index).round(0)
driverRaceDFPred['finishPosition'] = pd.Series(y_test, index=driverRaceDFPred.index)

# COMMAND ----------

driverRaceDFPred=spark.createDataFrame(driverRaceDFPred) 

# COMMAND ----------

driverRaceDFPred= driverRaceDFPred.join(drivers.select(col("driverId"), concat(drivers.forename,drivers.surname).alias("driverName"), col("nationality").alias("driverNat")), on=['driverId'],how="left").join(constructors.select(col("constructorId"),col("name").alias("constructorName"),col("nationality").alias("constructorNat")), on=['constructorId'], how="left").join(races.select(col("raceId"), col("name").alias("raceName"),col("round").alias("raceRound"), col("date").alias("raceDate")), on=['raceId'], how="left")

# COMMAND ----------

# MAGIC %md
# MAGIC Create the prediction result

# COMMAND ----------

# Creating a Binary column that says if a driver finished second or not
driverRaceDFPred = driverRaceDFPred.withColumn("drivSecPosPred", when(driverRaceDFPred.predictions==2,1) .otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC Save the dataframe

# COMMAND ----------

driverRaceDFPred.write.format('jdbc').options(
      url='jdbc:mysql://gc-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/gc2897_gr5069',
      driver='com.mysql.jdbc.Driver',
      dbtable='Final_Driver_Predit_Q2_G3',
      user='admin',
      password='12345678').mode('overwrite').save()

# COMMAND ----------

driverRaceDFPred_return = spark.read.format("jdbc").option("url", "jdbc:mysql://gc-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/gc2897_gr5069") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "Final_Driver_Predit_Q2_G3") \
    .option("user", "admin").option("password", "12345678").load()

# COMMAND ----------

display(driverRaceDFPred_return)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualization is in the Superset