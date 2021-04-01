# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## driver_results Model for Inference(Q1)
# MAGIC 
# MAGIC This file is for develping a model to predict the second posiiton in a rae for Inference(Q1)

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

import matplotlib.pyplot as plt

from numpy import savetxt

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# COMMAND ----------


driver_race_results_model =driver_race_results.select("*").toPandas()
driver_race_results_model = driver_race_results_model.dropna()

X_train, X_test, y_train, y_test = train_test_split(driver_race_results_model, driver_race_results_model[["positionOrder"]].values.ravel(), random_state=42)

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
  importance = pd.DataFrame(list(zip(driver_race_results_model.columns, rf.feature_importances_)), 
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