# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Driver Results Model for Inference(Q1)
# MAGIC 
# MAGIC This file is for develping a model to predict the second posiiton in a rae for Inference(Q1)

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import pandas as pd
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

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driverRaceDF= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driverRaceDF = driverRaceDF.drop('_c0')

# COMMAND ----------

driverRaceTrainDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)
driverRaceTestDF = driverRaceDF.filter(driverRaceDF.raceYear > 2010)

# COMMAND ----------

driverRaceTrainDF= driverRaceTrainDF.sort(driverRaceTrainDF.raceYear.desc())
driverRaceTrainDF.display()

# COMMAND ----------

driverRaceTrainDF.describe().display()

# COMMAND ----------

