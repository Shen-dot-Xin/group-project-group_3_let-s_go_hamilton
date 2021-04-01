# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## driver_results exploration and visualization
# MAGIC 
# MAGIC This file is for doing exploration on the driver_results data to understand it more for both the Inference (Q1) and Prediction (Q2)

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp
#import plotly

# COMMAND ----------

#Reading the Processed data from Group's S3 bucket
driver_race = spark.read.csv('s3://pp-gr5069/processed/driver_race_results_exp.csv', header=True)