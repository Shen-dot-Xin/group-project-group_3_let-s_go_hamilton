# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ##Feature Engineering & Preparation for Inference Question (Q1)
# MAGIC 
# MAGIC This file is for feature engineering and preparation for Inference Modelling to predict who will come in second position in a race.

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat, to_date,when, lead,lag,desc
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp
import pandas as pd

from io import StringIO # python3; python2: BytesIO 
import boto3
s3 = boto3.client('s3')
import pandas as pd

# COMMAND ----------

#Reading Drivers Performance data
driver_race_results_mod_sp= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod.csv', header=True, inferSchema=True)
driver_race_results_mod_sp = driver_race_results_mod_sp.drop('_c0')

#Reading Drivers Performance data
driver_race_results_info_sp= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_info.csv', header=True, inferSchema=True)
driver_race_results_info_sp = driver_race_results_info_sp.drop('_c0')

# COMMAND ----------

# Changing the data type to Date 
driver_race_results_mod_sp = driver_race_results_mod_sp.withColumn("raceDate", to_date(col("raceDate"),"yyyy-mm-dd"))

# COMMAND ----------

# Creating a Binary column that says if a driver finished second or not
driver_race_results_mod_sp = driver_race_results_mod_sp.withColumn("drivSecPos", when(driver_race_results_mod_sp.finishPosition==2,1) .otherwise(0))

# COMMAND ----------

# Creating a Column to get the Drivers finish postion in the Race_Minus_1 (RM1) race
driver_race_results_mod_sp = driver_race_results_mod_sp.withColumn('finishPositionRM1', lead('finishPosition', 1, 999).over(Window.partitionBy('driverId').orderBy(desc('raceDate'))))

# Creating a Column to get the Drivers finish postion in the Race_Minus_2 (RM2) race
driver_race_results_mod_sp = driver_race_results_mod_sp.withColumn('finishPositionRM2', lead('finishPosition', 2, 999).over(Window.partitionBy('driverId').orderBy(desc('raceDate'))))

# Creating a Column to get the Drivers finish postion in the Race_Minus_3 (RM3) race
driver_race_results_mod_sp = driver_race_results_mod_sp.withColumn('finishPositionRM3', lead('finishPosition', 3, 999).over(Window.partitionBy('driverId').orderBy(desc('raceDate'))))

# COMMAND ----------

#Ordering the dataframe for view convenience and viewing data to check 
driver_race_results_mod_sp= driver_race_results_mod_sp.sort(driver_race_results_mod_sp.raceId.desc(), driver_race_results_mod_sp.positionOrder)
driver_race_results_mod_sp.display()

# COMMAND ----------

#Ordering the dataframe for view convenience and viewing the infor data frame just to reference, compare and check columns 
driver_race_results_info_sp= driver_race_results_info_sp.sort(driver_race_results_info_sp.raceId.desc(), driver_race_results_info_sp.positionOrder)
driver_race_results_info_sp.display()

# COMMAND ----------

# Writing this data to S3 bucket
driver_race_results_mod_sp =driver_race_results_mod_sp.select("*").toPandas()
bucket = "group3-gr5069" # already created on S3
csv_buffer = StringIO()
driver_race_results_mod_sp.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'processed/driver_race_results_mod_feat.csv').put(Body=csv_buffer.getvalue())

# COMMAND ----------

