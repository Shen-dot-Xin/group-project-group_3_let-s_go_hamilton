# Databricks notebook source
#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp
import pandas as pd

# COMMAND ----------

#Reading the Processed data from Group's S3 bucket
driver_race_results = pd.read_csv('s3://group3-gr5069/processed/driver_race_results.csv', header=True)

# COMMAND ----------

