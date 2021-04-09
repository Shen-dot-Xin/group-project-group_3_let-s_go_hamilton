# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Driver Results data exploration and visualization
# MAGIC 
# MAGIC This file is for doing exploration on the driver_results data to understand it more for both the Inference (Q1) and Prediction (Q2)

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat,count, isnull,isnan,when
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType

from pyspark.mllib.stat import Statistics
import numpy as np
import matplotlib as mp
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import tempfile
import matplotlib.pyplot as plt
from matplotlib import cm

from numpy import savetxt

from io import StringIO # python3; python2: BytesIO 
import boto3
s3 = boto3.client('s3')


# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driver_race_results_mod_feat= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driver_race_results_mod_feat = driver_race_results_mod_feat.drop('_c0')

#Reading Drivers Performance data prepared for modeling with features
driver_race_results_info= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_info.csv', header=True, inferSchema=True)
driver_race_results_info = driver_race_results_info.drop('_c0')

#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Create a view or table out 
#driver_race_results_mod_feat = spark.createDataFrame(driver_race_results_mod_feat)
driver_race_results_mod_feat.createOrReplaceTempView("driver_race_results_mod_view")

# Create a view or table out 
#driver_race_results_info = spark.createDataFrame(driver_race_results_info)
driver_race_results_info.createOrReplaceTempView("driver_race_results_info_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC SELECT *
# MAGIC FROM `driver_race_results_view` mas
# MAGIC LEFT JOIN 
# MAGIC (SELECT raceId, MAX(raceLaps) AS max_raceLaps, AVG(raceDuration) as avg_raceLaps 
# MAGIC    FROM `driver_race_results_mod_view` GROUP BY raceId) as agg ON mas.raceId = agg.raceId
# MAGIC WHERE driverRacePoints = 0 
# MAGIC ORDER BY mas.raceId DESC, positionOrder;

# COMMAND ----------

num_features = [colname for colname, coltype in driver_race_results_info.toPandas().dtypes.iteritems() if coltype == 'int32']
driver_race_results_info.select(num_features).describe().toPandas().transpose()

# COMMAND ----------

fig= plt.figure(figsize=(24,24)) ## Plot Size
st = fig.suptitle("Distribution of Features", fontsize=50, verticalalignment= 'center') # Plot main title

for col, num in zip(driver_race_results_info.toPandas().describe().columns,range(1,11)):
  ax= fig.add_subplot(3,4,num)
  ax.hist(driver_race_results_info.toPandas()[col])
  plt.style.use('dark_background')
  plt.grid(False)
  plt.xticks(rotation=45, fontsize=20)
  plt.yticks(rotation=45)
  plt.title(col.upper(), fontsize=20)

plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85, hspace=0.4)
plt.show()


# COMMAND ----------

num_features = [colname for colname, coltype in driver_race_results_info.toPandas().dtypes.iteritems() if coltype == 'int32' or coltype == 'float64']
num_features_df = driver_race_results_info.select(num_features)
driver_race_results_info.toPandas().head()

# COMMAND ----------

col_names = num_features_df.columns
features = num_features_df.rdd.map(lambda row: row[0:])
corr_mat = Statistics.corr(features, method='pearson')
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names

corr_df