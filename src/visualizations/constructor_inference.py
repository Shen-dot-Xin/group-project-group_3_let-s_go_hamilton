# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship, Features

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.sql import Window
from pyspark.sql.functions import lag, col, asc, min, max, when

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# COMMAND ----------

df = spark.read.csv('s3://group3-gr5069/interim/constructor_features.csv', header = True, inferSchema = True)

# COMMAND ----------

display(df)

# COMMAND ----------

cols_to_plot =  ['avg_fastestspeed', 
                     'avg_fastestlap',
                     'race_count',
                     'engineproblem',
                     'avgpoints_c',  
                     'unique_drivers',
                     'position']

# COMMAND ----------

plot_df = df.toPandas()

# COMMAND ----------

fig, axes = plt.subplots(3, 3, figsize = (10,10))
plt.style.use('seaborn')
for i in range(3):
  for j in range(3):
    if 3*i+j >6:
      pass
    else:
      axes[i,j].hist(x = cols_to_plot[3*i+j], data = plot_df, rwidth=0.8)
      axes[i,j].set_xlabel( cols_to_plot[3*i+j])
plt.show()

# COMMAND ----------

cols_to_normalize = ['avg_fastestspeed', 
                     'avg_fastestlap',
                     'race_count',
                     'engineproblem',
                     'avgpoints_c',  
                     'unique_drivers',
                     'position',
                     'lag1_avg',
                     'lag2_avg', 
                     'lag1_pst',
                     'lag2_pst']

# COMMAND ----------

w = Window.partitionBy('year')
for c in cols_to_normalize:
    df = (df.withColumn('mini', min(c).over(w))
        .withColumn('maxi', max(c).over(w))
        .withColumn(c,  when(col('maxi') == col('mini'), 0).otherwise(((col(c) - col('mini')) / (col('maxi') - col('mini')))))
        .drop('mini')
        .drop('maxi'))

# COMMAND ----------

cols_to_plot =  ['avg_fastestspeed', 
                     'avg_fastestlap',
                     'race_count',
                     'engineproblem',
                     'avgpoints_c',  
                     'unique_drivers',
                     'position']

# COMMAND ----------

plot_df = df.toPandas()

# COMMAND ----------

fig, axes = plt.subplots(3, 3, figsize = (10,10))
plt.style.use('seaborn')
for i in range(3):
  for j in range(3):
    if 3*i+j >6:
      pass
    else:
      axes[i,j].hist(x = cols_to_plot[3*i+j], data = plot_df, rwidth=0.8)
      axes[i,j].set_xlabel( cols_to_plot[3*i+j])
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Marginal Effect

# COMMAND ----------

predDF_final.to_csv("s3://group3-gr5069/interim/constructor_inference.csv", index = False)

# COMMAND ----------

# MAGIC %md #### Read from db

# COMMAND ----------

predDF_final_done = spark.read.format("jdbc").option("url", "jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "test_airbnb_preds") \
    .option("user", "admin").option("password", "Xs19980312!").load()

# COMMAND ----------

