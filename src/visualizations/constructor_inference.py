# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Visualization

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

df_prob = spark.read.csv("s3://group3-gr5069/interim/constructor_inference.csv", header = True, inferSchema = True)

# COMMAND ----------

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

df_prob_pd = df_prob.toPandas()

# COMMAND ----------

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')

# Data for three-dimensional scattered points
zdata = df_prob_pd['prob_1']
xdata = df_prob_pd['lag1_avg']
ydata = df_prob_pd['race_count']

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap = 'coolwarm', s = 5)
ax.set_xlabel('Average Driver Point, Last Season ')
ax.set_ylabel('Race Completed, Current Season')
ax.set_zlabel('Probability of Championship')

# COMMAND ----------

fig = plt.figure(figsize=(10,8))
ax = plt.axes(projection='3d')

ax.plot_trisurf(xdata, ydata, zdata,
                cmap='coolwarm', edgecolor='none')
ax.set_xlabel('Average Driver Point, Last Season ')
ax.set_ylabel('Race Completed, Current Season')
ax.set_zlabel('Probability of Championship')

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Selected Features Distribution by Championship

# COMMAND ----------

plt.style.use('seaborn')
sns.boxplot(x="champion", y="race_count", palette="Set3",data=df_prob_pd, dodge=False)
sns.swarmplot(x="champion", y="race_count", data=df_prob_pd, color="tomato", s = 3)

# COMMAND ----------

sns.boxplot(x="champion", y="lag1_avg", palette="Set3",data=df_prob_pd, dodge=False)
sns.swarmplot(x="champion", y="lag1_avg", data=df_prob_pd, color="tomato", s = 3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modelfit
# MAGIC 1. championship = 1
# MAGIC 2. prediction = 1

# COMMAND ----------

df_cham1 = df_prob.filter(col('champion') == 1).toPandas()
df_pred1 = df_prob.filter(col('prediction') == 1).toPandas()

# COMMAND ----------

df_cham1

# COMMAND ----------

# MAGIC %md #### Read from db

# COMMAND ----------

predDF_final_done = spark.read.format("jdbc").option("url", "jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "test_airbnb_preds") \
    .option("user", "admin").option("password", "Xs19980312!").load()

# COMMAND ----------

