# Databricks notebook source
# MAGIC %md ### Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This notebook reads the processed data and engineers the features. 
# MAGIC 
# MAGIC  The following features are engineered: 
# MAGIC  - Circuits
# MAGIC  - Average Driver Point per race
# MAGIC  - First Participation
# MAGIC  - Engine Problems
# MAGIC 
# MAGIC  

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np
from pyspark.sql import Window

# COMMAND ----------

s3 = boto3.client('s3')
bucket = "group3-gr5069"

# COMMAND ----------

# Read data
c = "processed/constructor_info.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df = pd.read_csv(obj['Body'])
df.head() 

# COMMAND ----------

# filter races between 1950 and 2017
df = df[(df['year']>=1950) & (df['year']<=2017)]
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature I: Circuits

# COMMAND ----------

# get the name of each race
bucket = "columbia-gr5069-main"
c = "raw/races.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_1 = pd.read_csv(obj['Body'])
df_1 = df_1[['raceId','name','year']]
display(df_1)

# COMMAND ----------

#get a series of dummy variables
df_x = pd.concat([df_1['year'], pd.get_dummies(df_1['name'], prefix="")],  axis = 1 )
df_x = df_x.drop_duplicates(subset=['year'], keep='first', inplace=False)
df_x

# COMMAND ----------

# MAGIC %md #### Feature II & III: Average Driver Point per Race & Participation

# COMMAND ----------

# average points in this season
df_2 = df.groupby(['year','constructorId']).points.mean() # Average Point 
df_2 = df_2.reset_index(name='avgpoints_c')
df_2.loc[:, 'participation'] = 1 # Dummy variable for participation: 0 for absence, 1 for participation
df_2.head()

# COMMAND ----------

df_x = df_2.merge(df_x, how = 'left', on = ['year'])
df_x.head()

# COMMAND ----------

df_x['participation'] = df_x['participation'].replace(np.nan, 0)
df_x['avgpoints_c'] = df_x['avgpoints_c'].replace(np.nan, 0)

# COMMAND ----------

# MAGIC %md #### Feature IV: Engine problems

# COMMAND ----------

# Filter race results ending with engine problem (statusId = 5) 
df_3 = df[['year', 'raceId', 'constructorId','driverId','statusId']]

conditions = [
    (df_3['statusId'] == 5)]
fill_list = [1]
df_3['engine'] = np.select(conditions, fill_list, default = 0)

df_3.head()

# COMMAND ----------

# Count the number of engine problems each constructor had in each year
df_3 = df_3.groupby(['year','constructorId']).engine.mean()
df_3 = df_3.reset_index(name='engineproblem')
df_3.head()

# COMMAND ----------

df_x = df_3.merge(df_x, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Y: Championship

# COMMAND ----------

# filter the last round of each season
idx = df.groupby(['year'])['round'].transform(max) == df['round']
df_champion = df[idx]

# COMMAND ----------

# filter the winner in each last round, i.e. each season's constructor champion
df_champion = df_champion[df_champion['position'] == 1]
df_champion.head()

# COMMAND ----------

df_champion = df_champion[['year', 'constructorId']]

# COMMAND ----------

df_champion['champion'] = 1
df_champion.sort_values('year').tail()

# COMMAND ----------

df_xy = df_x.merge(df_champion, how = 'left', on = ['year', 'constructorId'])
df_xy.head()

# COMMAND ----------

df_xy['champion'] = df_xy['champion'].fillna(0)
df_xy.head()

# COMMAND ----------

df_xy.to_csv("s3://group3-gr5069/interim/constructor_features.csv", index = False)