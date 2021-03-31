# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This notebook cleans the datasets related to the f1 constructor success prediction.
# MAGIC 
# MAGIC The datasets are: 
# MAGIC  - constructors.csv
# MAGIC  - constructor_standings.csv
# MAGIC  - constructor_results.csv
# MAGIC  
# MAGIC  
# MAGIC  Problem: by race or by season???
# MAGIC  
# MAGIC  
# MAGIC  Processing: 
# MAGIC  - Engine Problems: Shows the reliability of the constructors' cars, especially focusing on the engine.
# MAGIC  - Drivers performance: the stronger the overall perfomance of drivers in the last season, the more the points the constructors win.
# MAGIC    - pre-race average driver standing: the smaller, the better. problem: what if a driver does not enter the last race?
# MAGIC    - 
# MAGIC  

# COMMAND ----------

dbutils.library.installPyPI("seaborn", "0.11.0")
dbutils.library.installPyPI("plotly", "4.14.3")

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px

# COMMAND ----------

s3 = boto3.client('s3')
bucket = "columbia-gr5069-main"

# COMMAND ----------

# constructors
c = "raw/constructors.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_constructors = pd.read_csv(obj['Body'])
df_constructors.head() # raceId, driverId, position(qualifying)

# COMMAND ----------

# constructor standings
c = "raw/constructor_standings.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cs = pd.read_csv(obj['Body'])
df_cs.head()

# COMMAND ----------

df = df_cs.merge(df_constructors[['constructorId', 'constructorRef']], how = 'left', on = 'constructorId')
df.head()

# COMMAND ----------

# MAGIC %md #### Results by races

# COMMAND ----------

# how many races did the constructors win?
px.histogram(df[df['wins'] ==1], x = 'constructorRef')

# COMMAND ----------

# results
c = "raw/races.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_race = pd.read_csv(obj['Body'])
df_race

# COMMAND ----------

# number of race each circuit has held
px.histogram(df_race, x = 'name')

# COMMAND ----------

# MAGIC %md #### Championship

# COMMAND ----------

df = df.merge(df_race[['raceId','year','round',	'circuitId']], how = 'left', on = 'raceId')

# COMMAND ----------

idx = df.groupby(['year'])['round'].transform(max) == df['round']
df_champion = df[idx]

# COMMAND ----------

df_champion = df_champion[df_champion['position'] == 1]

# COMMAND ----------

# how many championships did constructors win?
px.histogram(df_champion, x = 'constructorRef')

# COMMAND ----------

df_champion

# COMMAND ----------

# how many rounds are there each season?
px.scatter(df_champion[df_champion['position'] == 1], x = 'year', y = 'round')

# COMMAND ----------

# championship every season/year
df_champion[df_champion['position'] == 1].sort_values('year').tail(20)

# COMMAND ----------

# constructor result
c = "raw/constructor_results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cr = pd.read_csv(obj['Body'])
df_cr.head() 

# COMMAND ----------

# status
c = "raw/status.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_s = pd.read_csv(obj['Body'])

# COMMAND ----------

# result
c = "raw/results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_r = pd.read_csv(obj['Body'])
df_r

# COMMAND ----------

# why drivers did not finish their race?
df_p = df_r[['raceId', 'statusId']].merge(df_s, how = 'left', on = 'statusId')
df_p = df_p[~df_p['status'].str.contains('Lap')]
px.histogram(df_p[df_p['status'] != 'Finished'], x = 'status')

# COMMAND ----------

#add year，delete year = 2021
df_y = df_race[['raceId','year']].merge(df_r, how = 'left', on = 'raceId')
df_y = df_y[~df_y['year'].isin([2021])]
df_y

# COMMAND ----------

# status classification
c = "raw/status.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_status = pd.read_csv(obj['Body'])
display(df_status)

# COMMAND ----------

# MAGIC %md #### Engine problems

# COMMAND ----------

# The frequency of car breakdown (statusId = 5) for each constructor by seasons
df_b = df_y[df_y['statusId'].isin([5])]
df_b

# COMMAND ----------

df_b = df_b.groupby(['year','constructorId']).statusId.value_counts()
df_b = df_b.reset_index(name='engineproblem')
display(df_b)

# COMMAND ----------

# engine problems
df_car = df_b[['year','constructorId','engineproblem']].merge(df_y, how='outer', on=['constructorId', 'year'])
df_car['engineproblem'] = df_car['engineproblem'].replace(np.nan, 0)
df_car

# COMMAND ----------

# MAGIC %md #### Driver selection

# COMMAND ----------

# average points in this season
df_points = df_car.groupby(['year','constructorId']).points.mean()
df_points = df_points.reset_index(name='avgpoints_c')
df_points.loc[:, 'participation'] = 1
df_points

# COMMAND ----------

#driver selection
df_per = df_points[['year','constructorId','avgpoints_c','participation']].merge(df_car, how='outer', on=['constructorId', 'year'])
df_per['participation'] = df_per['participation'].replace(np.nan, 0)
df_per['avgpoints_c'] = df_per['avgpoints_c'].replace(np.nan, 0)
df_per

# COMMAND ----------

