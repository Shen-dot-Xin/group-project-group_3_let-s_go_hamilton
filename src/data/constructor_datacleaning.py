# Databricks notebook source
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
# MAGIC  - Drivers pairing strength: the stronger the pair of drivers, the more the points the constructors win.
# MAGIC    - pre-race average driver standing: the smaller, the better. problem: what if a driver does not enter the last race?
# MAGIC    - 
# MAGIC  - Historical performance: the better the constructor was doing in the last season, the better this season???
# MAGIC  - Car Retirement: Shows the reliability of the constructors' cars.

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px

# COMMAND ----------

dbutils.library.installPyPI("seaborn", "0.11.0")
dbutils.library.installPyPI("plotly", "4.14.3")

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

# how many races did the constructors win?
px.histogram(df[df['wins'] ==1], x = 'constructorRef')

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

# why drivers did not finish their race?
df_p = df_r[['raceId', 'statusId']].merge(df_s, how = 'left', on = 'statusId')
df_p = df_p[~df_p['status'].str.contains('Lap')]
px.histogram(df_p[df_p['status'] != 'Finished'], x = 'status')

# COMMAND ----------

# result
c = "raw/results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_r = pd.read_csv(obj['Body'])
df_r

# COMMAND ----------

# result
c = "raw/races.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_race = pd.read_csv(obj['Body'])
df_race

# COMMAND ----------

# number of race each circuit has held
px.histogram(df_race, x = 'name')

# COMMAND ----------

