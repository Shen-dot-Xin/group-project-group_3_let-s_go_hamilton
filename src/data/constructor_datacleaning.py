# Databricks notebook source


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This notebook cleans the datasets related to the f1 constructor championship prediction.
# MAGIC 
# MAGIC The datasets are: 
# MAGIC  - constructors.csv
# MAGIC  - constructor_standings.csv
# MAGIC  - constructor_results.csv
# MAGIC  - races.csv
# MAGIC  - results.csv
# MAGIC  - status.csv
# MAGIC  
# MAGIC  
# MAGIC  The following features are engineered: 
# MAGIC  - Engine Problems: Shows the reliability of the constructors' cars, especially focusing on the engine.
# MAGIC  - Drivers performance: the stronger the overall perfomance of drivers in the last season, the more the points the constructors win.
# MAGIC  - Constructor's consistency
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

# MAGIC %md 
# MAGIC ## Exploratory Data Analysis and Visualization

# COMMAND ----------

# reading constructors.csv
c = "raw/constructors.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_constructors = pd.read_csv(obj['Body'])
df_constructors.head() # raceId, driverId, position(qualifying)

# COMMAND ----------

# reading constructor_standings.csv
c = "raw/constructor_standings.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cs = pd.read_csv(obj['Body'])
df_cs.head()

# COMMAND ----------

# Append constructor reference name to the constructor standings dataframe, for the purpose of visualization
df_cs = df_cs.merge(df_constructors[['constructorId', 'constructorRef']], how = 'left', on = 'constructorId')
df_cs.head()

# COMMAND ----------

# how many races did the constructors win in the history of f1, up to March 2021?
px.histogram(df_cs[df_cs['wins'] ==1], x = 'constructorRef', title = "How many races did the constructors win in the history of f1, up to March 2021?")

# COMMAND ----------

# MAGIC %md #### Results by races

# COMMAND ----------

# reading race.csv
c = "raw/races.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_race = pd.read_csv(obj['Body'])
df_race.head()

# COMMAND ----------

# How many race has each circuit held in the history of f1?
px.histogram(df_race, x = 'name', title = "How many race has each circuit held in the history of f1?")

# COMMAND ----------

# MAGIC %md #### Constructor Championship

# COMMAND ----------

# append season, round and circuitId to constructor standing dataset
df_cs = df_cs.merge(df_race[['raceId','year','round','circuitId']], how = 'left', on = 'raceId')

# COMMAND ----------

# filter the last round of each season
idx = df_cs.groupby(['year'])['round'].transform(max) == df_cs['round']
df_champion = df_cs[idx]

# COMMAND ----------

# filter the winner in each last round, i.e. each season's constructor champion
df_champion = df_champion[df_champion['position'] == 1]
df_champion.head()

# COMMAND ----------

# how many rounds are there each season?
px.scatter(df_champion, x = 'year', y = 'round', title="How many rounds are there in each season?")

# COMMAND ----------

# Last five championship, including the winner of the first race of 2021.  
df_champion.sort_values('year').tail()

# COMMAND ----------

# how many championships did each constructors win in the history of f1?
px.histogram(df_champion, x = 'constructorRef', title = "How many championships did the constructors win in the history of f1?")

# COMMAND ----------

# result
c = "raw/results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_result = pd.read_csv(obj['Body'])
df_result.head()

# COMMAND ----------

# reading status.csv
c = "raw/status.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_status = pd.read_csv(obj['Body'])
display(df_status)

# COMMAND ----------

# why have drivers failed to finish their races?
df_plot = df_result[['raceId', 'statusId']].merge(df_status, how = 'left', on = 'statusId') # append status explanation to results
df_plot = df_plot[~df_plot['status'].str.contains('Lap')] # remove finishing with laps behind the lead
px.histogram(df_plot[df_plot['status'] != 'Finished'], x = 'status', title = "For what reason were the drivers out before the end of their races?") # remove the finished status

# COMMAND ----------

# MAGIC %md #### Feature I: Engine problems

# COMMAND ----------

# append year to race result and remove race results of 2021 (imcomplete season)
df_y = df_race[['raceId','year']].merge(df_result, how = 'left', on = 'raceId')
df_y = df_y[~df_y['year'].isin([2021])]
df_y.head()

# COMMAND ----------

# Filter race results ending with engine problem (statusId = 5) 
df_b = df_y[df_y['statusId'].isin([5])]
df_b.head()

# COMMAND ----------

# Count the number of engine problems each constructor had in each year
df_b = df_b.groupby(['year','constructorId']).statusId.value_counts()
df_b = df_b.reset_index(name='engineproblem')
display(df_b)

# COMMAND ----------

# Question: Why rejoin the count with the year dataframe
df_car = df_b[['year','constructorId','engineproblem']].merge(df_y, how='outer', on=['constructorId', 'year'])
df_car['engineproblem'] = df_car['engineproblem'].replace(np.nan, 0)
df_car.head()

# COMMAND ----------

# MAGIC %md #### Feature II: Driver Selection

# COMMAND ----------

# average points in this season
df_points = df_car.groupby(['year','constructorId']).points.mean() # Average Point 
df_points = df_points.reset_index(name='avgpoints_c')
df_points.loc[:, 'participation'] = 1 # Dummy variable for participation: 0 for absence, 1 for participation
df_points.head()

# COMMAND ----------

#driver selection
df_per = df_points[['year','constructorId','avgpoints_c','participation']].merge(df_car, how='outer', on=['constructorId', 'year'])
df_per['participation'] = df_per['participation'].replace(np.nan, 0)
df_per['avgpoints_c'] = df_per['avgpoints_c'].replace(np.nan, 0)
df_per

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature III: Metrics of constructor's consistency
# MAGIC 
# MAGIC - Measures whether a constructor participates consecutively, whether it is a new comer, or it rebrands itself frequently. 
# MAGIC - Weighted average of performance over the last five seasons. 
# MAGIC - Punish both absence of participation and low ranking. 

# COMMAND ----------

# constructor result
c = "raw/constructor_results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cr = pd.read_csv(obj['Body'])
df_cr.head() 