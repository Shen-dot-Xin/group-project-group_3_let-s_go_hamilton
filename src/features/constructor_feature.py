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
# MAGIC  - The number of completed races
# MAGIC  - Average Fastest Lap
# MAGIC  - Average Fastest Speed
# MAGIC  - Number of unique drivers
# MAGIC  - Constructor ranking

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np

#!pip install s3fs

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

# Read constructor championships
c = "processed/constructor_championships.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cc = pd.read_csv(obj['Body'])
df_cc.head() 

# COMMAND ----------

# filter races between 1950 and 2017
df_cc = df_cc[(df_cc['year']>=1950) & (df_cc['year']<=2017)]
df_cc.head()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature I: Circuits

# COMMAND ----------

df_1 = df_cc.merge(df[['raceId', 'circuitId']].drop_duplicates(), how = 'left', on = 'raceId')
df_1.head()

# COMMAND ----------

df_11 = df_1.pivot_table(values=['wins'], index=['year', 'constructorId'], columns='circuitId')
df_11.columns =['gp_' + str(s2) for (s1,s2) in df_11.columns.tolist()]
df_11= df_11.fillna(0).reset_index()
df_11.head()

# COMMAND ----------

'''get a series of dummy variables
df_x = pd.concat([df_1['year'], pd.get_dummies(df_1['circuitId'], prefix="gp")],  axis = 1 )
df_x = df_x.drop_duplicates(subset=['year'], keep='first', inplace=False).reset_index(drop = True)
df_x.head()'''

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

# MAGIC %md #### Feature V: Completed Races

# COMMAND ----------

df_4 = df.groupby(['year', 'constructorId']).position.count()
df_4 = df_4.reset_index(name='race_count')
df_4 = df_4.fillna(0)
df_4.head()

# COMMAND ----------

df_x = df_4.merge(df_x, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

# MAGIC %md #### Feature VI: Fastest Lap

# COMMAND ----------

df_5 = df.groupby(['year', 'constructorId']).fastestLap.mean()
df_5 = df_5.reset_index(name='avg_fastestlap')
df_5 = df_5.fillna(0)
df_5.head()

# COMMAND ----------

df_x = df_5.merge(df_x, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

# MAGIC %md #### Feature VII: Fastest Lap Speed

# COMMAND ----------

df_6 = df.groupby(['year', 'constructorId']).fastestLapSpeed.mean()
df_6 = df_6.reset_index(name='avg_fastestspeed')
df_6 = df_6.fillna(0)
df_6.head()

# COMMAND ----------

df_x = df_6.merge(df_x, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

# MAGIC %md #### Feature VIII: Number of unique drivers

# COMMAND ----------

df_8 = df.groupby(['year', 'constructorId']).driverId.nunique().reset_index(name='unique_drivers')
df_8.head()

# COMMAND ----------

df_x = df_x.merge(df_8, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

# MAGIC %md #### Feature IX: Constructor Ranking

# COMMAND ----------

# filter the last round of each season
idx = df_cc.groupby(['year'])['round'].transform(max) == df_cc['round']
df_9 = df_cc[idx]
df_9.head()

# COMMAND ----------

df_9 = df_9[['year', 'constructorId', 'position']]
df_x = df_x.merge(df_9, how = 'left', on = ['year', 'constructorId'])
df_x.head()

# COMMAND ----------

df_x['lag1_avg']=df_x.sort_values('year').groupby('constructorId')['avgpoints_c'].shift()
df_x['lag2_avg']=df_x.sort_values('year').groupby('constructorId')['lag1_avg'].shift()
df_x['lag1_ptc']=df_x.sort_values('year').groupby('constructorId')['participation'].shift()
df_x['lag2_ptc']=df_x.sort_values('year').groupby('constructorId')['lag1_ptc'].shift()
df_x['lag1_pst']=df_x.sort_values('year').groupby('constructorId')['position'].shift()
df_x['lag2_pst']=df_x.sort_values('year').groupby('constructorId')['lag1_pst'].shift()

# COMMAND ----------

df_x['lag1_avg']=df_x['lag1_avg'].fillna(0)
df_x['lag2_avg']=df_x['lag2_avg'].fillna(0)
df_x['lag1_ptc']=df_x['lag1_ptc'].fillna(0)
df_x['lag2_ptc']=df_x['lag2_ptc'].fillna(0)
df_x['lag1_pst']=df_x['lag1_pst'].fillna(0)
df_x['lag2_pst']=df_x['lag2_pst'].fillna(0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Y: Championship

# COMMAND ----------

# filter the winner in each last round, i.e. each season's constructor champion
df_champion = df_9[df_9['position'] == 1]
df_champion.head()

# COMMAND ----------

df_champion = df_champion[['year','constructorId']]
df_champion['champion'] = 1

# COMMAND ----------

df_xy = df_x.merge(df_champion, how = 'left', on = ['year', 'constructorId'])
df_xy.head()

# COMMAND ----------

df_xy['champion'] = df_xy['champion'].fillna(0)
df_xy.head()

# COMMAND ----------

df_xy.to_csv("s3://group3-gr5069/interim/constructor_features.csv", index = False)

# COMMAND ----------

