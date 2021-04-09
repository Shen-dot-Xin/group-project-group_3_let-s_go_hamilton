# Databricks notebook source
# MAGIC %md ### Data Pipeline
# MAGIC 
# MAGIC This notebook cleans raw datasets related to the f1 constructor championship prediction and output relevant data for next step.
# MAGIC 
# MAGIC The datasets are: 
# MAGIC  - results.csv
# MAGIC  - races.csv
# MAGIC  - constructors.csv

# COMMAND ----------

import boto3
import pandas as pd
import numpy as np

# COMMAND ----------

!pip install s3fs

# COMMAND ----------

s3 = boto3.client('s3')
bucket = "columbia-gr5069-main"

# COMMAND ----------

# main dataframe: result
c = "raw/results.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_result = pd.read_csv(obj['Body'])
df_result.head()

# COMMAND ----------

# reading race.csv
c = "raw/races.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_race = pd.read_csv(obj['Body'])
df_race.head()

# COMMAND ----------

# Append year, round, circuits Id to main frame
df_join = df_result.merge(df_race[['raceId', 'year', 'round', 'circuitId']], how = 'left', on = 'raceId')
df_join.head()

# COMMAND ----------

# reading constructors.csv
c = "raw/constructors.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_constructors = pd.read_csv(obj['Body'])
df_constructors.head() 

# COMMAND ----------

# Append constructor reference name to main frame
df_join = df_join.merge(df_constructors[['constructorId', 'constructorRef']], how = 'left', on = 'constructorId')
df_join.head()

# COMMAND ----------

df_join = df_join.replace({r'\\N': np.nan}, regex=True)

# COMMAND ----------

df_join.to_csv('s3://group3-gr5069/processed/constructor_info.csv', index = False)

# COMMAND ----------

# constructor standings
c = "raw/constructor_standings.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
df_cs = pd.read_csv(obj['Body'])
df_cs.head()

# COMMAND ----------

df_join2 = df_cs.merge(df_race[['raceId', 'year']], how = 'left', on = 'raceId')
df_join2.head()

# COMMAND ----------

df_join2 = df_join2.merge(df_constructors[['constructorId', 'constructorRef']], how = 'left', on = 'constructorId')
df_join2.head()

# COMMAND ----------

df_join2.to_csv('s3://group3-gr5069/processed/constructor_championships.csv', index = False)

# COMMAND ----------

