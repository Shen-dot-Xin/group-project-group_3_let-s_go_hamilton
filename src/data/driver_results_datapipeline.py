# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Driver Results- DataPipeline
# MAGIC 
# MAGIC Joining individual raw tables, handling datatypes, cleaning up or interpolating missing values etc. for Q1 and Q2

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import matplotlib as mp
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

from numpy import savetxt

from io import StringIO # python3; python2: BytesIO 
import boto3
s3 = boto3.client('s3')


# COMMAND ----------

# DBTITLE 1,Reading Raw Data
#Reading required datasets from the S3 bucket. 

#Reading Results data and selecting and renaming required columns
results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True, inferSchema=True)

df_results= results.select(col("resultId"), col("raceId"), col("driverId"), col("constructorId"), col("grid").alias("gridPosition"), col("position").alias("finishPosition"), col("positionOrder"),
                           col("points").alias("driverRacePoints"), col("laps").alias("raceLaps"), col("milliseconds").alias("raceDuration"), col("fastestLap"), col("rank").alias("fastestLapRank"), 
                           col("fastestLapSpeed"))

#Reading Drivers data
drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True, inferSchema=True)

#Reading Races data
races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True, inferSchema=True)

#Reading Pitstop data
pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header=True, inferSchema=True)
df_pitstops = pitstops.groupby('raceId','driverId').agg(round(sum('milliseconds'),0).alias('totPitstopDur'), round(avg('milliseconds'),0).alias('avgPitstopDur'), 
                                                        countDistinct(col('stop')).alias('countPitstops'), min(col('lap')).alias('firstPitstopLap') )
#Reading Constructors data
constructors = spark.read.csv('s3://columbia-gr5069-main/raw/constructors.csv', header=True, inferSchema=True)

#Reading Driver Standings data and selecting only required columns
driver_standings = spark.read.csv('s3://columbia-gr5069-main/raw/driver_standings.csv', header=True, inferSchema=True)

df_driver_standings = driver_standings.select(col("raceId"),col("driverId"), col("position").alias("driverStPosition"), col("points").alias("driverSeasonPoints"), col("wins").alias("driverSeasonWins"))

#Reading Constructor Standings data and selecting only required columns
const_standings = spark.read.csv('s3://columbia-gr5069-main/raw/constructor_standings.csv', header=True, inferSchema=True)

df_const_standings = const_standings.select(col("constructorId"),col("raceId"), col("position").alias("constStPosition"), col("points").alias("constSeasonPoints"), col("wins").alias("constSeasonWins"))

# COMMAND ----------

# DBTITLE 1,Driver-Race Data- Data to be used for Modeling
#Joining Results and Driver table 
driver_race_results= df_results.join(df_driver_standings, on=['raceId', 'driverId'])
driver_race_results= driver_race_results.join(races.select(col("raceId"), col("year").alias("raceYear"), col("date").alias("raceDate")), on=['raceId'])
driver_race_results= driver_race_results.join(df_const_standings, on=['raceId', 'constructorId'])
driver_race_results= driver_race_results.join(df_pitstops, on=['raceId', 'driverId'])

# Replacing the /N newline values with NA in this table
driver_race_results =driver_race_results.select("*").toPandas()
driver_race_results = driver_race_results.replace({r'\\N': np.nan}, regex=True)
driver_race_results = spark.createDataFrame(driver_race_results)

# Sorting the table to have recent races at the top
driver_race_results= driver_race_results.sort(driver_race_results.raceId.desc(), driver_race_results.positionOrder)

# COMMAND ----------

# Writing this interim data to S3 bucket
driver_race_results =driver_race_results.select("*").toPandas()
bucket = "group3-gr5069" # already created on S3
csv_buffer = StringIO()
driver_race_results.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'interim/driver_race_results_joined_interim.csv').put(Body=csv_buffer.getvalue())


# COMMAND ----------

#Reading the Interim data from Group's S3 bucket
bucket = "group3-gr5069"
f1_data = "interim/driver_race_results_joined_interim.csv"
obj = s3.get_object(Bucket= bucket, Key= f1_data)
driver_race_results_mod = pd.read_csv(obj['Body'], index_col=0)

# COMMAND ----------

# Create a view or table out 
driver_race_results_sp = spark.createDataFrame(driver_race_results_mod)
driver_race_results_sp.createOrReplaceTempView("driver_race_results_view")

# COMMAND ----------

# DBTITLE 1,Interpolating raceDuration, finishPosition from driverRaceDuration
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)
driver_race_results_sp = sqlContext.sql("SELECT mas.*, agg.max_raceLaps, agg.avg_raceDur  \
                      FROM `driver_race_results_view` mas \
                      LEFT JOIN \
                      (SELECT raceId, MAX(raceLaps) AS max_raceLaps, AVG(raceDuration) as avg_raceDur \
                         FROM `driver_race_results_view` GROUP BY raceId) as agg ON mas.raceId = agg.raceId \
                      WHERE driverRacePoints = 0 \
                      ORDER BY mas.raceId DESC, positionOrder;")
driver_race_results_mod =driver_race_results_sp.select("*").toPandas()

# COMMAND ----------

driver_race_results_mod.loc[driver_race_results_mod['fastestLap'].isnull(),'fastestLap'] = 0
driver_race_results_mod.loc[driver_race_results_mod['finishPosition'].isnull(),'finishPosition'] = 999
driver_race_results_mod.loc[driver_race_results_mod['fastestLapSpeed'].isnull(),'fastestLapSpeed'] = 0
driver_race_results_mod.loc[driver_race_results_mod['raceDuration'].isnull(),'raceDuration'] = driver_race_results_mod['avg_raceDur']
driver_race_results_mod = driver_race_results_mod.drop(columns=['max_raceLaps', 'avg_raceDur'])

# COMMAND ----------

# Writing this data to S3 bucket
bucket = "group3-gr5069" # already created on S3
csv_buffer = StringIO()
driver_race_results_mod.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'processed/driver_race_results_mod.csv').put(Body=csv_buffer.getvalue())

# COMMAND ----------

# DBTITLE 1,Driver-Race-Constructor Data- Data to be used for exploration 
#Reading Drivers data
driver_race_results_mod_sp = spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod.csv', header=True, inferSchema=True)
driver_race_results_mod_sp = driver_race_results_mod_sp.drop('_c0')

# COMMAND ----------

#Joining Driver,Race and Contructor Information just for exploration purposes 
driver_race_results_info= driver_race_results_mod_sp.join(drivers.select(col("driverId"), concat(drivers.forename,drivers.surname).alias("driverName"), col("nationality").alias("driverNat")), on=['driverId']).join(constructors.select(col("constructorId"),col("name").alias("constructorName"),col("nationality").alias("constructorNat")), on=['constructorId']).join(races.select(col("raceId"), col("name").alias("raceName"),col("round").alias("raceRound")), on=['raceId'])

# Rearranging and dropping few columns to make it readable in one view
driver_race_results_info = driver_race_results_info.select("raceId","raceName", "raceYear", "raceDate", "raceRound","driverName","driverNat", "constructorName", "gridPosition", "finishPosition", "positionOrder","driverRacePoints","raceLaps","raceDuration","fastestLap","fastestLapRank", "fastestLapSpeed","driverStPosition", "driverSeasonPoints", "driverSeasonWins","constStPosition", "constSeasonPoints", "constSeasonWins")

# Sorting the table to have recent races at the top
driver_race_results_info= driver_race_results_info.sort(driver_race_results_info.raceId.desc(), driver_race_results_info.positionOrder)

# COMMAND ----------

# Writing this data to S3 bucket
# Writing this data to S3 bucket
driver_race_results_info =driver_race_results_info.select("*").toPandas()
bucket = "group3-gr5069" # already created on S3
csv_buffer = StringIO()
driver_race_results_info.to_csv(csv_buffer)
s3_resource = boto3.resource('s3')
s3_resource.Object(bucket, 'processed/driver_race_results_info.csv').put(Body=csv_buffer.getvalue())
