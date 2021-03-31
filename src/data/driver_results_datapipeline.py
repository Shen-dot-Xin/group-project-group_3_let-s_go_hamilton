# Databricks notebook source
#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp

# COMMAND ----------

# DBTITLE 1,Reading Raw Data
#Reading required datasets from the S3 bucket. 

#Reading Results data and selecting and renaming required columns
results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True, inferSchema=True)

df_results= results.select(col("resultId"), col("raceId"), col("driverId"), col("constructorId"), col("grid").alias("gridPosition"), col("position").alias("finishPosition"), col("positionOrder"),
                           col("points").alias("driverRacePoints"), col("laps").alias("raceLaps"), col("milliseconds").alias("raceDuration"), col("fastestLap"), col("rank").alias("fastestLapRank"), col("fastestLapTime"), 
                           col("fastestLapSpeed"))

#Reading Drivers data
drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True, inferSchema=True)

#Reading Races data
races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True, inferSchema=True)

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
driver_race_results= driver_race_results.join(df_const_standings, on=['raceId', 'constructorId'])

# Replacing the /N newline values with NA in this table
driver_race_results =driver_race_results.select("*").toPandas()
driver_race_results = driver_race_results.replace({r'\\N': np.nan}, regex=True)
driver_race_results = spark.createDataFrame(driver_race_results)

# Sorting the table to have recent races at the top
driver_race_results= driver_race_results.sort(driver_race_results.raceId.desc(), driver_race_results.positionOrder)

# COMMAND ----------

# Writing this data to S3 bucket
driver_race_results.write.csv('s3://pp-gr5069/processed/driver_race_results.csv',mode="overwrite")

# COMMAND ----------

# DBTITLE 1,Driver-Race-Constructor Data- Data to be used for exploration 
#Joining Driver,Race and Contructor Information just for exploration purposes 
driver_race_results_info= driver_race_results.join(drivers.select(col("driverId"), concat(drivers.forename,drivers.surname).alias("driverName"), col("nationality").alias("driverNat")), on=['driverId']).join(constructors.select(col("constructorId"),col("name").alias("constructorName"),col("nationality").alias("constructorNat")), on=['constructorId']).join(races.select(col("raceId"), col("name").alias("raceName"),col("date").alias("raceDate"),col("round").alias("raceRound")), on=['raceId'])

# Rearranging and dropping few columns to make it readable in one view
driver_race_results_info = driver_race_results_info.select("raceId","raceName", "raceDate", "raceRound","driverName","driverNat", "constructorName", "gridPosition", "finishPosition", "positionOrder","driverRacePoints","raceLaps","raceDuration","fastestLap","fastestLapRank","fastestLapTime", "fastestLapSpeed","driverStPosition", "driverSeasonPoints", "driverSeasonWins","constStPosition", "constSeasonPoints", "constSeasonWins")

# Sorting the table to have recent races at the top
driver_race_results_info= driver_race_results_info.sort(driver_race_results_info.raceId.desc(), driver_race_results.positionOrder)

# COMMAND ----------

# Writing this data to S3 bucket
driver_race_results_info.write.csv('s3://pp-gr5069/processed/driver_race_results_exp.csv',mode="overwrite")