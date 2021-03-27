# Databricks notebook source
#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct
from pyspark.sql.types import IntegerType

# COMMAND ----------

#Reading required datasets from the S3 bucket. 

#Reading Pitstops data and selecting only required columns
pitstops = spark.read.csv('s3://columbia-gr5069-main/raw/pit_stops.csv', header=True   )
df_pitstops = pitstops.select('driverId','raceId', 'duration')

#Reading Drivers data and selecting only required columns
drivers = spark.read.csv('s3://columbia-gr5069-main/raw/drivers.csv', header=True)

#Reading Driver Standings data and selecting only required columns
driver_standings = spark.read.csv('s3://columbia-gr5069-main/raw/driver_standings.csv', header=True)

#Reading Races data and selecting only required columns
races = spark.read.csv('s3://columbia-gr5069-main/raw/races.csv', header=True)

#Reading Results data and selecting and renaming required columns
results = spark.read.csv('s3://columbia-gr5069-main/raw/results.csv', header=True, inferSchema=True)

df_results= results.select(col("resultId"), col("raceId"), col("driverId"), col("constructorId"), col("grid").alias("gridPosition"), col("position").alias("finishPosition"), 
                           col("racePoints"), col("raceLaps"), col("milliseconds").alias("raceDuration"), col("fastestLap"), col("rank").alias("fastestLapRank"), col("fastestLapTime"), col("fastestLapSpeed") )

# COMMAND ----------

display(results)