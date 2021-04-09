# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Inference

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

df = spark.read.csv('s3://group3-gr5069/interim/constructor_features.csv', header = True, inferSchema = True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Test Split
# MAGIC 
# MAGIC Keep 80% for the training set and set aside 20% of our data for the test set. We will use the `randomSplit` method [Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).

# COMMAND ----------

(trainDF, testDF) = df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

print(trainDF.cache().count())

# COMMAND ----------

display(trainDF)

# COMMAND ----------

df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols = [ 'avg_fastestspeed','avg_fastestlap','race_count','engineproblem','avgpoints_c',], outputCol = "features")

vecDF = vecAssembler.transform(df)

lr = LinearRegression(featuresCol = "features", labelCol = "champion")
lrModel = lr.fit(vecDF)

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = [ 'engineproblem','avgpoints_c', "lag1_avg", "lag2_avg" ], outputCol = "features")

vecDF = vecAssembler.transform(df)

lr = LinearRegression(featuresCol = "features", labelCol = "champion")
lrModel = lr.fit(vecDF)

# COMMAND ----------

print(lrModel.summary.rootMeanSquaredError)
print(lrModel.summary.r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply model to test set

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import lag, col, asc

window = Window.partitionBy('constructorId').orderBy(asc('year'))

# COMMAND ----------

df_lag = df.select('year','constructorId', 'avgpoints_c')

# COMMAND ----------

df = df.withColumn("lag1_avg", lag("avgpoints_c", 1, 0).over(window))

df = df.withColumn("lag2_avg", lag("avgpoints_c", 2, 0).over(window))


# COMMAND ----------

predDF = lrModel.transform(vecDF)

display(predDF)

# COMMAND ----------

print(lrModel.summary.rootMeanSquaredError)
print(lrModel.summary.r2)

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

lrModel.featureImportances

# COMMAND ----------

predDF_final = predDF.select('host_is_superhost','cancellation_policy','instant_bookable','host_total_listings_count','neighbourhood_cleansed','zipcode','latitude','longitude','property_type','room_type','accommodates','bathrooms','bedrooms','beds','bed_type','minimum_nights','number_of_reviews','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price','bedrooms_na','bathrooms_na','beds_na','review_scores_rating_na','review_scores_accuracy_na','review_scores_cleanliness_na','review_scores_checkin_na','review_scores_communication_na','review_scores_location_na','review_scores_value_na','prediction')

# COMMAND ----------

predDF_final.write.format('jdbc').options(
      url='jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200',
      driver='com.mysql.jdbc.Driver',
      dbtable='test_airbnb_preds',
      user='admin',
      password='Xs19980312!').mode('overwrite').save()

# COMMAND ----------

# MAGIC %md #### Read from db

# COMMAND ----------

predDF_final_done = spark.read.format("jdbc").option("url", "jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "test_airbnb_preds") \
    .option("user", "admin").option("password", "Xs19980312!").load()

# COMMAND ----------

