# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Inference

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Window
from pyspark.sql.functions import lag, col, asc

# COMMAND ----------

df = spark.read.csv('s3://group3-gr5069/interim/constructor_features.csv', header = True, inferSchema = True)

# COMMAND ----------

window = Window.partitionBy('constructorId').orderBy(asc('year'))

df = df.withColumn("lag1_avg", lag("avgpoints_c", 1, 0).over(window))
df = df.withColumn("lag2_avg", lag("avgpoints_c", 2, 0).over(window))

# COMMAND ----------

df = df.withColumn("lag1_fs", lag("avg_fastestspeed", 1, 0).over(window))
df = df.withColumn("lag2_fs", lag("avg_fastestspeed", 2, 0).over(window))

# COMMAND ----------

df = df.withColumn("lag1_fl", lag("avg_fastestlap", 1, 0).over(window))
df = df.withColumn("lag2_fl", lag("avg_fastestlap", 2, 0).over(window))

# COMMAND ----------

df = df.withColumn("lag1_nd", lag("unique_drivers", 1, 0).over(window))
df = df.withColumn("lag2_nd", lag("unique_drivers", 2, 0).over(window))

# COMMAND ----------

df = df.withColumn("lag1_standing", lag("position", 1, 0).over(window))
df = df.withColumn("lag2_standing", lag("position", 2, 0).over(window))

# COMMAND ----------

df.columns

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = [ 'avg_fastestspeed','avg_fastestlap','race_count','engineproblem','avgpoints_c',], outputCol = "features")

vecDF = vecAssembler.transform(df)

lr = LinearRegression(featuresCol = "features", labelCol = "champion")
lrModel = lr.fit(vecDF)

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = [ 'engineproblem','avgpoints_c', "lag1_avg", "lag2_avg" ], outputCol = "features")

vecDF = vecAssembler.transform(df)

lr = LinearRegression(featuresCol = "features", labelCol = "champion")
lrModel = lr.fit(vecDF)

print(lrModel.summary.rootMeanSquaredError)
print(lrModel.summary.r2)

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

predDF = lrModel.transform(vecDF)
display(predDF)

# COMMAND ----------

print(lrModel.summary.rootMeanSquaredError)
print(lrModel.summary.r2)
lrModel.coefficients

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

