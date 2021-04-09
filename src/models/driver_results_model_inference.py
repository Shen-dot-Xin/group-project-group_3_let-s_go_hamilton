# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Driver Results Model for Inference(Q1)
# MAGIC 
# MAGIC This file is for develping a model to predict the second posiiton in a rae for Inference(Q1)

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date,udf, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import tempfile
import matplotlib.pyplot as plt
from numpy import savetxt

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driverRaceDF= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driverRaceDF = driverRaceDF.drop('_c0')

#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Dropping the pitstops data since this data is only available post 2011 races
#driverRaceDF = driverRaceDF.drop("totPitstopDur","avgPitstopDur","countPitstops","firstPitstopLap","raceDate")

# Dropping similar columns to target variables
#driverRaceDF = driverRaceDF.drop("positionOrder","finishPosition")


driverRaceDF = driverRaceDF.select('raceId', 'driverId', 'constructorId', 'resultId', 'gridPosition','driverRacePoints', 'driverStPosition',
                            'driverSeasonPoints', 'driverSeasonWins', 'raceYear', 'constStPosition','constSeasonPoints', 'constSeasonWins', 
                            'drivSecPos')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression 

# COMMAND ----------

# MAGIC %md 
# MAGIC #### VectorAssembler

# COMMAND ----------

## Transforming a selection of features into a vector using VectorAssembler.
vecAssembler = VectorAssembler(inputCols = ['raceId', 'driverId', 'constructorId', 'resultId', 'gridPosition','driverRacePoints', 'driverStPosition',
                                            'driverSeasonPoints', 'driverSeasonWins', 'constStPosition','constSeasonPoints', 'constSeasonWins'], outputCol = "vectorized_features")

#
driverRaceDF = vecAssembler.transform(driverRaceDF)


# COMMAND ----------

# MAGIC %md 
# MAGIC #### StandardScaler

# COMMAND ----------

## 
scaler = StandardScaler()\
         .setInputCol('vectorized_features')\
         .setOutputCol('features')

scaler_model = scaler.fit(driverRaceDF)

#
driverRaceDF = scaler_model.transform(driverRaceDF)


# COMMAND ----------

# MAGIC %md 
# MAGIC #### LabelIndexer

# COMMAND ----------

## 
label_indexer = StringIndexer()\
         .setInputCol ("drivSecPos")\
         .setOutputCol ("label")

label_indexer_model=label_indexer.fit(driverRaceDF)
driverRaceDF=label_indexer_model.transform(driverRaceDF)


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Train-Test Split

# COMMAND ----------

#Splitting the data frame into Train and Test data based on the requirements of the Project and converting them to Pandas Dataframes
driverRaceTrainDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)
driverRaceTestDF = driverRaceDF.filter(driverRaceDF.raceYear > 2010)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Training

# COMMAND ----------

logr = LogisticRegression(featuresCol ='vectorized_features', labelCol = "drivSecPos", maxIter=5)
logrModel = logr.fit(driverRaceTrainDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Prediction

# COMMAND ----------

predictions = logrModel.transform(driverRaceTestDF)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Accuracy

# COMMAND ----------

accuracy = predictions.filter(predictions.drivSecPos == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)

# COMMAND ----------

trainingSummary = logrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Model Evaluation 

# COMMAND ----------

evaluator= BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(logrModel.regParam, [0.01, 0.5, 2.0])# regularization parameter
             .addGrid(logrModel.elasticNetParam, [0.0, 0.5, 1.0])# Elastic Net Parameter (Ridge = 0)
             .addGrid(logrModel.maxIter, [1, 5, 10])#Number of iterations
             .build())

cv = CrossValidator(estimator=logrModel, estimatorParamMaps=paramGrid, 
                    evaluator=evaluator, numFolds=5)

cvModel = cv.fit(driverRaceTrainDF)

# COMMAND ----------

driverRaceDF.toPandas().head()

# COMMAND ----------

driverRaceTrainDF.display()

# COMMAND ----------

