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

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driverRaceFeat= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driverRaceFeat = driverRaceFeat.drop('_c0')

#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Dropping the pitstops data since this data is only available post 2011 races

driverRaceDF = driverRaceFeat.select('raceYear','driverRacePoints', 'gridPosition','driverSeasonPoints',
                                      'drivSecPosRM1','drivSecPosRM2','drivSecPosRM3', 'drivSecPos')


# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression using Pyspark ML

# COMMAND ----------

# MAGIC %md 
# MAGIC #### VectorAssembler

# COMMAND ----------

## Transforming a selection of features into a vector using VectorAssembler.
vecAssembler = VectorAssembler(inputCols = ['driverRacePoints', 'gridPosition','driverSeasonPoints',
                                            'drivSecPosRM1','drivSecPosRM2','drivSecPosRM3'], outputCol = "vectorized_features")

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
driverRaceDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)

#Dropping the raceyear column before splitting
driverRaceDF = driverRaceDF.drop('raceYear')

driverRaceTrainDF, driverRaceTestDF = driverRaceDF.randomSplit([0.8,0.2], seed=2018)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Training with ML FLow - Linear Regression

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  logr = LogisticRegression(featuresCol ='vectorized_features', labelCol = "drivSecPos", maxIter=20)
  logrModel = logr.fit(driverRaceTrainDF)
  
  #
  predictions = logrModel.transform(driverRaceTestDF)
  
  #
  evaluator= BinaryClassificationEvaluator()
  
  #
  trainingSummary = logrModel.summary
  
  # Create metrics
  #objectiveHistory = trainingSummary.objectiveHistory
  accuracy = trainingSummary.accuracy
  falsePositiveRate = trainingSummary.weightedFalsePositiveRate
  truePositiveRate = trainingSummary.weightedTruePositiveRate
  fMeasure = trainingSummary.weightedFMeasure()
  precision = trainingSummary.weightedPrecision
  recall = trainingSummary.weightedRecall
  areaUnderROC = trainingSummary.areaUnderROC
  testAreaUnderROC = evaluator.evaluate(predictions)
  
  # Log metrics
  #mlflow.log_metric("objectiveHistory", objectiveHistory)
  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metric("falsePositiveRate", falsePositiveRate)
  mlflow.log_metric("truePositiveRate", truePositiveRate)
  mlflow.log_metric("fMeasure", fMeasure)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("areaUnderROC", areaUnderROC)
  mlflow.log_metric("testAreaUnderROC", testAreaUnderROC)
  
  # Collecting the Feature Importance through Model Coefficients
  importance = pd.DataFrame(list(zip(driverRaceTrainDF.columns, logrModel.coefficients)), 
                            columns=["Feature", "Importance"]
                          ).sort_values("Importance", ascending=False)
  
  # Log importances using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
  temp_name = temp.name
  try:
    importance.to_csv(temp_name, index=False)
    mlflow.log_artifact(temp_name, "feature-importance.csv")
  finally:
    temp.close() # Delete the temp file
  
  #Create ROC plot
  roc = trainingSummary.roc.toPandas()
  plt.plot(roc['FPR'],roc['TPR'])
  plt.ylabel('False Positive Rate')
  plt.xlabel('True Positive Rate')
  plt.title('ROC Curve')
  
  # Log ROC plot using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="ROC-Curve", suffix=".png")
  temp_name = temp.name
  try:
    plt.savefig(temp_name)
    mlflow.log_artifact(temp_name, "ROC-Curve.png")
  finally:
    temp.close() # Delete the temp file
  plt.show()
  
  #Create Beta-Coeff plot
  beta = np.sort(logrModel.coefficients)
  plt.plot(beta)
  plt.ylabel('Beta Coefficients')
  plt.title('Beta Coefficients Graph')
  
  # Log Beta-Coeff plot using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="Beta-Coeff", suffix=".png")
  temp_name = temp.name
  try:
    plt.savefig(temp_name)
    mlflow.log_artifact(temp_name, "Beta-Coeff.png")
  finally:
    temp.close() # Delete the temp file
  plt.show()
  
  #Create Precision-Recall plot
  pr = trainingSummary.pr.toPandas()
  plt.plot(pr['recall'],pr['precision'])
  plt.ylabel('Precision')
  plt.xlabel('Recall')
  plt.title('Precision-Recall Curve')
  
  # Log Precision-Recall plot using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="Precision-Recall", suffix=".png")
  temp_name = temp.name
  try:
    plt.savefig(temp_name)
    mlflow.log_artifact(temp_name, "Precision-Recall.png")
  finally:
    temp.close() # Delete the temp file
  plt.show()
  
  print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Logistic Regression using Statmodels API

# COMMAND ----------


# importing libraries
import statsmodels.api as sm
import pandas as pd 

# defining the dependent and independent variables
Xtrain = driverRaceTrainDF.select('driverRacePoints', 'gridPosition','driverSeasonPoints',
                                      'drivSecPosRM1','drivSecPosRM2','drivSecPosRM3').toPandas() 
ytrain = driverRaceTrainDF.select('drivSecPos').toPandas()
   
# building the model and fitting the data
log_reg = sm.Logit(ytrain, Xtrain).fit()


# printing the summary table
print(log_reg.summary())

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Prediction

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

# MAGIC %md 
# MAGIC #### Model Evaluation 

# COMMAND ----------

evaluator= BinaryClassificationEvaluator()
print('Test Area Under ROC', evaluator.evaluate(predictions))