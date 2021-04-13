# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## driver_results Prediction Model
# MAGIC 
# MAGIC This file is for creating a Model to Predict (Q2) if a Driver will end up in second position.

# COMMAND ----------

#Loading required Functions and Packages
from pyspark.sql import Window
from pyspark.sql.functions import datediff, current_date, avg, col, round, upper, max, min, lit, sum, countDistinct, concat
import pyspark.sql.functions as psf
from pyspark.sql.types import IntegerType
import numpy as np
import matplotlib as mp
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
import tempfile
from pyspark.sql.types import DoubleType

import matplotlib.pyplot as plt

from numpy import savetxt

dbutils.library.installPyPI("mlflow", "1.14.0")
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

from sklearn.ensemble import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler,StandardScaler,StringIndexer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# COMMAND ----------

#Reading Drivers Performance data prepared for modeling with features
driverRaceDF= spark.read.csv('s3://group3-gr5069/processed/driver_race_results_mod_feat.csv', header=True, inferSchema=True)
driverRaceDF = driverRaceDF.drop('_c0')

#Line of code to check data quality when needed
#driverRaceDF.select('avg_raceDur').withColumn('isNull_c',psf.col('avg_raceDur').isNull()).where('isNull_c = True').count()

# COMMAND ----------

# Dropping the pitstops data since this data is only available post 2011 races
driverRaceDF = driverRaceDF.drop("totPitstopDur","avgPitstopDur","countPitstops","firstPitstopLap","raceDate")

# COMMAND ----------

# Dropping similar columns to target variables
driverRaceDF = driverRaceDF.drop("positionOrder","driverRacePoints",'drivSecPosCat')
#driverRaceDF = driverRaceDF.withColumn('drivSecPosCat', driverRaceDF['drivSecPosCat'].cast(DoubleType()))

# COMMAND ----------

## Transforming a selection of features into a vector using VectorAssembler.
vecAssembler = VectorAssembler(inputCols = ['resultId', 'raceYear','constructorId','raceId','driverStPosition','gridPosition',
                                             'driverSeasonPoints', 'driverSeasonWins',
                                            'constSeasonPoints', 'constSeasonWins', 
                                            'drivSecPosRM1','drivSecPosRM2','drivSecPosRM3'], outputCol = "vectorized_features")
driverRaceDF = vecAssembler.transform(driverRaceDF)

# COMMAND ----------

## 
scaler = StandardScaler()\
         .setInputCol('vectorized_features')\
         .setOutputCol('features')

scaler_model = scaler.fit(driverRaceDF)

#
driverRaceDF = scaler_model.transform(driverRaceDF)

# COMMAND ----------

## 
label_indexer = StringIndexer()\
         .setInputCol ("drivSecPos")\
         .setOutputCol ("label")

label_indexer_model=label_indexer.fit(driverRaceDF)
driverRaceDF=label_indexer_model.transform(driverRaceDF)

# COMMAND ----------

#Splitting the data frame into Train and Test data based on the requirements of the Project
driverRaceTrainDF = driverRaceDF.filter(driverRaceDF.raceYear <= 2010)
driverRaceTestDF = driverRaceDF.filter(driverRaceDF.raceYear > 2010)

# COMMAND ----------

driverRaceTrainDF.groupby('label').count().show()

# COMMAND ----------

# Converting the Spark Dataframes into Pandas Dataframes
#driverRaceTrainDF =driverRaceTrainDF.select("*").toPandas()
#driverRaceTestDF =driverRaceTestDF.select("*").toPandas()

# COMMAND ----------

#X_train = driverRaceTrainDF.drop(['finishPosition'], axis=1)
#X_test = driverRaceTestDF.drop(['finishPosition'], axis=1)
#y_train = driverRaceTrainDF['finishPosition']
#y_test = driverRaceTestDF['finishPosition']

# COMMAND ----------

import sklearn
print(sklearn.__version__)

# COMMAND ----------

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  rf = RandomForestRegressor(featuresCol="vectorized_features", labelCol = "drivSecPos", numTrees=1000,maxDepth = 10 )
  logrModel = rf.fit(driverRaceTrainDF)
  #numTrees=1000, maxDepth = 10
  #
  predictions = rfModel.transform(driverRaceTestDF)
  
  #
  evaluator= BinaryClassificationEvaluator()
  
  #
  trainingSummary = rfModel.summary
  
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

logrModel.coefficients

# COMMAND ----------

trainingSummary.pValues

# COMMAND ----------

predictions = logrModel.transform(driverRaceTestDF)
predictions.select('label', 'features',  'rawPrediction', 'prediction', 'probability').toPandas().head(5)

# COMMAND ----------

accuracy = predictions.filter(predictions.drivSecPos == predictions.prediction).count() / float(predictions.count())
print("Accuracy : ",accuracy)

# COMMAND ----------

# Obtain the objective per iteration
objectiveHistory = trainingSummary.objectiveHistory
print("objectiveHistory:")
for objective in objectiveHistory:
    print(objective)

# for multiclass, we can inspect metrics on a per-label basis
print("False positive rate by label:")
for i, rate in enumerate(trainingSummary.falsePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("True positive rate by label:")
for i, rate in enumerate(trainingSummary.truePositiveRateByLabel):
    print("label %d: %s" % (i, rate))

print("Precision by label:")
for i, prec in enumerate(trainingSummary.precisionByLabel):
    print("label %d: %s" % (i, prec))

print("Recall by label:")
for i, rec in enumerate(trainingSummary.recallByLabel):
    print("label %d: %s" % (i, rec))

print("F-measure by label:")
for i, f in enumerate(trainingSummary.fMeasureByLabel()):
    print("label %d: %s" % (i, f))

accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
      % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

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

driverRaceDF.display()

# COMMAND ----------



# COMMAND ----------

## Run 1 - Run Params: {n_estimators = 100; max_depth = 6; max_features = 3}

# Enable autolog()
# mlflow.sklearn.autolog() requires mlflow 1.11.0 or above.
mlflow.sklearn.autolog()

# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  
with mlflow.start_run():
  
  # Set the model parameters. 
  n_estimators = 100
  max_depth = 6
  max_features = 3
  
  # Create and train model.
  rf = RandomForestRegressor(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features)
  rf.fit(X_train, y_train)
  
  # Use the model to make predictions on the test dataset.
  predictions = rf.predict(X_test)
  
  # Create metrics
  mse = mean_squared_error(y_test, predictions)
  mae = mean_absolute_error(y_test, predictions)
  r2 = r2_score(y_test, predictions)
  print("  mse: {}".format(mse))
  print("  mae: {}".format(mae))
  print("  R2: {}".format(r2))
  
  # Log metrics
  mlflow.log_metric("mse", mse)
  mlflow.log_metric("mae", mae)  
  mlflow.log_metric("r2", r2) 
  
  # Create feature importance
  importance = pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), 
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

  # Create plot
  fig, ax = plt.subplots()

  sns.residplot(predictions, y_test, lowess=True)
  plt.xlabel("Predicted values for Price ($)")
  plt.ylabel("Residual")
  plt.title("Residual Plot")

  # Log residuals using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
  temp_name = temp.name
  try:
    fig.savefig(temp_name)
    mlflow.log_artifact(temp_name, "residuals.png")
  finally:
    temp.close() # Delete the temp file

  display(fig)

# COMMAND ----------

