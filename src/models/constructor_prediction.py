# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Prediction

# COMMAND ----------

dbutils.library.installPyPI("mlflow", "1.14.0")

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.sql import Window
from pyspark.sql.functions import lag, col, asc, min, max

import pandas as pd
import numpy as np
import os
import boto3

# COMMAND ----------

import warnings
warnings.simplefilter(action='ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

# COMMAND ----------

# MAGIC %md #### SVC Model

# COMMAND ----------

s3 = boto3.client('s3')
bucket = "group3-gr5069"

c = "interim/constructor_features.csv"

obj = s3.get_object(Bucket= bucket, Key= c)
data = pd.read_csv(obj['Body'])
data = data[(data['year']>=1950) & (data['year']<=2017)]
df = data.fillna(0)
df.head() 

# COMMAND ----------

#split training and test dataset, the former using data between 1950-2010, the later contains data between 2011-2017
y = df[['champion','year']]
X = df.loc[:, ['race_count', 'lag1_avg', 'lag1_ptc', 'lag1_pst', 'lag2_pst', 'avg_fastestspeed', 'avg_fastestlap', 'year']]
X_train = X[(X['year']<=2010)]
X_test = X[(X['year']>=2011) & (X['year']<=2017)]
y_train = y[(y['year']<=2010)]
y_test = y[(y['year']>=2011) & (y['year']<=2017)]
X_train = X_train.drop(['year'], axis=1)
X_test = X_test.drop(['year'], axis=1)
y_train = y_train.drop(['year'], axis=1)
y_test = y_test.drop(['year'], axis=1)
X_test

# COMMAND ----------

#set KFold and use GridSearchCV to find the best paramater for SVC model
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

hyper_param = {'C': [0.1, 1, 10, 100, 1000]}

SVC_model = SVC()
model= GridSearchCV(SVC_model, hyper_param, cv = kfold)
model.fit(X_train, y_train.values.ravel())

print("Best Parameters", model.best_params_)

# COMMAND ----------

from sklearn.model_selection import cross_val_score
svc = SVC(kernel='linear', C=10).fit(X_train, y_train) 

print("SVC") 
print("Test set score: {:.2f}".format(svc.score(X_test, y_test)))

# Kfold cross validation
print("Mean Cross-Validation, Kfold: {:.2f}".format(np.mean(cross_val_score(svc, X_train, y_train, cv=kfold))))

svc_accuracy = np.mean(cross_val_score(svc, X_train, y_train, cv=kfold))

# COMMAND ----------

# MAGIC %md #### Compare SVC and logistic models

# COMMAND ----------

#use logistic model with normalized data
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
hyperparameters = {'C': [0.1, 1, 10, 100, 1000]}

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

logit = LogisticRegression()
model= GridSearchCV(logit, hyperparameters, cv = kfold)
model.fit(X_train_s, y_train.values.ravel())

print("Best Parameters", model.best_params_)

# COMMAND ----------

logreg = LogisticRegression(C=0.1).fit(X_train_s, y_train)

print("LOGREG for REGRESSION")

#test and training score
print("Training set score: {:.2f}".format(logreg.score(X_train_s, y_train)))
print("Test set score: {:.2f}".format(logreg.score(X_test_s, y_test)))

#Kfold cross validation
print("Mean Cross-Validation, Kfold: {:.2f}".format(np.mean(cross_val_score(logreg, X_train_s, y_train, cv=kfold))))
logreg_accuracy = np.mean(cross_val_score(logreg, X_train_s, y_train, cv=kfold))

# COMMAND ----------

print("SVC")
print("Accuracy: {:.2f}".format(svc_accuracy))
print("Test set score: {:.2f}".format(svc.score(X_test, y_test)))
print("Logistic")
print("Accuracy: {:.2f}".format(logreg_accuracy))
print("Test set score: {:.2f}".format(logreg.score(X_test_s, y_test)))

#The mean cross validation score shows that the accuracy of svc model and logistic model are the same. While the test set score of Logistic regression model is higher than that of SVC model(0.94 vs 0.91).Therefore, we choose the logistic regression model to do the prediction and save the output to a database.

# COMMAND ----------

# MAGIC %md #### Logistic Regression Prediction

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mlflow.sklearn
import seaborn as sns
import tempfile
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# COMMAND ----------

df = spark.read.csv('s3://group3-gr5069/interim/constructor_features.csv', header = True, inferSchema = True)

# COMMAND ----------

df.columns

# COMMAND ----------

cols_to_normalize = ['avg_fastestspeed', 
                     'avg_fastestlap',
                     'race_count',
                     'engineproblem',
                     'lag1_ptc',
                     'lag1_avg',
                     'lag1_pst',
                     'lag2_pst']

# COMMAND ----------

w = Window.partitionBy('year')
for c in cols_to_normalize:
    df = (df.withColumn('mini', min(c).over(w))
        .withColumn('maxi', max(c).over(w))
        .withColumn(c, ((col(c) - col('mini')) / (col('maxi') - col('mini'))))
        .drop('mini')
        .drop('maxi'))

# COMMAND ----------

feature_list =['avg_fastestspeed', 
                     'avg_fastestlap',
                     'race_count',
                     'engineproblem',  
                     'lag1_ptc',
                     'lag1_avg',
                     'lag1_pst',
                     'lag2_pst']

# COMMAND ----------

df = df.na.fill(value=0,subset=feature_list)

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = feature_list, outputCol = "features")

# COMMAND ----------

trainDF = vecAssembler.transform(df[df['year']<=2010])
testDF = vecAssembler.transform(df[(df['year']>=2011) & (df['year']<=2017)])

# COMMAND ----------

trainDF = trainDF.na.fill(value=0)
testDF = testDF.na.fill(value=0)

# COMMAND ----------

mlflow.sklearn.autolog()
# With autolog() enabled, all model parameters, a model score, and the fitted model are automatically logged.  

with mlflow.start_run():
  lr = LogisticRegression(featuresCol ='features', labelCol = "champion")
  lrModel = lr.fit(trainDF)

  predictions = lrModel.transform(testDF)
  
  evaluator= BinaryClassificationEvaluator(labelCol='champion')
  
  # Log model
  mlflow.spark.log_model(lrModel, "logistic-regression-model-with-selected-features")
  trainingSummary = lrModel.summary
  
  # Log parameters
  #mlflow.log_param("penalty", 0.001)
  
  # Create metrics
  #objectiveHistory = testSummary.objectiveHistory
  accuracy = trainingSummary.accuracy
  precision = trainingSummary.weightedPrecision
  recall = trainingSummary.weightedRecall
  
  falsePositiveRate = trainingSummary.weightedFalsePositiveRate
  truePositiveRate = trainingSummary.weightedTruePositiveRate
  
  #fMeasure = testSummary.weightedFMeasure()
  testAreaUnderROC = evaluator.evaluate(predictions)
  
  # Log metrics
  #mlflow.log_metric("objectiveHistory", objectiveHistory)
  mlflow.log_metric("accuracy", accuracy)
  mlflow.log_metric("falsePositiveRate", falsePositiveRate)
  mlflow.log_metric("truePositiveRate", truePositiveRate)
  #mlflow.log_metric("fMeasure", fMeasure)
  mlflow.log_metric("precision", precision)
  mlflow.log_metric("recall", recall)
  mlflow.log_metric("testAreaUnderROC", testAreaUnderROC)
  # Collecting the Feature Importance through Model Coefficients
  importance = pd.DataFrame(list(zip(feature_list, lrModel.coefficients)), 
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
  #Create ROC plot for test set
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
  beta = np.sort(lrModel.coefficients)
  plt.plot(beta)
  plt.ylabel('Coefficients')
  plt.title('Coefficients Graph')
  # Log Beta-Coeff plot using a temporary file
  temp = tempfile.NamedTemporaryFile(prefix="Coeff", suffix=".png")
  temp_name = temp.name
  try:
    plt.savefig(temp_name)
    mlflow.log_artifact(temp_name, "Coeff.png")
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

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

predictions.columns

# COMMAND ----------

predDF_final = predictions.select('year',
 'constructorId',
 'avg_fastestspeed',
 'avg_fastestlap',
 'race_count',
 'engineproblem',
 'avgpoints_c',
 'participation',
 'gp_1',
 'gp_2',
 'gp_3',
 'gp_4',
 'gp_5',
 'gp_6',
 'gp_7',
 'gp_8',
 'gp_9',
 'gp_10',
 'gp_11',
 'gp_12',
 'gp_13',
 'gp_14',
 'gp_15',
 'gp_16',
 'gp_17',
 'gp_18',
 'gp_19',
 'gp_20',
 'gp_21',
 'gp_22',
 'gp_23',
 'gp_24',
 'gp_25',
 'gp_26',
 'gp_27',
 'gp_28',
 'gp_29',
 'gp_30',
 'gp_31',
 'gp_32',
 'gp_33',
 'gp_34',
 'gp_35',
 'gp_36',
 'gp_37',
 'gp_38',
 'gp_39',
 'gp_40',
 'gp_41',
 'gp_42',
 'gp_43',
 'gp_44',
 'gp_45',
 'gp_46',
 'gp_47',
 'gp_48',
 'gp_49',
 'gp_50',
 'gp_51',
 'gp_52',
 'gp_53',
 'gp_54',
 'gp_55',
 'gp_56',
 'gp_57',
 'gp_58',
 'gp_59',
 'gp_60',
 'gp_61',
 'gp_62',
 'gp_63',
 'gp_64',
 'gp_68',
 'gp_69',
 'gp_70',
 'gp_71',
 'gp_73',
 'unique_drivers',
 'position',
 'lag1_avg',
 'lag2_avg',
 'lag1_ptc',
 'lag2_ptc',
 'lag1_pst',
 'lag2_pst',
 'champion',
 'prediction')

# COMMAND ----------

predDF_final.write.format('jdbc').options(
      url='jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200',
      driver='com.mysql.jdbc.Driver',
      dbtable='test_lr_preds',
      user='admin',
      password='Xs19980312!').mode('overwrite').save()

# COMMAND ----------

# MAGIC %md #### Read from db

# COMMAND ----------

predDF_final_done = spark.read.format("jdbc").option("url", "jdbc:mysql://sx2200-gr5069.ccqalx6jsr2n.us-east-1.rds.amazonaws.com/sx2200") \
    .option("driver", "com.mysql.jdbc.Driver").option("dbtable", "test_lr_preds") \
    .option("user", "admin").option("password", "Xs19980312!").load()

# COMMAND ----------

# MAGIC %md #### Marginal Effects 

# COMMAND ----------

from sklearn.inspection import plot_partial_dependence, partial_dependence
from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
%matplotlib inline

# COMMAND ----------

factors = X_train[['race_count','lag1_avg']]
# plot the partial dependence (marginal effect)
plot_partial_dependence(logreg, X_train, factors)  
# get the partial dependence (marginal effect)
partial_dependence(logreg, X_train_s, [0])  

# COMMAND ----------

# From the plots, the marginal effect of race_count and lag1_avg are shown. The relationship between races completed and whether or not the constructor would win a season is a linear relation with positive slope, and the line is quite cliffy. The relation between average points earned in last season and the constructor championship is also positive, while the slope of this curve is smaller than the slope of race_count.