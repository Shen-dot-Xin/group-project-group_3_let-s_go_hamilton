# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Inference

# COMMAND ----------

dbutils.library.installPyPI("mlflow", "1.14.0")

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Normalizer
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

feature_list =['avg_fastestspeed','avg_fastestlap','race_count','engineproblem','unique_drivers','lag1_avg','lag2_avg','lag1_fs','lag2_fs','lag1_fl','lag2_fl','lag1_nd','lag2_nd','lag1_standing','lag2_standing']

# COMMAND ----------

df = df.na.fill(value=0,subset=feature_list)

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = feature_list, outputCol = "features")

vecDF = vecAssembler.transform(df)

# COMMAND ----------

normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)

NormDF = normalizer.transform(vecDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Linear Regression

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import mlflow.sklearn
import seaborn as sns

from pyspark.sql.functions import stddev
from pyspark.ml.regression import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score 
from sklearn.metrics import roc_curve, auc
import tempfile

# COMMAND ----------

def log_lr(experimentID, run_name, features):
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    X = df.select(features)
    
    # Create model and train it
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Log model
    mlflow.sklearn.log_model(lr, "linear-regression-model")
    
    # Log params
    [mlflow.log_param(f) for f in features]
    
    # Create metrics
    rmse = lr.summary.rootMeanSquaredError
    r2 = lr.summary.r2
     
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 
    print("  rmse: {}".format(rmse))
    print("  r2: {}".format(r2)) 
  
    # Log coefficients and p value
    for index, name in feature_names(lr, X):
      mlflow.log_metric(f"Coef. {name}", lr.coefficients[index])
    if has_pvalue(lr):
      # P-values are not always available. This depends on the model configuration.
      mlflow.log_metric(f"P-val. {name}", lr.summary.pValues[index])

    # Log importances using a temporary file
    #temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    #temp_name = temp.name
    #try:
    #  importance.to_csv(temp_name, index=False)
    #  mlflow.log_artifact(temp_name, "feature-importance.csv")
    #finally:
    #  temp.close() # Delete the temp file
    
    # Create plot - roc curve
    # get false and true positive rates
    #fpr, tpr, t = roc_curve(y_test, predicted_proba[:,1])
    # get area under the curve
    #roc_auc = auc(fpr, tpr)
    
    # PLOT ROC curve
    #fig, ax = plt.subplots()
    #plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
    #plt.title('ROC Curve for RF classifier')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate (Recall)')
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.legend()

    # Log residuals using a temporary file
    #temp = tempfile.NamedTemporaryFile(prefix="ROC-", suffix=".png")
    #temp_name = temp.name
    #try:
    #  fig.savefig(temp_name)
    #  mlflow.log_artifact(temp_name, "ROC.png")
    #finally:
    #  temp.close() # Delete the temp file
    #  
    #display(fig)
    return run.info.run_uuid


# COMMAND ----------

log_lr(experimentID, "test Run", features = feature_list)

# COMMAND ----------

lr = LinearRegression(featuresCol = "features", labelCol = "champion")
lrModel = lr.fit(vecDF)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md #### Logistic Regression

# COMMAND ----------

def log_lr(experimentID, run_name, features):
  with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
    X = df.select(features)
    
    # Create model and train it
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Log model
    mlflow.sklearn.log_model(lr, "linear-regression-model")
    
    # Log params
    [mlflow.log_param(f) for f in features]
    
    # Create metrics
    rmse = lr.summary.rootMeanSquaredError
    r2 = lr.summary.r2
     
    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2) 
    print("  rmse: {}".format(rmse))
    print("  r2: {}".format(r2)) 
    
    # Create feature importance
    #importance = pd.DataFrame(list(zip(df_XY[['qPosr','dprStandingr','dprWinr','cprStandingr', 'grid']].columns, rf.feature_importances_)), 
                                #columns=["Feature", "Importance"]
                             # ).sort_values("Importance", ascending=False)
    
    #df_im = stddev
    
    # Log importances using a temporary file
    #temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
    #temp_name = temp.name
    #try:
    #  importance.to_csv(temp_name, index=False)
    #  mlflow.log_artifact(temp_name, "feature-importance.csv")
    #finally:
    #  temp.close() # Delete the temp file
    
    # Create plot - roc curve
    # get false and true positive rates
    #fpr, tpr, t = roc_curve(y_test, predicted_proba[:,1])
    # get area under the curve
    #roc_auc = auc(fpr, tpr)
    
    # PLOT ROC curve
    #fig, ax = plt.subplots()
    #plt.plot(fpr, tpr, lw=1, color='green', label=f'AUC = {roc_auc:.3f}')
    #plt.title('ROC Curve for RF classifier')
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate (Recall)')
    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.legend()

    # Log residuals using a temporary file
    #temp = tempfile.NamedTemporaryFile(prefix="ROC-", suffix=".png")
    #temp_name = temp.name
    #try:
    #  fig.savefig(temp_name)
    #  mlflow.log_artifact(temp_name, "ROC.png")
    #finally:
    #  temp.close() # Delete the temp file
    #  
    #display(fig)
    return run.info.run_uuid


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

