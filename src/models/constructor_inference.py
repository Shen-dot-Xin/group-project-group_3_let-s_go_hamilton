# Databricks notebook source
# MAGIC %md
# MAGIC #### Constructor Championship Inference

# COMMAND ----------

dbutils.library.installPyPI("mlflow", "1.14.0")

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, Normalizer, StandardScaler
from pyspark.sql import Window
from pyspark.sql.functions import lag, col, asc, min, max

# COMMAND ----------

df = spark.read.csv('s3://group3-gr5069/interim/constructor_features.csv', header = True, inferSchema = True)

# COMMAND ----------

window = Window.partitionBy('constructorId').orderBy(asc('year'))

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

df.columns

# COMMAND ----------

cols_to_normalize = ['avg_fastestspeed',
 'avg_fastestlap',
 'race_count',
 'engineproblem',
 'avgpoints_c',  'unique_drivers',
 'position',
 'lag1_avg',
 'lag2_avg', 'lag1_pst',
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
 'lag2_pst']

# COMMAND ----------

df = df.na.fill(value=0,subset=feature_list)

# COMMAND ----------

vecAssembler = VectorAssembler(inputCols = feature_list, outputCol = "features")

vecDF = vecAssembler.transform(df)

# COMMAND ----------

scalar = StandardScaler(inputCol="features", outputCol="ssFeatures")

ssDF = scalar.fit(vecDF)

# COMMAND ----------

ssdf = ssDF.transform(vecDF)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Logistic Regression

# COMMAND ----------

import os
import matplotlib.pyplot as plt
import mlflow.sklearn
import seaborn as sns
import tempfile
from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

df.count()

# COMMAND ----------

(trainDF, testDF) = ssdf.randomSplit([.8, .2], seed=42)

# COMMAND ----------

lr = LogisticRegression(featuresCol = "ssFeatures", labelCol = "champion")
lrModel = lr.fit(trainDF)

# COMMAND ----------

trainingSummary = lrModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# COMMAND ----------

print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# COMMAND ----------

['race_count','engineproblem','unique_drivers','lag1_avg','lag2_avg','lag1_fs','lag2_fs','lag1_fl','lag2_fl','lag1_nd','lag2_nd','lag1_standing','lag2_standing']

# COMMAND ----------

lrModel.coefficients

# COMMAND ----------

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# COMMAND ----------

with mlflow.start_run(run_name="Basic Linear Regression Experiment") as run:
  lr = LogisticRegression(featuresCol = "ssFeatures", labelCol = "champion")
  lrModel = lr.fit(ssdf)
    
  # Log model
  mlflow.sklearn.log_model(lrModel, "linear-regression-model")
    
  # Log params
  [mlflow.log_param(f) for f in features]
  
  # Log coefficients and p value
  for index, name in feature_names(lr, X):
    mlflow.log_metric(f"Coef. {name}", lrModel.coefficients[index])
  if has_pvalue(lr):
  # P-values are not always available. This depends on the model configuration.
    mlflow.log_metric(f"P-val. {name}", lrModel.summary.pValues[index])
  
  runID = run.info.run_uuid
  experimentID = run.info.experiment_id
  
  print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

# COMMAND ----------

def log_lr(experiment_ID, run_name, features):
  with mlflow.start_run(experiment_id=experiment_ID, run_name=run_name) as run:
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

log_lr("test Run", features = feature_list)

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

