import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from delta import configure_spark_with_delta_pip
import mlflow
import mlflow.spark

# Настройка Spark-сессии с поддержкой Delta Lake
print(">>> Инициализация Spark-сессии с Delta Lake...")
builder = (
    SparkSession.builder.appName("FlightDelayPrediction")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.databricks.delta.optimizeWrite.enabled", "true")
    .config("spark.databricks.delta.autoCompact.enabled", "true")
)
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Настройка MLflow
print(">>> Настройка MLflow...")
mlflow.set_tracking_uri("file:/app/logs/mlruns")
mlflow.set_experiment("Flight Delay Prediction Experiment")

# Bronze Layer: Загрузка сырых данных
print(">>> [3] Bronze Layer: Загрузка сырых данных из CSV...")
bronze_path = "/app/data/bronze"

schema = StructType([
    StructField("FL_DATE", StringType(), True),
    StructField("OP_CARRIER", StringType(), True),
    StructField("OP_CARRIER_FL_NUM", IntegerType(), True),
    StructField("ORIGIN", StringType(), True),
    StructField("DEST", StringType(), True),
    StructField("CRS_DEP_TIME", IntegerType(), True),
    StructField("DEP_TIME", DoubleType(), True),
    StructField("DEP_DELAY", DoubleType(), True),
    StructField("TAXI_OUT", DoubleType(), True),
    StructField("WHEELS_OFF", DoubleType(), True),
    StructField("WHEELS_ON", DoubleType(), True),
    StructField("TAXI_IN", DoubleType(), True),
    StructField("CRS_ARR_TIME", IntegerType(), True),
    StructField("ARR_TIME", DoubleType(), True),
    StructField("ARR_DELAY", DoubleType(), True),
    StructField("CANCELLED", DoubleType(), True),
    StructField("CANCELLATION_CODE", StringType(), True),
    StructField("DIVERTED", DoubleType(), True),
    StructField("CRS_ELAPSED_TIME", DoubleType(), True),
    StructField("ACTUAL_ELAPSED_TIME", DoubleType(), True),
    StructField("AIR_TIME", DoubleType(), True),
    StructField("DISTANCE", DoubleType(), True),
    StructField("CARRIER_DELAY", DoubleType(), True),
    StructField("WEATHER_DELAY", DoubleType(), True),
    StructField("NAS_DELAY", DoubleType(), True),
    StructField("SECURITY_DELAY", DoubleType(), True),
    StructField("LATE_AIRCRAFT_DELAY", DoubleType(), True),
    StructField("Unnamed: 27", StringType(), True) 
])


df_raw = spark.read.csv(
    "/app/2018_small.csv",
    header=True,
    schema=schema
)

if "Unnamed: 27" in df_raw.columns:
    df_raw = df_raw.drop("Unnamed: 27")

print(f">>> Запись в Bronze Delta-таблицу с партиционированием по FL_DATE...")
(df_raw.write
 .format("delta")
 .mode("overwrite")
 .partitionBy("FL_DATE")
 .save(bronze_path))

print(">>> Bronze-таблица успешно создана.")

# Silver Layer: Очистка и обогащение данных
print(">>> Silver Layer: Очистка и обогащение данных...")
silver_path = "/app/data/silver"
df_bronze = spark.read.format("delta").load(bronze_path)

df_filtered = (
    df_bronze.filter((col("CANCELLED") == 0) & (col("DIVERTED") == 0))
    .withColumn("is_delayed", when(col("ARR_DELAY") > 15, 1).otherwise(0))
)

feature_cols = [
    "CRS_DEP_TIME", "CRS_ARR_TIME", "OP_CARRIER", "ORIGIN",
    "DEST", "DISTANCE", "CRS_ELAPSED_TIME",
]
target_col = "is_delayed"

df_silver_base = df_filtered.select(feature_cols + [target_col])

numeric_features = [c for c, t in df_silver_base.dtypes if t in ('int', 'double', 'bigint')]
imputer = Imputer(inputCols=numeric_features, outputCols=numeric_features).setStrategy("mean")
df_silver = imputer.fit(df_silver_base).transform(df_silver_base)


print(">>> Запись в Silver Delta-таблицу с перепартиционированием...")
(df_silver.repartition(20)
 .write
 .format("delta")
 .mode("overwrite")
 .save(silver_path))

print(">>>  Оптимизация Silver-таблицы с Z-ORDER...")
spark.sql(f"OPTIMIZE delta.`{silver_path}` ZORDER BY (ORIGIN, DEST)")
print(">>> Silver-таблица успешно создана и оптимизирована.")


# Gold Layer: Подготовка данных для ML
print(">>> Gold Layer: Подготовка данных для ML...")
gold_path = "/app/data/gold"
df_silver_loaded = spark.read.format("delta").load(silver_path)

categorical_cols = ["OP_CARRIER", "ORIGIN", "DEST"]
indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_index", handleInvalid="keep")
    for c in categorical_cols
]
assembler_inputs = [f"{c}_index" for c in categorical_cols] + numeric_features
vector_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

pipeline = Pipeline(stages=indexers + [vector_assembler])
df_gold = pipeline.fit(df_silver_loaded).transform(df_silver_loaded)

df_gold_final = df_gold.select("features", col(target_col).alias("label"))

print(">>> Запись в Gold Delta-таблицу...")
df_gold_final.write.format("delta").mode("overwrite").save(gold_path)
print(">>> Gold-таблица, готовая для ML, успешно создана.")


#  ML Моделирование с MLflow
print(">>>  ML: Обучение модели и логирование в MLflow...")
df_ml_ready = spark.read.format("delta").load(gold_path)

train_data, test_data = df_ml_ready.randomSplit([0.8, 0.2], seed=42)
train_data.cache()
test_data.cache()

print(f"Размер обучающей выборки: {train_data.count()}")
print(f"Размер тестовой выборки: {test_data.count()}")

with mlflow.start_run(run_name="LogisticRegression_Flights") as run:
    lr = LogisticRegression(featuresCol="features", labelCol="label")
    model = lr.fit(train_data)

    predictions = model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    print(f">>> Логирование результатов...")
    mlflow.log_param("reg_param", lr.getRegParam())
    mlflow.log_param("elastic_net_param", lr.getElasticNetParam())
    mlflow.log_metric("AUC", auc)
    print(f"  AUC на тестовой выборке: {auc}")
    mlflow.spark.log_model(model, "logistic-regression-flight-model")
    print(f">>> MLflow run ID: {run.info.run_id}")

print(">>> Скрипт успешно завершен!")
spark.stop()