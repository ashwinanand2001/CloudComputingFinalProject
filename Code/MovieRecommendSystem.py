# importing libraries needed for code to run
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression

# Starting a spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Making database and reading the data from csv file
database_path = "/home/ashwin/Desktop/CloudComputingandBigData/FinalProject/database.csv"
database = spark.read.csv(database_path, header=True, inferSchema=True)

# Selecting columns of data that is needed for
custom_data = database.select("Series_Title", "Overview", "IMDB_Rating")

# Tokenizing Movies/TV Title and Overivew
token_title = Tokenizer(inputCol="Series_Title", outputCol="title_words")
token_overview = Tokenizer(inputCol="Overview", outputCol="overview_words")

# Vectorizing Movies/TV Title and Overview
vector_title = CountVectorizer(inputCol="title_words", outputCol="title_features")
vector_overview = CountVectorizer(inputCol="overview_words", outputCol="overview_features")

# Combining the vectorizers together
combiner = VectorAssembler(inputCols=["title_features", "overview_features"], outputCol="features")

# Training model using linear regression
linear_reg_train = LinearRegression(featuresCol="features", labelCol="IMDB_Rating")

# Making a pipeline to be used in stages
pipeline_stages = Pipeline(stages=[token_title, token_overview, vector_title, vector_overview, combiner, linear_reg_train])

# Sending custom data through pipeline stages for model
pipeline_training_model = pipeline_stages.fit(custom_data)

# Search for movie recommendation based on user_search_input
user_search_input = "Christmas Movies"
user_search_data = spark.createDataFrame([(user_search_input,"")], ["Series_Title","Overview"])
user_search_data_transformed = pipeline_training_model.transform(user_search_data)

# Calculated the predicted rating of movie based on user_search_input
user_search_recommendations = user_search_data_transformed.select("Series_Title","prediction").orderBy(col("prediction").desc())
user_search_recommendations.show()

















