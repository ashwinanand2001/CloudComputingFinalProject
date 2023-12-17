''' importing libraries needed
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

# Creating a spark session for project to run on
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Setting up the database path to be stores
database_path = "/home/ashwin/Desktop/CloudComputingandBigData/FinalProject/database.csv"
dataset = spark.read.csv(database_path, header=True, inferSchema=True)

# Data preprocessing
# selecting the title and overview sections of data
selected_data = dataset.select("Series_Title", "Overview")

# Tokenize the movie titles and overviews
tokenizer = Tokenizer(inputCol="Series_Title", outputCol="title_words")
title_words_data = tokenizer.transform(selected_data)
tokenizer = Tokenizer(inputCol="Overview", outputCol="overview_words")
overview_words_data = tokenizer.transform(title_words_data)

# Converting Words to Vectors
vectorizer_title = CountVectorizer(inputCol="title_words", outputCol="title_features")
vectorized_title_data = vectorizer_title.fit(overview_words_data).transform(overview_words_data)
vectorizer_overview = CountVectorizer(inputCol="overview_words", outputCol="overview_features")
vectorized_data = vectorizer_overview.fit(vectorized_title_data).transform(vectorized_title_data)

# Build ALS model for collaborative filtering
als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="IMDB_Rating")
model = als.fit(vectorized_data)

# Example search for movie recommendations
user_search = "Christmas Movies/Shows with John Candy"
search_vector_title = vectorizer_title.transform(tokenizer.transform([(user_search,)]))
search_vector_overview = vectorizer_overview.transform(search_vector_title)
user_recommendations = model.transform(search_vector_overview)

# Show top recommendations
user_recommendations.select("Series_Title", "prediction").orderBy(col("prediction").desc()).show()

'''

# importing libraries needed
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, CountVectorizer, VectorAssembler
from pyspark.sql.functions import col
from pyspark.ml.regression import LinearRegression

# Initialize Spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# making database and reading the data from csv file
database_path = "/home/ashwin/Desktop/CloudComputingandBigData/FinalProject/database.csv"
dataset = spark.read.csv(database_path, header=True, inferSchema=True)


# selecting columns of data that is needed
selected_data = dataset.select("Series_Title", "Overview", "IMDB_Rating")

# tokenizing series title and overview
tokenizer_title = Tokenizer(inputCol="Series_Title", outputCol="title_words")
tokenizer_overview = Tokenizer(inputCol="Overview", outputCol="overview_words")

# vectorizing series title and overview
vectorizer_title = CountVectorizer(inputCol="title_words", outputCol="title_features")
vectorizer_overview = CountVectorizer(inputCol="overview_words", outputCol="overview_features")

# combining all together
assembler = VectorAssembler(inputCols=["title_features", "overview_features"], outputCol="features")

# training model using linear regression
lr = LinearRegression(featuresCol="features", labelCol="IMDB_Rating")

# making a pipeline to be used in stagees
pipeline = Pipeline(stages=[tokenizer_title, tokenizer_overview, vectorizer_title, vectorizer_overview, assembler, lr])

# sending data to the pipeline
pipeline_model = pipeline.fit(selected_data)

# seach for movie recommendations
user_search = "Christmas Movies"
search_data = spark.createDataFrame([(user_search,"")], ["Series_Title","Overview"])
search_data_transformed = pipeline_model.transform(search_data)

# displaying highes rating for user_search
user_recommendations = search_data_transformed.select("Series_Title","prediction").orderBy(col("prediction").desc())
user_recommendations.show()


