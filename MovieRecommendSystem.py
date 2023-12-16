# importing libraries needed
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



