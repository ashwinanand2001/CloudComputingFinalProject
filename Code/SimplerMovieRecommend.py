# importing libraries needed for code to run
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer
from pyspark.sql.functions import monotonically_increasing_id, array_contains

# Starting a spark session
spark = SparkSession.builder.appName("MovieRecommendation").getOrCreate()

# Reading the data from csv file into spark data frame
database = spark.read.csv("database.csv", header=True, inferSchema=True)

# Adding item to the database to make every movie unique
database_dataframe = database.withColumn("item_id", monotonically_increasing_id())

# Tokenizing Overivew
token = Tokenizer(inputCol="Overview", outputCol="token_overview")
database_dataframe = token.transform(database_dataframe)

# Display token data from Overview
print("Token Overview:")
database_dataframe.select("item_id", "token_overview").show(truncate=False)

# Movies filtered by user search
Data_Filtered = database_dataframe.filter(array_contains(database_dataframe["token_overview"], "christmas"))

# Display filtered data
print("Filtered Data:")
Data_Filtered.select("item_id", "Series_Title", "IMDB_Rating").show(truncate=False)





















