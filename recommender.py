import os
import math
import time
import config
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row, SparkSession
from pyspark.mllib.recommendation import ALS

# Adding paths to the datasets
large_dataset_path = 'dataset/movies_large/'
small_dataset_path = 'dataset/movies_small/'

# Getting the SparkContext
sc = SparkContext()
# Initializing the SQLContext
sqlContext = SQLContext(sc)
# Initializing Spark Session
spark = SparkSession \
    .builder \
    .appName("netflix-recommendation-system") \
    .getOrCreate()


# Creating the Dataframe for the small dataset using SQLContext
small_ratings_file = os.path.join(small_dataset_path, 'ratings.csv')
small_ratings_raw_data = sc.textFile(small_ratings_file)
small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]
small_ratings_raw_data_DF = sqlContext.read.csv(small_ratings_file, header=True, inferSchema=True)
small_ratings_raw_data_DF.show(10)

# Creating dataframe for visualization in temp table 'D'
data = sc.textFile(small_ratings_file)
data = data.filter(lambda line: line != small_ratings_raw_data_header).map(lambda line: line.split(',')). \
    map(lambda x: Row(userId=int(x[0]), movieId=int(x[1]), rating=float(x[2]), timestamp=str(x[3])))
dataDF = sqlContext.createDataFrame(data)
dataDF.registerTempTable("D")

# Displaying the temp table "D"
print(spark.sql("Select * from D").show())

# Creating RDD using only userID, movieID, rating since we don't need timestamp
small_ratings_data = small_ratings_raw_data \
    .filter(lambda line: line != small_ratings_raw_data_header) \
    .map(lambda line: line.split(",")) \
    .map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()

# Creating the small dataset Dataframe
small_movies_file = os.path.join(small_dataset_path, 'movies.csv')
small_movies_raw_data = sc.textFile(small_movies_file)
small_movies_raw_data_header = small_movies_raw_data.take(1)[0]
small_movies_raw_data = sc.textFile(small_movies_file)
data = small_movies_raw_data.filter(lambda line: line != small_movies_raw_data_header).map(
    lambda line: line.split(',')). \
    map(lambda x: Row(movieId=int(x[0]), title=(x[1]).encode('utf-8')))
dataDF = sqlContext.createDataFrame(data)

# Displaying the dataframe schema
print(dataDF.select("movieId", "title").show())

# Validation datasets
training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0L)
validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))

# ALS algorithm configuration
seed = config.seed
iterations = config.iterations
regularization_parameter = config.regularization_parameter
ranks = config.ranks
errors = config.errors
err = config.err
tolerance = config.tolerance

# ALS algorithm training step
min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations, lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_predictions = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    errors[err] = error
    err += 1
    print('For rank %s the RMSE is %s' % (rank, error))
    if error < min_error:
        min_error = error
        best_rank = rank

print('The best model was trained with rank %s' % best_rank)

model = ALS.train(training_RDD, best_rank, seed=seed, iterations=iterations,
                  lambda_=regularization_parameter)

predictions = model.predictAll(test_for_predict_RDD) \
    .map(lambda r: ((r[0], r[1]), r[2]))

rates_and_predictions = test_RDD \
    .map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))) \
    .join(predictions)

error = math.sqrt(rates_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

print('For testing data the RMSE is %s' % error)

complete_movies_file = os.path.join(large_dataset_path, 'movies.csv')
complete_movies_raw_data = sc.textFile(complete_movies_file)
complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]

# Parse
complete_movies_data = complete_movies_raw_data \
    .filter(lambda line: line != complete_movies_raw_data_header) \
    .map(lambda line: line.split(",")) \
    .map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).cache()

complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

print("There are %s movies in the large dataset" % (complete_movies_titles.count()))

complete_ratings_file = os.path.join(large_dataset_path, 'ratings.csv')
complete_ratings_raw_data = sc.textFile(complete_ratings_file)
complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

complete_ratings_data = complete_ratings_raw_data \
    .filter(lambda line: line != complete_ratings_raw_data_header) \
    .map(lambda line: line.split(",")) \
    .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
    .cache()

print('There are %s recommendations in the large dataset' % (complete_ratings_data.count()))


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1])) / nratings)


movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

new_user_ID = 0

# The format of each line is (userID, movieID, rating)
new_user_ratings = [
    (0, 921, 4),
    (0, 272, 3),
    (0, 296, 4),
    (0, 300, 5),
    (0, 305, 3),
    (0, 370, 2),
    (0, 441, 3),
    (0, 468, 3),
    (0, 680, 2),
    (0, 750, 4),
    (0, 778, 4),
    (0, 908, 4),
    (0, 921, 3),
]
new_user_ratings_RDD = sc.parallelize(new_user_ratings)
print('New user ratings: %s' % new_user_ratings_RDD.take(10))

complete_data_with_new_ratings_RDD = small_ratings_data.union(new_user_ratings_RDD)

t0 = time.time()
new_ratings_model = ALS.train(complete_data_with_new_ratings_RDD, best_rank, seed=seed,
                              iterations=iterations, lambda_=regularization_parameter)
tt = time.time() - t0

print("New model trained in %s seconds" % round(tt, 3))

new_user_ratings_ids = map(lambda x: x[1], new_user_ratings)

new_user_unrated_movies_RDD = complete_movies_data \
    .filter(lambda x: x[0] not in new_user_ratings_ids) \
    .map(lambda x: (new_user_ID, x[0]))

print(new_user_unrated_movies_RDD.count())

new_user_recommendations_RDD = new_ratings_model.predictAll(new_user_unrated_movies_RDD)

print(new_user_recommendations_RDD.take(5))

new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)

print(new_user_recommendations_rating_title_and_count_RDD.take(6))

new_user_recommendations_rating_title_and_count_RDD = \
    new_user_recommendations_rating_title_and_count_RDD \
        .map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))

top_movies = new_user_recommendations_rating_title_and_count_RDD \
    .filter(lambda r: r[2] >= 25) \
    .takeOrdered(25, key=lambda x: -x[1])

print('Recommended movies for you (with more than 25 reviews):\n%s' %
      '\n'.join(map(str, top_movies)))
