# Movie Recommender System

**Movie Recommender System** is simple recommender system for movies built with PySpark.

**This system was designed with the following purposes in mind:**

- Recommend n number of movies according to the users preference
- Learn more about Spark and MLlib
- To learn collaborative filtering in action

### Specifications

- Python 2.7.13
- PySpark 2.1.0
- AWS EMR Cluster
- AWS S3
- Spark's Alternating Least Squares algorithm
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)


### Getting Started

How to install spark?

- Follow the link to a guide on how to install [Spark](https://github.com/KristianHolsheimer/pyspark-setup-guide)

How to install the dependencies?

- ``` pip install -r requirements.txt ```

How to execute the program?

- ```python recommender.py```

### TODO

- ~~Added config option for user to added their movie preferences~~
- Web Interface using Flask

# Project Details

**To make recommendations in a real world application, let’s take our intuition and apply it to a machine learning algorithm called Collaborative Filtering.**

### Step 1 – Initialize The Movie Ratings

Simple but scalable scenario

* 10 movies
* 5 users
* 3 features (we’ll discuss this in Step 3)

<img src="http://i.imgur.com/MjDQbq9.png" width="300">

Let’s initialize a 10 X 5 matrix called ‘ratings’; this matrix holds all the ratings given by all users, for all movies. **Note: Not all users may have rated all movies, and this is okay.**

Here’s what the ratings matrix looks like:

```
[[ 8  4  0  0  4]
 [ 0  0  8 10  4]
 [ 8 10  0  0  6]
 [10 10  8 10 10]
 [ 0  0  0  0  0]
 [ 2  0  4  0  6]
 [ 8  6  4  0  0]
 [ 0  0  6  4  0]
 [ 0  6  0  4 10]
 [ 0  4  6  8  8]]
```

Each column represents all the movies rated by a single user
Each row represents all the ratings (from different users) received by a single movie
 

**Recall that our rating system is from 1-10. Notice how there are 0’s to denote that no rating has been given.**


### Step 2 – Determine Whether a User Rated a Movie

Let’s also declare a binary matrix (0’s and 1’s) to denote whether a user rated a movie.

`did_rate = (ratings != 0) * 1;`

### Step 3 – User Preferences and Movie Features/Characteristics

This is where it gets interesting. In order for us to build a robust recommendation engine, we need to know user preferences and movie features (characteristics). After all, a good recommendation is based off of knowing this key user and movie information.

For example, a user preference could be how much the user likes comedy movies, on a scale of 1-5. A movie characteristic could be to what degree is the movie considered a comedy, on a scale of 0-1

**Example 1: User preferences (user_prefs) -> Sample preferences for a single user Chelsea**

<img src="http://i.imgur.com/mU7fTe7.png" width="300">

**Example 2: Movie features (movie_features) -> Sample features for a single movie Bad Boys**

<img src="http://i.imgur.com/kJFTvK3.png" width="300">

Note: The user preferences are the exact same as the movie features; in other words, we can map each user preference to a movie feature. 

Note 2: We can use these numbers to ‘predict’ ratings for movies.

```
Chelsea's (C) rating (R) of Bad Boys (BB): RC,BB = comedy feature product * action feature product * romance feature product
RC,BB; = (4.5 * 0.8) + (4.9 * 0.5)  + (3.6 * 0.4)
RC,BB; = 7.49
```

**Collaborative filtering does all this for us!**

### Step 4: Rate Some Movies

Here's a list of 10 movies

```
1 Harold and Kumar Escape From Guantanamo Bay (2008)
2 Ted (2012)
3 Straight Outta Compton (2015)
4 A Very Harold and Kumar Christmas (2011)
5 Notorious (2009)
6 Get Rich Or Die Tryin' (2005)
7 Frozen (2013)
8 Tangled (2010)
9 Cinderella (2015)
10 Toy Story 3 (2010)
```

Now, rate some movies. Ratings can be represented by a 10 X 1 column vector user_ratings. Initialize it to 0’s and make some ratings:

```
user_ratings = zeros((10, 1))
user_ratings[0] = 2
user_ratings[4] = 9
```

Update ratings and did_rate with the **user_ratings**:

```
ratings = append(user_ratings, ratings, axis=1)
did_rate = append(((user_ratings != 0) * 1), did_rate, axis = 1)
```

### Step 5: Mean Normalize All The Ratings

TO recommend a movie to a user who has never placed a rating:

**We simply suggest the highest average rated movie. That’s the best we can do, since we know nothing about the user. This is made possible because of mean normalization.**

#### What is mean normalization?

* Find the average of the 1st row. In other words, find the average rating received by the first movie 
* Subtract this average from each rating (entry) in the 1st row
* The first row has now been normalized. This row now has an average of 0.
* Repeat steps 1 & 2 for all rows.

```
ratings_norm, ratings_mean = normalize_ratings(ratings, did_rate)
```

### Step 6: Collaborative Filtering with ALS (Implicit Matrix Factorization)


<img src="http://i.imgur.com/yMymkLr.jpg" width="400">

Spark MLlib library for Machine Learning provides a Collaborative Filtering implementation by using Alternating Least Squares. The implementation in MLlib has the following parameters:

	•	numBlocks is the number of blocks used to parallelize computation (set to -1 to auto-configure).
	•	rank is the number of latent factors in the model.
	•	iterations is the number of iterations to run.
	•	lambda specifies the regularization parameter in ALS.
	•	implicitPrefs specifies whether to use the explicit feedback ALS variant or one adapted for implicit feedback data.
	•	alpha is a parameter applicable to the implicit feedback variant of ALS that governs the baseline confidence in preference observations
	
<img src="http://i.imgur.com/BENh8Yx.png" width="550">


### Evaluate the model using RMSE

The use of RMSE is very common and it makes an excellent general purpose error metric for numerical predictions.

#### Root Mean Squared Error (RMSE)

The square root of the mean/average of the square of all of the error.

Compared to the similar Mean Absolute Error, RMSE amplifies and severely punishes large errors.

<img src="http://i.imgur.com/kK6SmVz.png" width="300">

```
def rmse(predictions, targets):

    differences = predictions - targets                       #the DIFFERENCEs.

    differences_squared = differences ** 2                    #the SQUAREs of ^

    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^

    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^

    return root_of_of_the_mean_of_the_differences_squared     #get the ^
```

