## Netflix Recommendation System

**Netflix Recommendation System** is simple recommender systems for movies built using PySpark.

**This system was designed with the following purposes in mind:**

- Recommend n number of movies according to the users preference.

## Specifications

- Python 2.7.13
- PySpark 2.1.0
- AWS EMR Cluster
- AWS S3
- Spark's Alternating Least Squares algorithm
- [MovieLens dataset](https://grouplens.org/datasets/movielens/)


## Getting Started

How to install spark?

- Follow the link to a guide on how to install [Spark](https://github.com/KristianHolsheimer/pyspark-setup-guide)

How to install the dependencies?

- ``` pip install -r requirements.txt ```

How to execute the program?

- ```python recommender.py```

## TODO

- Added config option for user to added their movie preferences
- Web Interface using Flask
