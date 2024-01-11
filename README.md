# Review Stars Prediction

This project is a machine learning model that predicts the star rating of a review based on the review text.
dataset from https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset 
## Overview

The project uses a pipeline of three steps:

1. **TfidfVectorizer**: This step transforms the text data into a matrix of TF-IDF features.
2. **SelectKBest with chi2**: This step selects the 10,000 best features according to the chi-squared statistic.
3. **LinearSVC**: This is the machine learning model. It's a Support Vector Machine (SVM) classifier with a linear kernel.

## Usage

To use this project, you need to have a dataset with two columns: 'cleaned' (the preprocessed text data) and 'stars' (the target variable that we want to predict).

The main script is `main.py`. Run this script to train the model and print the top 10 keywords per class and the accuracy score of the model.

You can also use the model to predict the star rating of a review by calling `model.predict(['Your review text here'])`. 

## Results
First you get top 10 keywords for each category, after that you get the amount of start for your own input. Looks something like this:

<img width="815" alt="Screenshot 2024-01-11 at 20 27 45" src="https://github.com/Sekseli3/NLP_yelpStartPredict/assets/120391401/4bb079a2-a38b-4802-8360-5fcd6acc2b78">

## Dependencies

This project requires the following Python libraries:

- numpy
- pandas
- sklearn
- nltk

## Future Work

Future improvements could include tuning the model parameters, using a different machine learning model, or using a different feature selection method.
