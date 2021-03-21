# Executive Summary

## Problem Statement

Big Tech comprises the largest and most valuable companies in the world. Also known as the S&P 5, they are:
- Facebook (r/facebook)
- Apple (r/apple)
- Amazon (r/amazon)
- Google (r/google)
- Microsoft (r/microsoft)
We want to be able to correctly classify when a comment or post in reddit belongs to either companies subreddit.


The process we go through in this project is as follows:
1. Successfully use the Pushshift API (see https://github.com/pushshift/api) to collect and store text data from each subreddit. A built-in function is used that loops through both post submissions and comments

2. Clean, transform, and analyze this data for NLP. We do this through
- Dropping any duplicated rows.
- Filling any Null values.
- Using redditcleaner() to clean all text data.
- Remove any numbers or text under 2 letters.
- Lemmatizing all words.

3. Pass data into a machine learning model for multi-label classification using CountVectorizer(). Then running GridSearchCV across 7 possible models such as MultiNomial Naive Bayes, Random Forests, Logistic Regression, SVM, etc.

4. Optimize model structure and hyperparameters for performance metrics (accuracy, precision, recall). We use GridSearchCV again to dial down and optimize any promising models, then combine them in a Stacked Voting Classifier.


## Data Description
23651 submissions. (~4500 each from each company subreddit).

# Conclusion
Our baseline accuracy was 0.213%. Through the processes described above we are able to increase test accuracy to 0.785% -- a nearly **4X** increase relative to the baseline accuracy. 
Our stacked voting classifier allowed us to leverage the individual abilities of each model. For instance, some models had higher sensitivity, and were able to maximize true positives. Another model might be more precise, or have mapped out the difference between Apple and Microsoft better. The model's final predicition is able to allow each model to state their probablities, and pick the highest cumulative score. 
Some Potential Improvements:
Collect more training data.
Add more stop_words in preprocessing.
Feature engineering (especially with Apple/Microsoft, where overlapping creates noise).
