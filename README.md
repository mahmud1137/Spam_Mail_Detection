Spam Mail Detection:

This repository contains a machine learning pipeline for classifying text messages and emails as either 'spam' or 'ham' (legitimate). The project explores various natural language processing (NLP) techniques and machine learning models to effectively identify spam while handling class imbalance.

Project Overview:

Spam detection is a classic text classification problem. In this project, we process raw text data, extract features using TF-IDF, balance the dataset using SMOTE, and evaluate four different classification algorithms to determine the most effective approach.

Repository Structure:

experiment.ipynb: The main Jupyter Notebook containing the code for data cleaning, feature engineering, model training, and evaluation.
literature_review.md: Summaries of academic papers related to spam filtering, text classification, and handling imbalanced datasets.
methodology.md: A detailed breakdown of the research question, data processing steps, and evaluation metrics.

Methodology:

Data Preprocessing: The text data undergoes rigorous cleaning, which includes lowercasing, and the removal of punctuation, stopwords, numbers, URLs, and email addresses. The remaining text is lemmatized using NLTK's WordNetLemmatizer.

Feature Engineering: 

The cleaned text is converted into a numerical matrix using TfidfVectorizer (capped at 5,000 max features).

Class Balancing:

To prevent models from biasing toward the majority 'ham' class, Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training set.

Models Trained:

Multinomial Naive Bayes, Logistic Regression, Random Forest Classifier, Linear Support Vector Classifier (LinearSVC)

Results:

All models benefited significantly from the text preprocessing and SMOTE balancing.
Top Performers: LinearSVC and Logistic Regression achieved the highest accuracy (both 98.6%) and balanced macro F1-scores of 0.98.
Other Models Random Forest achieved an accuracy of 98.3%, while Multinomial Naive Bayes yielded an accuracy of 95.1%, struggling slightly more with false positives.
