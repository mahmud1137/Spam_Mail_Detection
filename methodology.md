Research Question: How effectively can traditional machine learning models classify text messages or emails as 'spam' or 'ham' using TF-IDF feature extraction and balanced training data?

Dataset Description: The dataset consists of text data with two main columns: label (target variable, containing 'ham' or 'spam') and text (the raw message content). The initial dataset exhibits a class imbalance, with 3,672 'ham' examples and 1,499 'spam' examples before balancing. A limitation is that standard text datasets can contain noisy artifacts like raw HTML, URLs, and arbitrary numbers.

Data Cleaning Plan: The text will be converted to lowercase. Subsequent steps include the removal of punctuation, stopwords, numbers, URLs, emails, and special characters. Finally, the cleaned text will undergo lemmatization using NLTK's WordNetLemmatizer.

Feature Engineering Plan: The cleaned text will be transformed into numerical features using TfidfVectorizer with english stop words and a maximum of 5,000 features. To address the class imbalance, SMOTE (Synthetic Minority Over-sampling Technique) will be applied to the training set, balancing both classes to 2,937 samples each.

Models: The experiment will evaluate Multinomial Naive Bayes, Logistic Regression, Random Forest Classifier, and Linear Support Vector Classifier (LinearSVC). These models cover probabilistic, linear, and ensemble approaches, providing a robust performance comparison.

Evaluation Metrics: The models will be evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrices. These metrics, especially F1-score and the confusion matrix, are critical for spam detection to understand the trade-off between false positives (legitimate messages marked as spam) and false negatives (spam reaching the inbox).
