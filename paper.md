# Comparative Machine Learning for Spam Email Detection

**Author:** MD. Al-Mahmud  
**Course project:** Machine Learning  
**Task:** Binary classification of email messages as ham or spam

## Abstract

Spam email can expose users to unwanted advertising, scams, and malicious links. This project compares four supervised machine-learning classifiers—Multinomial Naive Bayes, Logistic Regression, Random Forest, and LinearSVC—for distinguishing spam from legitimate email. The notebook processes 5,171 labeled messages, applies text cleaning and TF-IDF vectorization, and evaluates models on a stratified 80/20 train-test split. The test set contains 1,035 messages. Logistic Regression and LinearSVC achieved the highest observed accuracy, 98.65%, followed by Random Forest at 98.36% and Multinomial Naive Bayes at 95.17%. Logistic Regression detected 296 of 300 spam messages, while LinearSVC detected 293. These results show strong performance on this dataset, but they do not establish general performance on other email sources. In addition, although SMOTE-balanced training data were created in the notebook, the classifiers were fitted using the original training matrix, so the reported scores do not measure the effect of SMOTE.

## 1. Introduction

Email remains an important communication channel, but unwanted messages can waste time and expose recipients to deceptive content. A spam filter can be framed as a binary text-classification problem: given an email's text, predict whether it is legitimate (ham) or spam. This project asks: **How do Multinomial Naive Bayes, Logistic Regression, Random Forest, and LinearSVC compare on the same labeled email dataset when text is represented with TF-IDF?**

Prior research on message filtering has examined both traditional machine-learning approaches and newer neural methods. Delany, Buckley, and Greene review SMS spam-filtering methods and emphasize the importance of representative, reliable datasets for evaluating filters [1]. Guzella and Caminhas provide a broader survey of spam-filtering approaches and describe the role of machine learning in filtering unwanted messages [2]. Research comparing classical and deep-learning approaches to short-message spam has also reported that performance depends on the dataset and method; for example, Zhang et al. investigate CNN and LSTM approaches for SMS spam classification [3]. These studies concern related message-filtering problems, especially SMS, and are background rather than direct evidence about the email dataset used here.

This project contributes a small, reproducible comparison of four classifiers using a common train-test split and TF-IDF features. The objective is to compare test accuracy and class-level errors, not to claim a novel spam-detection algorithm.

## 2. Methodology

### 2.1 Dataset

The notebook loads a CSV file named `dataset.csv` with email text and labels. After dropping the `Unnamed: 0` and `label_num` columns, the working data contain the `label` and `text` fields. The notebook reports 5,171 messages: 3,672 ham and 1,499 spam. No missing values were reported in the displayed initial check. The dataset source and collection details are not documented in the exported notebook, which limits reproducibility and assessment of representativeness.

### 2.2 Text preprocessing

The notebook lowercases text, removes punctuation, removes English stopwords, replaces URLs and email addresses with placeholder tokens, removes non-alphanumeric characters, and creates a lemmatized text column. The final feature variable used for the model experiments is `clean_text`, rather than the `lemmatized_text` column. Thus, lemmatization is performed but does not feed into the reported model inputs.

### 2.3 Train-test split and feature extraction

The data are split into 80% training and 20% test data using `train_test_split`, with `random_state=42` and stratification. `TfidfVectorizer` is fitted on the training text with `max_features=5000` and English stopword removal; the fitted vectorizer transforms the test text. This avoids fitting the vocabulary on the held-out test set.

### 2.4 Class imbalance and model training

The original dataset is imbalanced: 3,672 ham messages versus 1,499 spam messages. The notebook applies SMOTE to the sparse TF-IDF training features and prints a balanced training distribution of 2,937 examples per class. However, the four classifier cells subsequently call `.fit(X_train, y_train)`, not `.fit(X_train_balanced, y_train_balanced)`. Therefore, the reported model scores correspond to training on the original, imbalanced training data; SMOTE is not part of the fitted models in the reported results.

The evaluated classifiers are:
- **Multinomial Naive Bayes:** a probabilistic baseline commonly used for text classification.
- **Logistic Regression:** a linear classifier that estimates class probabilities.
- **Random Forest:** an ensemble of decision trees.
- **LinearSVC:** a linear support-vector classifier suited to high-dimensional feature spaces.

The notebook uses default model settings and does not report hyperparameter tuning or repeated cross-validation.

### 2.5 Evaluation

Evaluation uses held-out test accuracy, precision, recall, F1-score, and confusion matrices. In the confusion matrices, rows represent actual labels and columns represent predicted labels, ordered as ham then spam. Accuracy is the proportion of all test messages classified correctly. Spam recall is especially relevant because it measures the proportion of actual spam messages that the model catches.

## 3. Results

The held-out test set contains 1,035 messages: 735 ham and 300 spam. Table 1 summarizes the reported accuracy values from the notebook.

**Table 1. Test-set performance reported in the notebook**

| Model | Accuracy | Correct / total |
|---|---:|---:|
| Multinomial Naive Bayes | 95.17% | 985 / 1,035 |
| Logistic Regression | 98.65% | 1,021 / 1,035 |
| Random Forest | 98.36% | 1,018 / 1,035 |
| LinearSVC | 98.65% | 1,021 / 1,035 |

<img width="1393" height="810" alt="Model Accuracy Comparison" src="https://github.com/user-attachments/assets/4a6519bb-ee4f-40a4-bb59-b67407156309" />

**Figure 1.** Accuracy comparison for the four classifiers. Logistic Regression and LinearSVC tie for the highest observed accuracy.

Table 2 shows the confusion-matrix counts reproduced from the notebook.

**Table 2. Confusion matrices (rows = actual; columns = predicted)**

| Model | Actual ham → ham | Actual ham → spam | Actual spam → ham | Actual spam → spam |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 705 | 30 | 20 | 280 |
| Logistic Regression | 725 | 10 | 4 | 296 |
| Random Forest | 719 | 16 | 1 | 299 |
| LinearSVC | 728 | 7 | 7 | 293 |


<img width="444" height="393" alt="NB" src="https://github.com/user-attachments/assets/d973f0d1-1355-4193-9060-aa452ca06505" />
<img width="444" height="393" alt="LR" src="https://github.com/user-attachments/assets/1cb29d53-82f2-42b2-b37b-82c3a9c00485" />
<img width="444" height="393" alt="RF" src="https://github.com/user-attachments/assets/7600692a-1fbb-4b25-8f68-357e4ed8ad17" />
<img width="444" height="393" alt="SVM" src="https://github.com/user-attachments/assets/87b6f8e6-7099-40fc-9272-81c7172ffb44" />





**Figure 2.** Confusion matrices for the same held-out test set. False negatives are spam messages predicted as ham; false positives are legitimate messages predicted as spam.

Logistic Regression and LinearSVC achieved the top accuracy at 98.65%. Their error profiles differ: Logistic Regression produced 10 false positives and 4 false negatives, while LinearSVC produced 7 false positives and 7 false negatives. Random Forest had the fewest false negatives (1), but it produced 16 false positives. Multinomial Naive Bayes had the lowest accuracy and made 50 total errors (30 false positives and 20 false negatives).

## 4. Discussion

The results suggest that linear classifiers are strong choices for this TF-IDF representation on the present dataset. Logistic Regression and LinearSVC both achieved 98.65% accuracy, but the confusion matrices show that equal accuracy does not mean identical behavior. Logistic Regression missed fewer spam messages, whereas LinearSVC flagged fewer legitimate messages as spam. Which trade-off is preferable depends on the application: missed spam and incorrectly quarantined legitimate email have different costs.

Random Forest reached 98.36% accuracy and detected 299 of 300 spam messages, but it incorrectly flagged 16 legitimate messages. Multinomial Naive Bayes performed less strongly in this experiment, although it remains a useful baseline. These results should be interpreted as a comparison under the notebook's specific split and default model settings, not as a universal ranking of the algorithms.

Several limitations matter. First, the dataset source, collection period, and email-domain coverage are not recorded in the notebook. Second, only one random train-test split is reported, so variation across splits is unknown. Third, hyperparameter tuning and statistical uncertainty are not reported. Fourth, the notebook creates SMOTE-balanced data but does not use them when fitting the models; a separate, controlled experiment would be required to assess whether resampling helps. Fifth, preprocessing includes both stopword removal in the cleaning step and in TF-IDF, and the lemmatized column is not used as model input. Finally, a dataset-specific result may not transfer to modern email containing new spam tactics, HTML, attachments, multilingual content, or adversarial wording.

A stronger follow-up would use a documented dataset, a scikit-learn pipeline, cross-validation on the training data, explicit comparison of original versus resampled training, precision-recall analysis, and evaluation on a separate or newer email corpus. Any resampling should be applied only within training folds to avoid data leakage.

## 5. Conclusion

This project compared four machine-learning classifiers for email spam detection using cleaned text and TF-IDF features. Logistic Regression and LinearSVC achieved the highest test accuracy at 98.65%, with different false-positive and false-negative trade-offs; Random Forest achieved 98.36%, and Multinomial Naive Bayes achieved 95.17%. The results indicate that traditional linear classifiers can perform well on this dataset, but the findings are limited by undocumented dataset provenance, evaluation on a single split, and the fact that the generated SMOTE-balanced data were not used to train the reported models. Future work should validate the comparison with cross-validation, controlled resampling experiments, and an independent dataset.

## References

[1] Dada, E. G., Bassi, J. S., Chiroma, H., Abdulhamid, S. M., Adetunmbi, A. O., & Ajibuwa, O. E. (2019). Machine learning for email spam filtering: review, approaches and open research problems. Heliyon, 5, e01802. https://doi.org/10.1016/j.heliyon.2019.e01802

[2] Khurana, A., & Verma, O. P. (2023). Optimal Feature Selection for Imbalanced Text Classification. IEEE Transactions on Artificial Intelligence, 4, 135–147. https://doi.org/10.1109/tai.2022.3144651

[3] Prasad, P. (2024). Exploring Machine Learning Algorithms for Email Spam Filtering. SPAST Reports, 1. https://doi.org/10.69848/sreports.v1i6.5079
