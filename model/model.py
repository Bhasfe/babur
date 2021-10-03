from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from sklearn import decomposition, ensemble
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from google.oauth2 import service_account
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from google.cloud import bigquery
from sklearn import naive_bayes
from dotenv import load_dotenv
from datetime import date
from sklearn import svm
import pandas as pd
import numpy as np
import operator
import xgboost
import logging
import string
import pickle
import nltk
import json
import sys
import os
import re

MODEL_TEST_MODE = True

nltk.download('punkt')


def load_data() -> pd.DataFrame:
    """Get the data from BigQuery into a pandas DataFrame"""

    # Authenticate BigQuery
    client = bigquery.Client()

    query = """
            select 
                sentence, 
                label 
            from `project_id.dataset_id.table_id`
            where label is not null;
    """

    query_job = client.query(query)

    df = query_job.result().to_dataframe()

    logging.info(df.info())
    logging.info(f"\nClass Distribution: \n{df['label'].value_counts()}")

    return df


def preprocessing(msg: str) -> list:
    """Apply preprocessing on the data"""

    # Cast to lowercase
    msg = msg.lower()

    # Remove mentions and hashtag
    msg = re.sub(r'\@\w+|\#', '', msg)

    # Remove numbers
    msg = re.sub("^\d+\s|\s\d+\s|\s\d+$", '', msg)

    # Remove tables
    msg = re.sub("\w+.\w+.\w+\.\w+", '', msg)

    # Tokenize the words
    tokenized = word_tokenize(msg)

    # Remove non-alphabetic characters and keep the words contains three or more letters
    msg = " ".join(
        [token for token in tokenized if token not in string.punctuation])

    return msg


def get_tfidf(X_train: pd.DataFrame, X_test=None, ngram_range=(1,1), analyzer='word') -> pd.DataFrame:
    """Extract tf-idf features from the corpus"""

    # Initialize a Tf-idf Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, analyzer=analyzer)

    # Fit and transform the vectorizer
    if MODEL_TEST_MODE:
        tfidf_train = vectorizer.fit_transform(X_train)
        tfidf_test = vectorizer.transform(X_test)

        pickle.dump(vectorizer, open("tfidf-vectorizer.pkl", "wb"))

        return tfidf_train, tfidf_test

    else:
        tfidf = vectorizer.fit_transform(X_train)

        pickle.dump(vectorizer, open("tfidf-vectorizer.pkl", "wb"))

        return tfidf, None


def logistic_regression_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a logistic regression model"""

    lr = LogisticRegression()

    if MODEL_TEST_MODE:
        lr_model = lr.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

    else:
        lr.fit(X_train, y_train)
        pickle.dump(lr, open("model.pkl", "wb"))

        return 'Success'

    return acc

def naive_bayes_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a naive bayes model"""

    nb = naive_bayes.MultinomialNB()

    if MODEL_TEST_MODE:
        nb_model = nb.fit(X_train,y_train)
        y_pred = nb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
    else:
        nb.fit(X_train, y_train)
        pickle.dump(nb, open("model.pkl", "wb"))

        return 'Success'

    return acc

def random_forest_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a random forest model"""

    rf = ensemble.RandomForestClassifier()

    if MODEL_TEST_MODE:
        rf_model = rf.fit(X_train,y_train)
        y_pred = rf_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
    else:
        rf.fit(X_train, y_train)
        pickle.dump(rf, open("model.pkl", "wb"))

        return 'Success'

    return acc


def xgboost_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a xgboost model"""

    xgb = xgboost.XGBClassifier(eval_metric='mlogloss', use_label_encoder =False)
    
    if MODEL_TEST_MODE:
        xgb_model = xgb.fit(X_train,y_train)
        y_pred = xgb_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
    else:
        xgb.fit(X_train, y_train)
        pickle.dump(xgb, open("model.pkl", "wb"))

        return 'Success'

    return acc

def svm_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a svm model"""

    clf = svm.SVC()

    if MODEL_TEST_MODE:
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
    else:
        clf.fit(X_train, y_train)
        pickle.dump(clf, open("model.pkl", "wb"))
        return 'Success'

    return acc

def catboost_model(X_train= None, X_test = None, y_train= None, y_test= None) -> float:
    """Trains and evaluates a catboost model"""

    clf = CatBoostClassifier(learning_rate=0.1, depth=2, loss_function='MultiClass')
    cat_features = [0, 1, 2, 3]
    if MODEL_TEST_MODE:
        clf.fit(X_train, y_train, cat_features=cat_features)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
    else:
        clf.fit(X_train, y_train, cat_features=cat_features)
        pickle.dump(clf, open("model.pkl", "wb"))
        return 'Success'

    return acc


if __name__ == '__main__':
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO,
                        filename='../logs/model_logs/' + date.today().strftime("%d-%m-%Y") + '.log', 
                        filemode='a')


    load_dotenv()

    ########################  LOAD & PREPROCESS DATA #############################

    # Load data
    logging.info("Loading data...")
    df = load_data()

    # Preprocessing
    logging.info("Preprocessing...")

    le = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])

    reduced = df.drop("sentence", axis=1).drop_duplicates().sort_values("label_enc")
    logging.info(f"Encoded labels:\n{reduced}\n")
    reduced.to_csv('../data/encoded_labels.csv', index=False)

    # Get features and targets
    X = df['sentence']
    y = df['label_enc']


    ################################## TRAINING ################################

    logging.info("Training...")

    # Define models & accuracies dictionary
    scores = dict()

    # Get count vectors
    vectorizer = CountVectorizer()
    x_count = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(x_count, y, random_state=1, test_size=0.33, stratify=y)

    # Count Vectors + Logistic Regression
    scores["count_vectors_logistic_regression"] = logistic_regression_model(X_train, X_test, y_train, y_test)
    logging.info(f"Count Vectors - Logistic Regression Accuracy {scores['count_vectors_logistic_regression']}")   

    # Count Vectors + Naive Bayes
    scores["count_vectors_naive_bayes"] = naive_bayes_model(X_train, X_test, y_train, y_test)
    logging.info(f"Count Vectors - Naive Bayes Accuracy: {scores['count_vectors_naive_bayes']}")

    # Count Vectors + Random Forest
    scores["count_vectors_random_forest"] = random_forest_model(X_train, X_test, y_train, y_test)
    logging.info(f"Count Vectors - Random Forest Accuracy: {scores['count_vectors_random_forest']}")

    # Count Vectors + XGBoost
    scores["count_vectors_xgboost"] = xgboost_model(X_train, X_test, y_train, y_test)
    logging.info(f"Count Vectors - XGBoost Accuracy: {scores['count_vectors_xgboost']}")

    # Count Vectors + SVM model
    scores["count_vectors_svm"] = svm_model(X_train, X_test, y_train, y_test)
    logging.info(f"Count Vectors - SVM Accuracy: {scores['count_vectors_svm']}")

    # Word-Level Tfidf

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.33, stratify=y)

    tfidf_train, tfidf_test = get_tfidf(X_train, X_test)

    # Word-Level Tfidf + Logistic Regression
    scores["word_level_tfidf_logistic_regression"] = logistic_regression_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Word-level Tfidf - Logistic Regression Accuracy: {scores['word_level_tfidf_logistic_regression']}")

    # Word-Level Tfidf + Naive Bayes
    scores["word_level_tfidf_naive_bayes"] = naive_bayes_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Word-level Tfidf - Naive Bayes Accuracy: {scores['word_level_tfidf_naive_bayes']}")

    # Word-Level Tfidf + Random Forest
    scores["word_level_tfidf_random_forest"] = random_forest_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Word-level Tfidf - Random Forest Accuracy: {scores['word_level_tfidf_random_forest']}")
 
    # Word-Level Tfidf + XGBoost
    scores["word_level_tfidf_xgboost"] = xgboost_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Word-level Tfidf - XGBoost Accuracy: {scores['word_level_tfidf_xgboost']}")

    # Word-Level Tfidf + SVM
    scores["word_level_tfidf_svm"] = svm_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Word-level Tfidf - SVM Accuracy: {scores['word_level_tfidf_svm']}")

    # Ngram-Level Tfidf 
    tfidf_train, tfidf_test = get_tfidf(X_train, X_test, ngram_range = (2,3))

    # Ngram-Level Tfidf + Logistic Regression
    scores["ngram_level_tfidf_logistic_regression"] = logistic_regression_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Ngram-level Tfidf - Logistic Regression Accuracy: {scores['ngram_level_tfidf_logistic_regression']}")
    
    # Ngram-Level Tfidf + Naive Bayes
    scores["ngram_level_tfidf_naive_bayes"] = naive_bayes_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Ngram-level Tfidf - Naive Bayes Accuracy: {scores['ngram_level_tfidf_naive_bayes']}")
        
    # Ngram-Level Tfidf + Random Forest
    scores["ngram_level_tfidf_random_forest"] = random_forest_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Ngram-level Tfidf - Random Forest Accuracy: {scores['ngram_level_tfidf_random_forest']}")
        
    # Ngram-Level Tfidf + XGBoost
    scores["ngram_level_tfidf_xgboost"] = xgboost_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Ngram-level Tfidf - XGBoost Accuracy: {scores['ngram_level_tfidf_xgboost']}")
        
    # Ngram-Level Tfidf + SVM
    scores["ngram_level_tfidf_svm"] = svm_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Ngram-level Tfidf - SVM Accuracy: {scores['ngram_level_tfidf_svm']}")

    # Character-level Tfidf 

    tfidf_train, tfidf_test = get_tfidf(X_train, X_test, ngram_range = (2,3), analyzer = "char")
    
    # Character-level Tfidf + Logistic Regression
    scores["char_level_tfidf_logistic_regression"] = logistic_regression_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Character-level Tfidf - Logistic Regression Accuracy: {scores['char_level_tfidf_logistic_regression']}")
        
    # Character-level Tfidf + Naive Bayes
    scores["char_level_tfidf_naive_bayes"] = naive_bayes_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Character-level Tfidf - Naive Bayes Accuracy: {scores['char_level_tfidf_naive_bayes']}")
        
    # Character-level Tfidf + Random Forest
    scores["char_level_tfidf_random_forest"] = random_forest_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Character-level Tfidf - Random Forest Accuracy: {scores['char_level_tfidf_random_forest']}")
        
    # Character-level Tfidf + XGBoost
    scores["char_level_tfidf_xgboost"] = xgboost_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Character-level Tfidf - XGBoost Accuracy: {scores['char_level_tfidf_xgboost']}")
        
    # Character-level Tfidf + SVM
    scores["char_level_tfidf_svm"] = svm_model(tfidf_train, tfidf_test, y_train, y_test)
    logging.info(f"Character-level Tfidf - SVM Accuracy: {scores['char_level_tfidf_svm']}")

    ############################################################################

    MODEL_TEST_MODE = False

    # Get the most successful model
    best_model = max(scores.items(), key=operator.itemgetter(1))[0]

    with open("../data/best_model.txt","w") as f:
        f.write(best_model)

    logging.info("################################################################# \n")
    logging.info(f"Best Model: {best_model}\n")

    # Train the most successful model with all data
    logging.info("Training with all data")
        
    if 'char_level_tfidf_' in best_model:
        tfidf, _ = get_tfidf(X, ngram_range = (2,3), analyzer = "char")
        locals()[best_model.replace("char_level_tfidf_", "") + "_model"](X_train=tfidf, y_train=y)
    elif 'ngram_level_tfidf_' in best_model:
        tfidf, _ = get_tfidf(X, ngram_range = (2,3), analyzer = "char")
        locals()[best_model.replace("ngram_level_tfidf_", "") + "_model"](X_train=tfidf, y_train=y)
    elif 'word_level_tfidf_' in best_model:
        tfidf, _ = get_tfidf(X)
        locals()[best_model.replace("word_level_tfidf_", "") + "_model"](X_train=tfidf, y_train=y)
    elif 'count_vectors_' in best_model:
        vectorizer = CountVectorizer()
        pickle.dump(vectorizer, open("count-vectorizer.pkl", "wb"))
        count_vectors = vectorizer.fit_transform(X)
        locals()[best_model.replace("count_vectors_", "") + "_model"](X_train=count_vectors, y_train=y)

    logging.info("\n###########################  FINISHED ########################### ")
