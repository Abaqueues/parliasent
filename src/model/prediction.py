import pandas as pd
import os
from joblib import dump, load
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



# Set folder path for preprocessed_data.csv input and predicted_data.csv output
_PREPROCESSED_DATA_INPUT = ".\\SAModel\\data\\preprocessed_data.csv"
_PREDICTED_DATA_OUTPUT = ".\\SAModel\\data\\predicted_data.csv"
_RESULTS_FOLDER = ".\\SAModel\\results\\"
_MODELS_FOLDER = ".\\SAModel\\models\\"

# Check whether the input and output folders exist
os.makedirs(".\\SAModel\\data\\", exist_ok=True)
os.makedirs(_RESULTS_FOLDER, exist_ok=True)
os.makedirs(_MODELS_FOLDER, exist_ok=True)

# Read the preprocessed_data.csv file to create a data object
data = pd.read_csv(_PREPROCESSED_DATA_INPUT, usecols={"id":str, "speakername":str, "url":str, "vote":str, "date":str, "speech_text_new":str, "sentiment_score":float, "overall_sentiment":str})
print(data)

# Transform data into a 'document-term matrix'
# from sklearn.feature_extraction.text import CountVectorizer
# print("Transforming training data into a document-term matrix...")
# vect_doc_term = CountVectorizer()
# vect_doc_term.fit(data["speech_text_new"].values.astype(str))
# vect_doc_term.get_feature_names_out()
# simple_train_dtm = vect_doc_term.transform(data["speech_text_new"].values.astype(str))

# Apply BoW Model to data
from sklearn.feature_extraction.text import CountVectorizer
print("\nApplying BoW model...")
bow_counts = CountVectorizer(tokenizer = word_tokenize, ngram_range=(1, 3))
bow_data = bow_counts.fit_transform(data["speech_text_new"].values.astype(str))

# # Apply TF-IDF model to data
# print("Applying TF-IDF model...")
# from sklearn.feature_extraction.text import TfidfVectorizer
# noise_words = []
# tfidf_counts = TfidfVectorizer(tokenizer = word_tokenize, stop_words = noise_words, ngram_range=(1,1))
# tfidf_data = tfidf_counts.fit_transform(data["speech_text_new"].values.astype(str))

# Load pickled model
nb_model = load(_MODELS_FOLDER + "nb_model.joblib")
lr_model = load(_MODELS_FOLDER + "lr_model.joblib")
sgd_model = load(_MODELS_FOLDER + "sgd_model.joblib")

# # Predict and KFold Cross Validation with TF-IDF MultinomialNB model
# print("\nPredicting with MultinomialNB model...")
# pred_nb_all = nb_model.predict(tfidf_data)
# data["nb_prediction"] = pred_nb_all
# scores = cross_val_score(nb_model, tfidf_data, data["overall_sentiment"], cv=5, n_jobs = -1)
# print("\nMultinomialNB Cross Validation Results:")
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# cm = confusion_matrix(data["overall_sentiment"], data["nb_prediction"])
# ConfusionMatrixDisplay(cm).plot()
# plt.savefig(_RESULTS_FOLDER + "nb_prediction_cm")

# # Predict and KFold Cross Validation with TF-IDF LogisticRegression model
# print("\nPredicting with LogisticRegression model...")
# pred_lr_all = lr_model.predict(tfidf_data)
# data["lr_prediction"] = pred_lr_all
# scores = cross_val_score(lr_model, tfidf_data, data["overall_sentiment"], cv=5, n_jobs = -1)
# print("\nLinearRegression Cross Validation Results:")
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# cm = confusion_matrix(data["overall_sentiment"], data["lr_prediction"])
# ConfusionMatrixDisplay(cm).plot()
# plt.savefig(_RESULTS_FOLDER + "lr_prediction_cm")

# # Predict and KFold Cross Validation with TF-IDF SGDClassifier model
# print("\nPredicting with SGDClassifier model...")
# pred_sgd_all = sgd_model.predict(tfidf_data)
# data["sgd_prediction"] = pred_sgd_all
# scores = cross_val_score(sgd_model, tfidf_data, data["overall_sentiment"], cv=5, n_jobs = -1)
# print("\nSGDClassifier Cross Validation Results:")
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# cm = confusion_matrix(data["overall_sentiment"], data["sgd_prediction"])
# ConfusionMatrixDisplay(cm).plot()
# plt.savefig(_RESULTS_FOLDER + "sgd_prediction_cm")

# Predict and KFold Cross Validation with BoW MultinomialNB model
print("\nPredicting with MultinomialNB model...")
pred_nb_all = nb_model.predict(bow_data)
data["nb_prediction"] = pred_nb_all
scores = cross_val_score(nb_model, bow_data, data["overall_sentiment"], cv=5, n_jobs = 1)
print("\nMultinomialNB Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
cm = confusion_matrix(data["overall_sentiment"], data["nb_prediction"])
ConfusionMatrixDisplay(cm).plot()
plt.savefig(_RESULTS_FOLDER + "nb_prediction_cm")

# Predict and KFold Cross Validation with BoW LogisticRegression model
print("\nPredicting with LogisticRegression model...")
pred_lr_all = lr_model.predict(bow_data)
data["lr_prediction"] = pred_lr_all
scores = cross_val_score(lr_model, bow_data, data["overall_sentiment"], cv=5, n_jobs = 1)
print("\nLinearRegression Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
cm = confusion_matrix(data["overall_sentiment"], data["lr_prediction"])
ConfusionMatrixDisplay(cm).plot()
plt.savefig(_RESULTS_FOLDER + "lr_prediction_cm")

# Predict and KFold Cross Validation with BoW SGDClassifier model
print("\nPredicting with SGDClassifier model...")
pred_sgd_all = sgd_model.predict(bow_data)
data["sgd_prediction"] = pred_sgd_all
scores = cross_val_score(sgd_model, bow_data, data["overall_sentiment"], cv=5, n_jobs = 1)
print("\nSGDClassifier Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
cm = confusion_matrix(data["overall_sentiment"], data["sgd_prediction"])
ConfusionMatrixDisplay(cm).plot()
plt.savefig(_RESULTS_FOLDER + "sgd_prediction_cm")

print(data)

# Write predicted data to predicted_data.csv
data.to_csv(_PREDICTED_DATA_OUTPUT, encoding = "utf-8")