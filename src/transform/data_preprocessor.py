import pandas as pd
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk.tokenize import word_tokenize 
import re
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from wordcloud import WordCloud



# Set folder path for JSON input and preprocessed_data.csv output
_JSON_INPUT_FOLDER = ".\\TWFYPreprocessor\\json\\"
_PREPROCESSED_DATA_OUTPUT = ".\\SAModel\\data\\preprocessed_data.csv"
_RESULTS_FOLDER = ".\\SAModel\\results\\"

# Check whether the input and output folders exist
os.makedirs(_JSON_INPUT_FOLDER, exist_ok=True)
os.makedirs(".\\SAModel\\data\\", exist_ok=True)
os.makedirs(_RESULTS_FOLDER, exist_ok=True)

### Preprocessing

def populate_dataframe():
    data = pd.DataFrame()
    for file in os.listdir(_JSON_INPUT_FOLDER):
        if file.endswith(".json"):
            print(file)
            temp_data = pd.read_json(os.path.join(_JSON_INPUT_FOLDER, file))
            data = pd.concat([data, temp_data], ignore_index = True)
    print("\nDataframe populated...\n")
    print(data)
    data.replace({"aye" : 1, "no" : 0}, inplace = True)
    edited_speech = data["text"].copy()
    data["speech_without_stopwords"] = edited_speech
    return data

# with open(_JSON_TEST_FILE, "r") as file:
#     data = pd.read_json(file.read())

# Preprocess text data for sentiment analysis
def preprocess_speech_data(data, column):
    print("\nPreprocessing data...")
    # Convert text to lowercase
    data[column] = data[column].astype(str).str.lower()
    # Remove "{'p': ' ... '}" from each sentence
    data[column] = data[column].apply(lambda x:re.sub(r'^.{5}|.{2}$', "", x))
    # Remove "x__" from each sentence
    data[column] = data[column].apply(lambda x:re.sub(r'\bx\w+\b', "", x))
    # Remove any links
    data[column] = data[column].apply(lambda x:re.sub(r"http\S+", "", x))
    # Remove all special characters
    data[column] = data[column].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Replace multiple spaces with single spaces
    data[column] = data[column].apply(lambda x:re.sub(r'\s+', " ", x, flags=re.I))
    # Remove all single characters in the text
    data[column] = data[column].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', "", x))  
    
def remove_stopwords_tokenize(data, column):
    print("Removing stopwords and tokenizing data...")
    nltk.download("stopwords", quiet = True)
    nltk.download("punkt", quiet = True)
    def getting_filter(paragraph):
        example_paragraph = paragraph
        filtered_paragraph = [] 
        stop_words = set(stopwords.words("english")) 
        word_tokens_filter = word_tokenize(example_paragraph) 
        filtered_paragraph = [w for w in word_tokens_filter if not w in stop_words] 
        return filtered_paragraph
    # Append filtered paragraph to data
    x = []
    for i in data[column].values:
        x.append(getting_filter(i))
    data[column] = x

lemmatizer = WordNetLemmatizer()
def lemmatize_data(data, column):
    print("Lemmatizing data...")
    def getting_lemmatize(paragraph):
        nltk.download("wordnet", quiet = True)
        example_paragraph = paragraph
        output_paragraph = []
        word_tokens_lemmatize = word_tokenize(example_paragraph)
        output_paragraph = [lemmatizer.lemmatize(w) for w in word_tokens_lemmatize]
        # Remove characters where length < 2
        without_single_char = [word for word in output_paragraph if len(word) > 2]
        # Remove numbers
        cleaned_data = [word for word in without_single_char if not word.isnumeric()]
        return cleaned_data
    # Append lemmatized paragraph to data
    x = []
    for i in data[column].values:
        x.append(getting_lemmatize(i))
    data[column] = x

def join_paragraph(data, column):
    print("Joining text into paragraphs...")
    data[column] = data[column].apply(lambda x:" ".join([i + " " for i in x]))
    # Remove any double spaces
    data[column] = data[column].apply(lambda x:re.sub(r'\s+', " ", x, flags = re.I))
    
def generate_wordcloud(data, filename):
    text = " ".join(data.tolist())
    wordcloud = WordCloud(width = 800, height = 400, background_color = "white", repeat = False, colormap = "tab20c").generate(text)
    plt.figure(figsize = (12, 10))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(_RESULTS_FOLDER + filename)

# Function calls to preprocess the data
data = populate_dataframe()
preprocess_speech_data(data, "speech_without_stopwords")
generate_wordcloud(data["speech_without_stopwords"], "wordcloud_before_preprocessing")
remove_stopwords_tokenize(data, "speech_without_stopwords")
join_paragraph(data, "speech_without_stopwords")

final_data = data["speech_without_stopwords"].copy()
data["post_lemmatization"] = final_data

lemmatize_data(data, "post_lemmatization")
join_paragraph(data, "post_lemmatization")
generate_wordcloud(data["post_lemmatization"], "wordcloud_after_preprocessing")

# Append POS (part of speech) tag to individual words
def pos_tagging():
    print("Appending Part of Speech tags to words...")
    pos = neg = obj = count = 0
    pos_tagging = []
    nltk.download("averaged_perceptron_tagger", quiet = True)
    for speech in data["post_lemmatization"]:
        list = word_tokenize(speech)
        pos_tagging.append(nltk.pos_tag(list))
    data["pos_tags"] = pos_tagging

# Wordnet lookup
def tags_to_wordnet(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    elif tag.startswith("V"):
        return wordnet.VERB
    return None

# Function that returns list of pos-neg and objective score
# Returns empty list if not present in SentiWordNet
def get_sentiment(word, tag):
    nltk.download("sentiwordnet", quiet = True)
    wordnet_tag = tags_to_wordnet(tag)
    if wordnet_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
        return []
    # Lemmatize words
    lemma = lemmatizer.lemmatize(word, pos = wordnet_tag)
    if not lemma:
        return []
    # Look up word using Synsets (synonymous words), part of WordNet
    synsets = wordnet.synsets(word, pos = wordnet_tag)
    if not synsets:
        return []

    synset = synsets[0]
    sentiwordnet_synset = sentiwordnet.senti_synset(synset.name())
    return [synset.name(), sentiwordnet_synset.pos_score(), sentiwordnet_synset.neg_score(), sentiwordnet_synset.obj_score()]

# Calculate sentiment score
def calculate_sentiment():
    print("Calculating sentiment scores...")
    pos = 0    
    neg = 0
    sentiment_score = []
    for pos_value in data["pos_tags"]:
        sentiment_value = [get_sentiment(x,y) for (x,y) in pos_value]
        for score in sentiment_value:
            try:
                pos = pos + score[1]
                neg = neg + score[2]
            except:
                continue
        sentiment_score.append(pos - neg)
        pos = 0
        neg = 0   
    data["sentiment_score"] = sentiment_score

    overall_sentiment = []
    overall_label = []
    
    for i in range(len(data)):
        if data["sentiment_score"][i] >= 0.05:
            overall_sentiment.append("Positive")
            overall_label.append(1)
        elif data["sentiment_score"][i] <= -0.05:
            overall_sentiment.append("Negative")
            overall_label.append(0)
        else:
            overall_sentiment.append("Neutral")
            overall_label.append(1)
    data["overall_sentiment"] = overall_sentiment
    data["overall_label"] = overall_label

# Functions to append POS tags and calculate sentiment scores
pos_tagging()
calculate_sentiment()

# Confusion Matrix for MP votes against overall sentiment
cm = confusion_matrix(data["vote"], data["overall_label"])
ConfusionMatrixDisplay(cm).plot()
plt.savefig(_RESULTS_FOLDER + "votes_against_overall_sentiment_cm")

# Write Dataframe to CSV file
data["speech_text_new"] = data["post_lemmatization"].copy()
data.to_csv(_PREPROCESSED_DATA_OUTPUT, encoding = "utf-8")