import re, string, unicodedata
from sklearn.utils import shuffle
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import sys
import itertools
import numpy as np
from collections import Counter
import pickle
import pandas as pd



"""
Adapted from https://github.com/dennybritz/cnn-text-classification-tf
"""

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas


def word_checker(words):
    _words = []
    for word in words:
        wds = re.sub("/10$","",word)
        _words.append(wds)
    return _words



def normalize(words):
    words = nltk.word_tokenize(words)
    words = word_checker(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    words = " ".join(words)
    return words


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"<s>","\'s\'",string)
    string = re.sub(r"</s>","\'\'s\'\'",string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels():
    
    x_text, y = load_sentences_and_labels1()
    x_text = [s.split(" ") for s in x_text]
    _size = [len(zz) for zz in x_text]
    #y = [[0, 1] if label==1 else [1, 0] for label in y]
    y = [1 if label==1 else 0 for label in y]
    print("The is the average size ",np.mean(_size))
    return [x_text, y]



def load_sentences_and_labels1():
    with open("data/random_50.csv") as files:
        all_examples = list(files.readlines())
    text = [s.strip() for s in all_examples]
    x_text = []
    y = []
    _sizes = []
    for tx in text:
        tx = tx.split("\t")
        try:
            if float(tx[-1]) == 0 or float(tx[-1]) == 1:
                #print(tx[-1])
                print(tx[0],tx[1])
                x_text.append(clean_str(tx[0]))
                #print(clean_str(tx[0]),float(tx[-1]))
                y.append(float(tx[-1]))
        except Exception as ex:
            print(ex)
    #print(len(x_text),len(y))

    return x_text,y



def pad_sentences_1(sentences,padding_word="<PAD/>"):
    sequence_length = 80
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        if len(sentence)>80:
         new_sentence = sentence[:80]
        else:
         num_padding = sequence_length - len(sentence)
         new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
 



def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    #print(y[:10])
    #y = y.argmax(axis=1)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    #print(labels[:20])
    #print("Phase I ",len(sentences),len(labels))
    sentences_padded = pad_sentences_1(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    print(Counter(y))
    return [x, y, vocabulary, vocabulary_inv]


def loads_data():
    

    df_train = pd.read_csv("Train.csv",sep="\t")
    df_test = pd.read_csv("Test.csv",sep="\t")
    df_train = shuffle(df_train)
    df_test = shuffle(df_test)
    df_train['text']=df_train['text'].apply(normalize)
    df_test['text']=df_test['text'].apply(normalize)
    df_train.to_csv("data/pre_train.csv",sep="\t",header=True,index=False)
    df_test.to_csv("data/pre_test.csv",sep="\t",header=True,index=False)
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    for i in range(len(df_train)):
          X_train.append(df_train.iloc[i]['text'])
          Y_train.append(df_train.iloc[i]['label'])
    for i in range(len(df_test)):
          X_test.append(df_test.iloc[i]['text'])
          Y_test.append(df_test.iloc[i]['label'])
    sentences1 = [nltk.word_tokenize(words) for words in X_train]
    sentences2 = [nltk.word_tokenize(words) for words in X_test]
    sentences_1_padded = pad_sentences_1(sentences1)
    sentences_2_padded = pad_sentences_1(sentences2)
    sentences_1_padded.extend(sentences_2_padded)
    sentences_padded = sentences_1_padded
    print("Total Sentence Length ",len(sentences_padded))
    labels = np.concatenate((Y_train,Y_test),axis=0)
    vocabulary,vocabulary_inv = build_vocab(sentences_padded)
    embeddings_index = {}
    embed_shape = 0
    with open('wiki-news-300d-1M-subword.vec') as f:
      for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embed_shape = coefs.shape
        embeddings_index[word] = coefs
    embedding_matrix = np.zeros((len(vocabulary),300))
    for word,i in vocabulary.items():
      embed_vec = embeddings_index.get(word)
      if embed_vec is not None:
        #print(word,i,embed_vec)
        embedding_matrix[i] = embed_vec
      else:
         embedding_matrix[i] = np.random.uniform(-1, 1, size=embed_shape)

    _sentences_1_padded = pad_sentences_1(sentences1)
    _sentences_2_padded = pad_sentences_1(sentences2)
    x_train,y_train = build_input_data(_sentences_1_padded,Y_train,vocabulary)
    x_test,y_test  =   build_input_data(_sentences_2_padded,Y_test,vocabulary)
    #print(x.shape)
    #print(y.shape)
    print(len(vocabulary),len(vocabulary_inv))
    return [x_train,y_train,x_test,y_test,vocabulary,vocabulary_inv,embedding_matrix]


loads_data()
#x,y = load_sentences_and_labels1()
#print(x[20])
#print(y[20])
