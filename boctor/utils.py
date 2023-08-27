import nltk
import numpy as np
import json 

from nltk.stem.porter import PorterStemmer 

nltk.download('punkt')
stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

def stem(word):
    return nltk.PorterStemmer().stem(word.lower())


def bag_of_words(tokenized_sentence , all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

def dataloader(file):

    file_extension = file.split('.')[-1].lower()

    if file_extension == 'json':
        with open(file) as f:
            intents = json.load(f)
    else:
        raise Exception('File format not supported')

    all_words = []
    tags = []

    xy = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w,tag))


    ignore_words = ['?','!','.',',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words)) #set is used to remove duplicates
    tags=sorted(set(tags))


    X_train = []
    y_train = []

    for (pattern_sentence,tag) in xy:
        bag = bag_of_words(pattern_sentence,all_words)
        X_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train,y_train,all_words,tags