# install nltk
# install numpy
# install tflearn
# install tensorflowx

import json
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

with open("vartalap.json") as file:
    data = json.load(file)

word = []
label = []
docs_x = []
docs_y = []

for vartal in data["vartalap"]:
    for pattern in vartal["patterns"]:

        wrds = nltk.word_tokenize(pattern)
        # tokenizing words means to basically separate every word to be more specific
        # this will return a list with all of the words in it

        word.extend(wrds)
        docs_x.append(pattern)
        # each entry in x corresponds to that in y
        # i.e for every pattern in x, a value will be stored in y that tells what tag it is a part of

        docs_y.append(vartal["tag"])

    if vartal["tag"] not in label:
        label.append(vartal["tag"])

# Stemming through the json file, specifically patterns
# stemming brings a word or sentence back to root word, i.e
# if the sentence is 'how are you?' stemming removes the '?' to
# make the word more readable and easy to understand

# stemming here makes us go through the list, while deleting the duplicate words
# thus figuring out how many words has the system read already
word = [stemmer.stem(w.lower()) for w in word]
word = sorted(list(set(word)))

label = sorted(label)

training = []
output = []

out_empty = [0 for _ in range(len(label))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc if w != "?"]

    for w in word:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[label.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)
