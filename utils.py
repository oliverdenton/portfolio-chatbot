import nltk
import torch
import numpy as np
from nltk.stem.porter import PorterStemmer
# nltk.download('punkt')

stemmer = PorterStemmer()

# Splits sentence into words or 'tokens'
def tokenize(sentence):
  return nltk.word_tokenize(sentence)


# Reduces a given word to its stem
def stem(word):
  return stemmer.stem(word.lower())


# Returns 1 at correct index if word in sentence is in 
# list of all known words, otherwise returns 0
def bag_of_words(tokens, all):
  tokens = [stem(w) for w in tokens]

  bag = np.zeros(len(all))
  for idx, w in enumerate(all):
    if w in tokens:
      bag[idx] = 1

  return bag