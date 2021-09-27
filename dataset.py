import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import bag_of_words, tokenize, stem

# Pre-processesing ready for training/dataset creation
def create_training_data():
    with open("intents.json") as f:
        intents = json.load(f)

    all_words = []
    tags = []
    xy = []

    for intent in intents["intents"]:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_chars = ["!", "?", ",", "."]
    all_words = [stem(w) for w in all_words if w not in ignore_chars]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    X = []
    y = []
    for(pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X.append(bag)
        label = tags.index(tag)
        y.append(label)

    # Save data to be loaded in demo
    save_data = {
        "all_words": all_words,
        "tags": tags
    }
    torch.save(save_data, "intents_data.pth")

    return np.array(X), np.array(y)

# Create dataset
class ChatbotData(Dataset):
    def __init__(self):
        X_train, y_train = create_training_data()
        self.num_samples = len(X_train)
        self.X = X_train
        self.y = y_train

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.num_samples
