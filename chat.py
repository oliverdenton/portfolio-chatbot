import random
import json
import torch
from model import ChatbotClassifier
from utils import bag_of_words, tokenize

# Load model data and initialize
model_data = torch.load("model_data.pth")
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
model_state = model_data["model_state"]
net = ChatbotClassifier(input_size, hidden_size, output_size)
net.load_state_dict(model_state)
net.eval()

# Load intents/pattern data
intents_data = torch.load("intents_data.pth")
all_words = intents_data["all_words"]
tags = intents_data["tags"]

with open('intents.json', 'r') as data:
    intents = json.load(data)

# Chatbot demo
chatbot_name = "Virtual Oliver"
print(f"{chatbot_name}: Let's chat! What would you like to know? (type 'quit' to exit)")
while True:
    print("\n")
    sentence = input("You: ")
    if sentence.lower() == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = net(X.float())
    _, y_pred = torch.max(output, dim=1)
    tag = tags[y_pred.item()]

    # Check whether user has inputted something reasonable/relevant
    certainty = torch.softmax(output, dim=1)
    certainty = certainty[0][y_pred.item()]
    if certainty.item() > 0.7:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print("\n")
                print(f"{chatbot_name}: {random.choice(intent['responses'])}")
    else:
        print("\n")
        print(f"{chatbot_name}: I do not understand...")
        