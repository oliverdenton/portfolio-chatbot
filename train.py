import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset import ChatbotData
from model import ChatbotClassifier

# Load dataset
dataset = ChatbotData()
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Hyper-parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
HIDDEN = 500
INPUT = 137             #no. of patterns
OUTPUT = 8              #no. of classes

# Initialize model
net = ChatbotClassifier(INPUT, HIDDEN, OUTPUT)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in tqdm(range(NUM_EPOCHS)):
    for(words, labels) in train_dataloader:
        output = net(words.float())
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print(f"=Final loss: {loss.item()}")

# Save data to be loaded in demo
save_data = {
    "model_state": net.state_dict(),
    "input_size": INPUT,
    "hidden_size": HIDDEN,
    "output_size": OUTPUT
}
torch.save(save_data, "model_data.pth")