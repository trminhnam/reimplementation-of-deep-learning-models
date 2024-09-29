import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from bigram import BigramLM
from utils import get_batch, estimate_loss


# Hyperparameters
batch_size = 4
batch_size = 128
block_size = 128
max_steps = 2000
train_test_ratio = 0.9
eval_interval = 200
eval_steps = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
with open("input.txt", "r") as file:
    data = file.read()

print(f"Loaded data with {len(data)} characters")
print(f'First 100 characters: "{data[:100]}"')
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"There are {vocab_size} unique characters in data:\n{chars}")

# Create mapping from characters to integers and vice versa
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

# Encode and decode functions (from characters to integers and vice versa)
encode = lambda x: [stoi[c] for c in x]
decode = lambda x: "".join([itos[i] for i in x])

# Convert data to integers
data = encode(data)
data = torch.tensor(data, dtype=torch.long)

# Train/Test split
train_data = data[: int(len(data) * train_test_ratio)]
test_data = data[int(len(data) * train_test_ratio) :]


x, y = get_batch(train_data, batch_size=2, block_size=4)
print(f"x: {x}")
print(f"y: {y}")

# Create model
model = BigramLM(vocab_size)

# Create optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

model = model.to(device)
model.train()

# Training loop
pbar = tqdm(range(max_steps), leave=True)
for i in pbar:
    x, y = get_batch(train_data, batch_size, block_size)
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    loss, logits = model(x, y)
    loss.backward()
    optimizer.step()

    pbar.set_description(f"Loss: {loss.item():.4f}")

    if i % eval_interval == 0:
        loss = estimate_loss(
            model, test_data, batch_size, block_size, eval_steps, device
        )
        print(f"Step {i}, Test loss: {loss:.4f}")


print(f"Final loss: {loss.item():.4f}")
generated_content = model.generate(
    torch.tensor([[1], [2]], device=device), max_new_tokens=100
).tolist()
print(f"Generated text: {decode(generated_content[0])}")
