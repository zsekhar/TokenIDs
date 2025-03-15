import tiktoken
import torch
import torch.nn as nn

# Initialize tokenizer
encoder = tiktoken.get_encoding("cl100k_base")

# Sample text
text = "Hello"

# Tokenize and get token IDs
token_ids = encoder.encode(text)
substrings = [encoder.decode([tid]) for tid in token_ids]

#The cl100k_base tokenizer has 100,256 tokens. Retrieve the string corresponding to token ID
string = encoder.decode([20000])

print("Text", text)
print("Substrings:", substrings)
print("Token IDs:", token_ids)
print("Token string:", string)
