import tiktoken
import mlx.core as mx
import mlx.nn as nn

# Initialize tokenizer
encoder = tiktoken.get_encoding("cl100k_base")

# Sample text
text = "Where there is a will, there is a way"

# Tokenize and get token IDs
token_ids = encoder.encode(text)

substrings = [encoder.decode([token]) for token in token_ids]

#The cl100k_base tokenizer has 100,256 tokens. Retrieve the string corresponding to token ID
string = encoder.decode([20000])

print("Text:", text)
print("Sub Strings:", substrings)
print("Token IDs:", token_ids)
print("Token string:", string)
