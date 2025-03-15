# Tokenization and PyTorch Model

## Overview
This project demonstrates how to tokenize text using the `tiktoken` library and retrieve token IDs along with their corresponding substrings. Additionally, PyTorch is used to set up a simple neural network model.

## Requirements
Ensure you have the following dependencies installed before running the script:

```sh
pip install tiktoken torch
```

## Usage
1. **Initialize Tokenizer:** The script initializes the `tiktoken` tokenizer using the `cl100k_base` encoding.
2. **Tokenization:** Given an input text (e.g., "Hello"), the script tokenizes it into smaller substrings and retrieves their corresponding token IDs.
3. **Output Display:** The tokenized substrings and their respective token IDs are printed.

## Code Explanation
```python
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

print("Text:", text)
print("Substrings:", substrings)
print("Token IDs:", token_ids)
```

## Expected Output
When executed, the script outputs:
```
Text: Hello
Substrings: ['Hello']
Token IDs: [9906]
```

## License
This project is licensed under the MIT License.

