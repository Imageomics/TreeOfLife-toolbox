This directory contains the support set class embeddings used for few-shot classification tasks.

Each `.pt` file is a PyTorch dictionary where the key is a class label (string), and the value is the corresponding class embedding (usually a PyTorch tensor).

The embeddings may be generated from either:
- Support set images
- Textual embeddings of the class label 

``` python
import torch

# Load the .pt file
embeddings_dict = torch.load("path/to/class_embeddings.pt")

# Check the available class labels
print("Class labels:", list(embeddings_dict.keys()))

# Access an embedding for a specific class
class_label = "landscape"
embedding = embeddings_dict[class_label]

print(f"Embedding for '{class_label}':", embedding)
print("Shape:", embedding.shape)
```
