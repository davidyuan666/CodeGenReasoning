import json
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm

# Load pre-trained model for embeddings
def load_model(model_name="microsoft/codebert-base"):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

# Generate embeddings for a list of texts
def generate_embeddings(texts, tokenizer, model, max_length=512, batch_size=8):
    print("Generating embeddings...")
    embeddings = []
    
    # Process in batches to avoid memory issues
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                          max_length=max_length, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Use the CLS token embedding as the text representation
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# Visualize embeddings in 2D
def visualize_embeddings(text_embeddings, code_embeddings, n_samples=100, save_path="embeddings_visualization.png"):
    print("Visualizing embeddings...")
    
    # Select random samples if there are more than n_samples
    if len(text_embeddings) > n_samples:
        indices = random.sample(range(len(text_embeddings)), n_samples)
        text_embeddings_sample = text_embeddings[indices]
        code_embeddings_sample = code_embeddings[indices]
    else:
        text_embeddings_sample = text_embeddings
        code_embeddings_sample = code_embeddings
    
    # Combine embeddings for t-SNE
    combined_embeddings = np.vstack([text_embeddings_sample, code_embeddings_sample])
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # Split back into text and code embeddings
    text_embeddings_2d = embeddings_2d[:len(text_embeddings_sample)]
    code_embeddings_2d = embeddings_2d[len(text_embeddings_sample):]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                c='blue', marker='o', alpha=0.7, label='Text')
    plt.scatter(code_embeddings_2d[:, 0], code_embeddings_2d[:, 1], 
                c='red', marker='x', alpha=0.7, label='Code')
    
    # Add labels and legend
    plt.title('Embedding Distribution of Text and Code', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Show the plot
    plt.show()

# Main function to process the DS1000 dataset
def process_ds1000(file_path="data/ds1000.json", n_samples=100):
    print(f"Reading data from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    # Load the JSON data
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    print(f"Loaded {len(data)} entries from dataset")
    
    # Extract text and code from the data
    texts = []
    codes = []
    
    for item in data:
        # Adjust these fields according to the actual structure of your JSON file
        if 'problem' in item and 'solution' in item:
            texts.append(item['problem'])
            codes.append(item['solution'])
    
    print(f"Extracted {len(texts)} text-code pairs")
    
    # If we have too few samples, adjust n_samples
    n_samples = min(n_samples, len(texts))
    
    # Load model and generate embeddings
    tokenizer, model = load_model()
    text_embeddings = generate_embeddings(texts, tokenizer, model)
    code_embeddings = generate_embeddings(codes, tokenizer, model)
    
    # Visualize the embeddings
    visualize_embeddings(text_embeddings, code_embeddings, n_samples)
    
    # Also create a visualization with UMAP for comparison
    try:
        import umap
        print("Generating UMAP visualization...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        
        # Apply UMAP to a subset of embeddings
        indices = random.sample(range(len(text_embeddings)), n_samples) if len(text_embeddings) > n_samples else range(len(text_embeddings))
        combined_embeddings = np.vstack([text_embeddings[indices], code_embeddings[indices]])
        embeddings_2d = reducer.fit_transform(combined_embeddings)
        
        # Split back into text and code embeddings
        text_embeddings_2d = embeddings_2d[:n_samples]
        code_embeddings_2d = embeddings_2d[n_samples:]
        
        # Create plot
        plt.figure(figsize=(12, 10))
        plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                    c='blue', marker='o', alpha=0.7, label='Text')
        plt.scatter(code_embeddings_2d[:, 0], code_embeddings_2d[:, 1], 
                    c='red', marker='x', alpha=0.7, label='Code')
        
        plt.title('UMAP Embedding Distribution of Text and Code', fontsize=16)
        plt.xlabel('Dimension 1', fontsize=12)
        plt.ylabel('Dimension 2', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig("umap_embeddings_visualization.png", dpi=300, bbox_inches='tight')
        print("UMAP visualization saved to umap_embeddings_visualization.png")
        
        plt.show()
    except ImportError:
        print("UMAP not installed. Skipping UMAP visualization.")

if __name__ == "__main__":
    # Process the DS1000 dataset
    process_ds1000()