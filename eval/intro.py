import os
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.text_embedding import TextEmbedding

def process_ds1000_embeddings(file_path="data/ds1000.json", embedding_dir="data/embedding"):
    print(f"Reading data from {file_path}...")
    
    # 创建embedding保存目录
    os.makedirs(embedding_dir, exist_ok=True)
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    # 逐行读取JSON数据
    texts = []
    codes = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                item = json.loads(line)
                if 'problem' in item and 'solution' in item:
                    texts.append(item['problem'])
                    codes.append(item['solution'])
            except json.JSONDecodeError as e:
                print(f"Error parsing line {idx}: {e}")
                continue
    
    print(f"Extracted {len(texts)} text-code pairs")
    
    # 加载模型并生成embeddings
    tokenizer, model = load_model()
    text_embeddings = generate_embeddings(texts, tokenizer, model)
    code_embeddings = generate_embeddings(codes, tokenizer, model)
    
    # 保存每个样本的embeddings
    for idx in range(len(text_embeddings)):
        embedding_data = {
            'text_embedding': text_embeddings[idx].tolist(),
            'code_embedding': code_embeddings[idx].tolist()
        }
        output_path = os.path.join(embedding_dir, f"embedding_{idx}.json")
        with open(output_path, 'w') as f:
            json.dump(embedding_data, f)
    
    # 使用t-SNE进行降维可视化
    print("Generating t-SNE visualization...")
    combined_embeddings = np.vstack([text_embeddings, code_embeddings])
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # 分离文本和代码的embeddings
    text_embeddings_2d = embeddings_2d[:len(texts)]
    code_embeddings_2d = embeddings_2d[len(texts):]
    
    # 创建散点图
    plt.figure(figsize=(12, 10))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
                c='blue', marker='o', alpha=0.5, label='Prompt')
    plt.scatter(code_embeddings_2d[:, 0], code_embeddings_2d[:, 1], 
                c='red', marker='x', alpha=0.5, label='Reference Code')
    
    plt.title('Distribution of Prompt and Reference Code Embeddings', fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("embeddings_distribution.png", dpi=300, bbox_inches='tight')
    print("Visualization saved to embeddings_distribution.png")
    plt.show()

if __name__ == "__main__":
    process_ds1000_embeddings()