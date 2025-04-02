import os
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.text_embedding import TextEmbedder

text_embedder = TextEmbedder()

def process_ds1000_embeddings(file_path="data/ds1000.json", embedding_dir="data/ds1000_embeddings"):
    try:
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
                    if 'prompt' in item and 'reference_code' in item:
                        texts.append(item['prompt'])
                        codes.append(item['reference_code'])
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {idx}: {e}")
                    continue
        
        print(f"Extracted {len(texts)} text-code pairs")
        
        # 检查是否存在已保存的embeddings文件
        text_embedding_path = os.path.join(embedding_dir, "1000_text_embeddings.npy")
        code_embedding_path = os.path.join(embedding_dir, "1000_code_embeddings.npy")
        
        if os.path.exists(text_embedding_path) and os.path.exists(code_embedding_path):
            print("Loading existing embeddings...")
            text_embeddings = np.load(text_embedding_path)
            code_embeddings = np.load(code_embedding_path)
        else:
            print("Generating new embeddings...")
            # 加载模型并生成embeddings
            text_reference_dict = text_embedder.get_reference_dictionary(texts)
            text_embeddings = np.array(list(text_reference_dict.values()))
            with open(text_embedding_path, "wb") as f:
                np.save(f, text_embeddings)

            code_reference_dict = text_embedder.get_reference_dictionary(codes)
            code_embeddings = np.array(list(code_reference_dict.values()))
            with open(code_embedding_path, "wb") as f:
                np.save(f, code_embeddings)

        
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
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    process_ds1000_embeddings()