import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch
from transformers import AutoTokenizer, AutoModel
import os
from tqdm import tqdm
import random

# 加载预训练模型用于生成embeddings
def load_model(model_name="microsoft/codebert-base"):
    print(f"正在加载模型 {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return tokenizer, model, device

# 为文本生成embeddings
def generate_embeddings(texts, tokenizer, model, device, max_length=512, batch_size=8):
    print("正在生成embeddings...")
    embeddings = []
    
    # 分批处理以避免内存问题
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, 
                         max_length=max_length, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 使用CLS token的embedding作为文本表示
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# 将embeddings可视化为2D图
def visualize_embeddings(text_embeddings, code_embeddings, method='tsne', n_samples=100, save_path=None):
    print(f"使用{method}可视化embeddings...")
    
    # 如果样本数大于n_samples，随机选择n_samples个样本
    if len(text_embeddings) > n_samples:
        indices = random.sample(range(len(text_embeddings)), n_samples)
        text_embeddings_sample = text_embeddings[indices]
        code_embeddings_sample = code_embeddings[indices]
    else:
        text_embeddings_sample = text_embeddings
        code_embeddings_sample = code_embeddings
    
    # 合并embeddings用于降维
    combined_embeddings = np.vstack([text_embeddings_sample, code_embeddings_sample])
    
    # 应用降维算法
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embeddings)-1))
        title = 't-SNE Visualization of Text and Code Embeddings'
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        title = 'UMAP Visualization of Text and Code Embeddings'
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    embeddings_2d = reducer.fit_transform(combined_embeddings)
    
    # 分离回文本和代码embeddings
    text_embeddings_2d = embeddings_2d[:len(text_embeddings_sample)]
    code_embeddings_2d = embeddings_2d[len(text_embeddings_sample):]
    
    # 创建图像
    plt.figure(figsize=(12, 10))
    plt.scatter(text_embeddings_2d[:, 0], text_embeddings_2d[:, 1], 
               c='blue', marker='o', alpha=0.7, label='Text (Prompt)')
    plt.scatter(code_embeddings_2d[:, 0], code_embeddings_2d[:, 1], 
               c='red', marker='x', alpha=0.7, label='Code (Reference)')
    
    # 添加标签和图例
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图像
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化已保存到 {save_path}")
    
    # 显示图像
    plt.show()

# 处理DS1000数据集
def process_ds1000(file_path="data/ds1000.json", n_samples=100):
    print(f"从 {file_path} 读取数据...")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 未找到!")
        return
    
    # 读取JSON数据，按行解析
    texts = []
    codes = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if 'prompt' in item and 'reference_code' in item:
                        # 清理文本，只保留问题描述部分
                        prompt = item['prompt']
                        if "Problem:" in prompt:
                            # 提取Problem部分
                            prompt = prompt.split("Problem:")[1]
                            if "A:" in prompt:
                                prompt = prompt.split("A:")[0]
                        
                        texts.append(prompt.strip())
                        codes.append(item['reference_code'].strip())
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return
    
    print(f"已提取 {len(texts)} 个文本-代码对")
    
    # 如果样本太少，调整n_samples
    n_samples = min(n_samples, len(texts))
    
    # 加载模型并生成embeddings
    tokenizer, model, device = load_model()
    text_embeddings = generate_embeddings(texts, tokenizer, model, device)
    code_embeddings = generate_embeddings(codes, tokenizer, model, device)
    
    # 使用t-SNE可视化embeddings
    visualize_embeddings(text_embeddings, code_embeddings, method='tsne', 
                         n_samples=n_samples, save_path="tsne_embeddings.png")
    
    # 使用UMAP可视化embeddings
    try:
        visualize_embeddings(text_embeddings, code_embeddings, method='umap', 
                             n_samples=n_samples, save_path="umap_embeddings.png")
    except ImportError:
        print("UMAP未安装，跳过UMAP可视化。")

if __name__ == "__main__":
    # 处理DS1000数据集，显示100个点
    process_ds1000(n_samples=100)