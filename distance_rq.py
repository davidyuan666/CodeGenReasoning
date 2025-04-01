import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def load_distances(base_path):
    distances = {
        'chatgpt': {'codebert': [], 'openai': []},
        'deepseek': {'codebert': [], 'openai': []}
    }
    
    # 遍历所有文件
    for model in ['chatgpt', 'deepseek']:
        for embedding in ['codebert', 'openai']:
            path_pattern = f"{base_path}/{model}/{embedding}/*/distances.json"
            files = glob(path_pattern)
            
            for file in files:
                with open(file, 'r', encoding='utf-8') as f:  # Added encoding='utf-8'
                    data = json.load(f)
                    # 提取每个文件中的distances
                    file_distances = [item['distance'] for item in data]
                    distances[model][embedding].append(file_distances)
    
    return distances

def analyze_distances(distances):
    stats = {}
    for model in distances:
        stats[model] = {}
        for embedding in distances[model]:
            # 将所有distance展平成一个列表
            all_distances = [d for sublist in distances[model][embedding] for d in sublist]
            stats[model][embedding] = {
                'mean': np.mean(all_distances),
                'median': np.median(all_distances),
                'std': np.std(all_distances),
                'min': np.min(all_distances),
                'max': np.max(all_distances)
            }
    return stats

def plot_distributions(distances):
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制分布图
    sns.histplot(data=[d for d in distances['chatgpt']['codebert']], ax=ax1)
    ax1.set_title('ChatGPT - CodeBERT')
    
    sns.histplot(data=[d for d in distances['chatgpt']['openai']], ax=ax2)
    ax2.set_title('ChatGPT - OpenAI')
    
    sns.histplot(data=[d for d in distances['deepseek']['codebert']], ax=ax3)
    ax3.set_title('DeepSeek - CodeBERT')
    
    sns.histplot(data=[d for d in distances['deepseek']['openai']], ax=ax4)
    ax4.set_title('DeepSeek - OpenAI')
    
    plt.tight_layout()
    # plt.show()
    os.makedirs('metrics/distance', exist_ok=True)
    plt.savefig(f'metrics/distance/distance_distributions.png')
    plt.close()

def plot_boxplots(distances):
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    for model in distances:
        for embedding in distances[model]:
            flat_distances = [d for sublist in distances[model][embedding] for d in sublist]
            data.append(flat_distances)
            labels.append(f'{model}-{embedding}')
    
    plt.boxplot(data, labels=labels)
    plt.title('Distance Distributions Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Distance')
    plt.tight_layout()
    # plt.show()
    os.makedirs('metrics/distance', exist_ok=True)
    plt.savefig('metrics/distance/distance_boxplots.png')
    plt.close()

# 主执行代码
base_path = 'metrics'
distances = load_distances(base_path)
stats = analyze_distances(distances)

# 打印统计信息
for model in stats:
    print(f"\n{model} 统计信息:")
    for embedding in stats[model]:
        print(f"\n{embedding}:")
        for metric, value in stats[model][embedding].items():
            print(f"{metric}: {value:.4f}")

# 生成可视化
plot_distributions(distances)
plot_boxplots(distances)