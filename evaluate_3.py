import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from dotenv import load_dotenv

load_dotenv()


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
            
            # Check if we have any distances to analyze
            if all_distances:
                stats[model][embedding] = {
                    'mean': np.mean(all_distances),
                    'median': np.median(all_distances),
                    'std': np.std(all_distances),
                    'min': np.min(all_distances),
                    'max': np.max(all_distances)
                }
            else:
                # Handle empty arrays by setting default values
                stats[model][embedding] = {
                    'mean': 0,
                    'median': 0,
                    'std': 0,
                    'min': 0,
                    'max': 0
                }
    return stats

def plot_distributions(distances):
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 绘制分布图
    for model in distances:
        for embedding in distances[model]:
            flat_distances = [d for sublist in distances[model][embedding] for d in sublist]
            if flat_distances:  # Only plot if we have data
                if model == 'chatgpt' and embedding == 'codebert':
                    sns.histplot(data=flat_distances, ax=ax1)
                    ax1.set_title('ChatGPT - CodeBERT')
                elif model == 'chatgpt' and embedding == 'openai':
                    sns.histplot(data=flat_distances, ax=ax2)
                    ax2.set_title('ChatGPT - Text-embedding-3-large')
                elif model == 'deepseek' and embedding == 'codebert':
                    sns.histplot(data=flat_distances, ax=ax3)
                    ax3.set_title('DeepSeek - CodeBERT')
                elif model == 'deepseek' and embedding == 'openai':
                    sns.histplot(data=flat_distances, ax=ax4)
                    ax4.set_title('DeepSeek - Text-embedding-3-large')
    
    plt.tight_layout()
    os.makedirs(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}', exist_ok=True)
    plt.savefig(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}/distance_distributions.png')
    plt.close()

def plot_boxplots(distances):
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    for model in distances:
        for embedding in distances[model]:
            flat_distances = [d for sublist in distances[model][embedding] for d in sublist]
            if flat_distances:  # Only add to plot if we have data
                data.append(flat_distances)
                labels.append(f'{model}-{embedding}')
    
    if data:  # Only create plot if we have any data
        plt.boxplot(data, labels=labels)
        plt.title('Distance Distributions Comparison')
        plt.xticks(rotation=45)
        plt.ylabel('Distance')
        plt.tight_layout()
        os.makedirs(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}', exist_ok=True)
        plt.savefig(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}/distance_boxplots.png')
    plt.close()

def calculate_ard(distances):
    """
    Calculate Average Reasoning Distance (ARD)
    
    ARD = (1/(n-1)) * sum(RSD_i) for i=1 to n-1
    where n is the total number of reasoning steps
    """
    ard_results = {
        'chatgpt': {'codebert': [], 'openai': []},
        'deepseek': {'codebert': [], 'openai': []}
    }
    
    for model in distances:
        for embedding in distances[model]:
            # For each sequence of distances
            for sequence in distances[model][embedding]:
                if len(sequence) > 1:  # Need at least 2 steps to calculate ARD
                    # ARD is average of all distances in the sequence
                    ard = sum(sequence) / len(sequence)
                    ard_results[model][embedding].append(ard)
    
    return ard_results

def analyze_ard(ard_results):
    """Analyze ARD results and return statistics"""
    stats = {}
    for model in ard_results:
        stats[model] = {}
        for embedding in ard_results[model]:
            if ard_results[model][embedding]:  # Check if list is not empty
                stats[model][embedding] = {
                    'mean': np.mean(ard_results[model][embedding]),
                    'median': np.median(ard_results[model][embedding]),
                    'std': np.std(ard_results[model][embedding]),
                    'min': np.min(ard_results[model][embedding]),
                    'max': np.max(ard_results[model][embedding])
                }
            else:
                stats[model][embedding] = {
                    'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0
                }
    return stats

def plot_ard_comparison(ard_results):
    """Plot ARD comparison between models and embeddings"""
    plt.figure(figsize=(10, 6))
    
    data = []
    labels = []
    for model in ard_results:
        for embedding in ard_results[model]:
            if ard_results[model][embedding]:  # Check if list is not empty
                data.append(ard_results[model][embedding])
                labels.append(f'{model}-{embedding}')
    
    plt.boxplot(data, labels=labels)
    plt.title('Average Reasoning Distance (ARD) Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('ARD')
    plt.tight_layout()
    os.makedirs(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}', exist_ok=True)
    plt.savefig(f'metrics/distance/{os.getenv("COT_MODEL")}/{os.getenv("EMBEDDING_MODEL")}/ard_comparison.png')
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



# 计算和分析ARD
ard_results = calculate_ard(distances)
ard_stats = analyze_ard(ard_results)

# 打印ARD统计信息
print("\n\n===== Average Reasoning Distance (ARD) 统计信息 =====")
for model in ard_stats:
    print(f"\n{model}:")
    for embedding in ard_stats[model]:
        print(f"\n{embedding}:")
        for metric, value in ard_stats[model][embedding].items():
            print(f"{metric}: {value:.4f}")

# 生成可视化
plot_distributions(distances)
plot_boxplots(distances)
plot_ard_comparison(ard_results)
