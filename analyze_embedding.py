import json
import numpy as np
from utils.text_embedding import TextEmbedder
from tqdm import tqdm
import os
from dotenv import load_dotenv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
import seaborn as sns
from scipy.stats import entropy
import matplotlib.pyplot as plt

load_dotenv()

class EmbeddingAnalyzer:
    def __init__(self):
        self.embedder = TextEmbedder()
        
    def load_cot_data(self, file_path):
        """加载Chain of Thought数据"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def compute_average_embedding(self, reasoning_chains):
        """计算reasoning chains的平均embedding"""
        # 获取每个步骤的embedding
        embeddings = []
        for chain in reasoning_chains:
            embedding = self.embedder.get_embedding(chain,normalize=True)
            embeddings.append(embedding)
        # 计算平均embedding
        return np.mean(embeddings, axis=0)
    
    def process_embeddings(self, input_path, output_dir):
        """处理数据并保存embeddings"""
        # 加载数据
        data = self.load_cot_data(input_path)
        
        # 创建一个字典来存储id到embedding的映射
        id_to_embedding = {}
        
        # 处理每个样本
        for item in tqdm(data, desc="Processing embeddings"):
            sample_id = item['id']
            reasoning_chains = item['reasoning_chains']
            
            # 计算平均embedding
            avg_embedding = self.compute_average_embedding(reasoning_chains)
            
            # 存储结果
            id_to_embedding[sample_id] = avg_embedding
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取模型名称从输入路径
        model_name = os.path.basename(os.path.dirname(input_path))
        embedding_model = os.getenv("EMBEDDING_MODEL")
        
        # 保存结果
        output_path = os.path.join(output_dir, f"{model_name}_{embedding_model}_embeddings.npy")
        np.save(output_path, id_to_embedding)
        print(f"Saved embeddings to {output_path}")
        
        return id_to_embedding

def genrate_embeddings(cot_model,limit):
    # 设置路径
    input_path = f"data/chains/{cot_model}/ds{limit}_cot.json"
    output_dir = f"data/embeddings/{cot_model}"
    
    # 创建分析器实例
    analyzer = EmbeddingAnalyzer()
    
    # 处理数据
    embeddings = analyzer.process_embeddings(input_path, output_dir)
    
    # 打印一些基本信息
    print(f"Processed {len(embeddings)} samples")
    print(f"Embedding dimension: {next(iter(embeddings.values())).shape}")


def visualize_embeddings_comparison(embeddings1, embeddings2, output_dir, title="Embeddings Distribution"):
    """
    使用TSNE可视化两组embeddings的分布
    Args:
        embeddings1: 第一组embeddings字典 (id -> embedding)
        embeddings2: 第二组embeddings字典 (id -> embedding)
        output_dir: 输出目录
        title: 图表标题
    """
    # 转换embeddings为numpy数组
    emb1_array = np.array(list(embeddings1.values()))
    emb2_array = np.array(list(embeddings2.values()))
    
    # 合并两组embeddings
    combined_embeddings = np.vstack([emb1_array, emb2_array])
    
    # 使用TSNE降维到2D
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(combined_embeddings)
    
    # 分离两组数据的降维结果
    n1 = len(embeddings1)
    emb1_2d = embeddings_2d[:n1]
    emb2_2d = embeddings_2d[n1:]
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(emb1_2d[:, 0], emb1_2d[:, 1], c='red', label='With Reference Code', alpha=0.6)
    plt.scatter(emb2_2d[:, 0], emb2_2d[:, 1], c='blue', label='Without Reference Code', alpha=0.6)
    
    plt.title(title)
    plt.legend()
    
    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'embedding_distribution.png'))
    plt.close()

def analyze_embeddings_statistics(embeddings1, embeddings2, output_dir, title_prefix=""):
    """分析两组embeddings的统计特性"""
    # 转换为numpy数组
    emb1_array = np.array(list(embeddings1.values()))
    emb2_array = np.array(list(embeddings2.values()))
    
    # 1. 计算基本统计量并确保使用Python原生float类型
    stats = {
        'With Reference': {
            'mean': float(np.mean(emb1_array)),
            'std': float(np.std(emb1_array)),
            'norm': float(np.linalg.norm(emb1_array, axis=1).mean()),
            'samples': int(len(emb1_array))
        },
        'Without Reference': {
            'mean': float(np.mean(emb2_array)),
            'std': float(np.std(emb2_array)),
            'norm': float(np.linalg.norm(emb2_array, axis=1).mean()),
            'samples': int(len(emb2_array))
        }
    }
    
    # 2. 计算组内余弦相似度分布
    def compute_pairwise_similarities(embeddings):
        return cosine_similarity(embeddings)
    
    sim1 = compute_pairwise_similarities(emb1_array)
    sim2 = compute_pairwise_similarities(emb2_array)
    
    # 绘制相似度分布直方图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.histplot(sim1[np.triu_indices(sim1.shape[0], k=1)], 
                 label='With Reference Code', alpha=0.5, color='red')
    sns.histplot(sim2[np.triu_indices(sim2.shape[0], k=1)], 
                 label='Without Reference Code', alpha=0.5, color='blue')
    plt.title(f'{title_prefix}\nPairwise Cosine Similarity Distribution')
    plt.legend()
    
    # 3. 计算维度方差分布
    dim_var1 = np.var(emb1_array, axis=0)
    dim_var2 = np.var(emb2_array, axis=0)
    
    plt.subplot(1, 2, 2)
    sns.kdeplot(dim_var1, label='With Reference Code', color='red')
    sns.kdeplot(dim_var2, label='Without Reference Code', color='blue')
    plt.title('Dimension Variance Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'embedding_statistics.png'))
    plt.close()
    
    # 4. 计算Wasserstein距离
    w_distance = float(wasserstein_distance(
        np.mean(emb1_array, axis=0), 
        np.mean(emb2_array, axis=0)
    ))
    
    # 5. 计算平均聚集度（使用平均余弦相似度）
    cohesion1 = float(np.mean(sim1[np.triu_indices(sim1.shape[0], k=1)]))
    cohesion2 = float(np.mean(sim2[np.triu_indices(sim2.shape[0], k=1)]))
    
    # 添加到统计结果
    stats['Comparison'] = {
        'wasserstein_distance': w_distance,
        'cohesion_with_ref': cohesion1,
        'cohesion_without_ref': cohesion2
    }
    
    # 打印统计结果
    print("\nEmbedding Statistics:")
    print("=" * 50)
    for category, metrics in stats.items():
        print(f"\n{category}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
    
    return stats


def show_embeddings_comparison(cot_model,limit):
    # 加载两组embeddings
    cot_model = os.getenv("COT_MODEL")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    limit = os.getenv("LIMIT")
    
    # 加载有reference code的embeddings
    with_ref_path = f"data/embeddings/{cot_model}/{cot_model}_{embedding_model}_embeddings_with_refcode.npy"
    with_ref_embeddings = np.load(with_ref_path, allow_pickle=True).item()
    
    # 加载没有reference code的embeddings
    without_ref_path = f"data/embeddings/{cot_model}/{cot_model}_{embedding_model}_embeddings_without_refcode.npy"
    without_ref_embeddings = np.load(without_ref_path, allow_pickle=True).item()
    
    # 可视化对比
    output_dir = f"data/visualizations/{cot_model}"
    visualize_embeddings_comparison(
        with_ref_embeddings,
        without_ref_embeddings,
        output_dir,
        f"Embedding Distribution ({cot_model} - {embedding_model})"
    )
    
    print("Visualization saved to embedding_distribution.png")


    # 添加统计分析
    stats = analyze_embeddings_statistics(
        with_ref_embeddings,
        without_ref_embeddings,
        output_dir,
        f"{cot_model} - {embedding_model}"
    )
    
    # 保存统计结果
    with open(os.path.join(output_dir, 'embedding_statistics.json'), 'w') as f:
        json.dump(stats, f, indent=4)



if __name__ == "__main__":
    cot_model = os.getenv("COT_MODEL")
    limit = os.getenv("LIMIT")
    # genrate_embeddings(cot_model,limit)
    show_embeddings_comparison(cot_model,limit)