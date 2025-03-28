from utils.metric import CodeReasoningMetrics
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

def main():
    # 1. phase_steps: 各阶段步骤计数
    phase_data = {
        'search': 5,      # 搜索阶段有5个推理步骤
        'thinking': 12,   # 思考阶段有12个推理步骤
        'conclusion': 3   # 结论阶段有3个推理步骤
    }

    # 2. embeddings: 推理步骤的嵌入向量 (20个步骤，每个步骤128维向量)
    embedded_vectors = np.random.rand(20, 128)  # 实际应用中这是从模型获得的嵌入向量

    # 3. phase_labels: 每个步骤属于哪个阶段 (0=search, 1=thinking, 2=conclusion)
    labels = np.array([
        0, 0, 0, 0, 0,                 # 5个搜索阶段步骤
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 12个思考阶段步骤
        2, 2, 2                        # 3个结论阶段步骤
    ])

    # 4. predicted_boundaries: 模型预测的阶段转换点（步骤索引）
    pred_bounds = [5, 17]  # 第5步后转换到thinking，第17步后转换到conclusion

    # 5. actual_boundaries: 实际的阶段转换点（人工标注）
    true_bounds = [5, 17]  # 真实的转换点

    # 6. 计算原始空间和嵌入空间的距离矩阵
    # 原始空间距离（这里用随机值模拟，实际应用中应该是基于原始特征计算的距离）
    orig_dist = np.random.rand(20, 20)
    orig_dist = (orig_dist + orig_dist.T) / 2  # 确保对称
    np.fill_diagonal(orig_dist, 0)  # 对角线设为0

    # 嵌入空间距离（基于embedded_vectors计算）

    emb_dist = euclidean_distances(embedded_vectors)

    # 使用示例
    metrics = CodeReasoningMetrics(alpha=0.6, beta=0.4, gamma=0.7, delta=0.3)

    # 计算所有指标
    results = metrics.evaluate_all_metrics(
        phase_steps=phase_data,
        embeddings=embedded_vectors,
        phase_labels=labels,
        predicted_boundaries=pred_bounds,
        actual_boundaries=true_bounds,
        original_distances=orig_dist,
        embedded_distances=emb_dist
    )

    # 打印结果
    print("评估结果:")
    for metric, value in results.items():
        print(f"{metric}: {value}")



    
if __name__ == "__main__":
    main()


