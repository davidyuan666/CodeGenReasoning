from utils.metric import CodeReasoningMetrics

def main():
    # 初始化评估器
    metrics = CodeReasoningMetrics(alpha=0.6, beta=0.4, gamma=0.7, delta=0.3)

    # 计算单个指标
    rsd_scores = metrics.reasoning_step_distribution({
        'search': 10,
        'thinking': 25,
        'conclusion': 15
    })

    # 计算所有指标
    all_scores = metrics.evaluate_all_metrics(
        phase_steps=phase_data,
        embeddings=embedded_vectors,
        phase_labels=labels,
        predicted_boundaries=pred_bounds,
        actual_boundaries=true_bounds,
        original_distances=orig_dist,
        embedded_distances=emb_dist
    )


    
if __name__ == "__main__":
    main()


