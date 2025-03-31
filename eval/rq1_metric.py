import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def calculate_metrics(model_results):
    """
    Calculate QCMR and REI metrics for different model versions and problem categories.
    
    Args:
        model_results: Dictionary containing embedding distances and reasoning steps for each model
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    for model_name, data in model_results.items():
        # Initialize metrics dictionary for this model
        metrics[model_name] = {
            'qcmr': {},
            'rei': {},
            'qcmr_by_complexity': {}
        }
        
        # Calculate average QCMR
        initial_distances = np.array(data['initial_distances'])
        final_distances = np.array(data['final_distances'])
        qcmr_values = (1 - final_distances) * 100  # Convert to percentage
        metrics[model_name]['qcmr']['average'] = np.mean(qcmr_values)
        
        # Calculate REI
        distance_reductions = initial_distances - final_distances
        reasoning_steps = np.array(data['reasoning_steps'])
        rei_values = distance_reductions / reasoning_steps
        metrics[model_name]['rei']['average'] = np.mean(rei_values)
        
        # Calculate QCMR by complexity
        for complexity in ['simple', 'medium', 'complex']:
            indices = data['complexity_indices'][complexity]
            complexity_qcmr = np.mean(qcmr_values[indices])
            metrics[model_name]['qcmr_by_complexity'][complexity] = complexity_qcmr
        
        # Calculate metrics by problem category (for V3 only)
        if model_name == 'DeepSeek-V3':
            metrics[model_name]['categories'] = {}
            for category, indices in data['category_indices'].items():
                category_qcmr = np.mean(qcmr_values[indices])
                category_rei = np.mean(rei_values[indices])
                
                # Calculate QCMR by complexity for this category
                category_qcmr_by_complexity = {}
                for complexity in ['simple', 'medium', 'complex']:
                    # Get indices that are both in this category and this complexity level
                    complexity_indices = data['complexity_indices'][complexity]
                    combined_indices = list(set(indices).intersection(set(complexity_indices)))
                    if combined_indices:
                        category_qcmr_by_complexity[complexity] = np.mean(qcmr_values[combined_indices])
                    else:
                        category_qcmr_by_complexity[complexity] = np.nan
                
                metrics[model_name]['categories'][category] = {
                    'qcmr': category_qcmr,
                    'rei': category_rei,
                    'qcmr_by_complexity': category_qcmr_by_complexity
                }
    
    return metrics

def visualize_results(metrics):
    """
    Visualize the calculated metrics
    
    Args:
        metrics: Dictionary with calculated metrics
    """
    # Create figure for model comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Extract model names and metrics
    models = list(metrics.keys())
    qcmr_values = [metrics[model]['qcmr']['average'] for model in models]
    rei_values = [metrics[model]['rei']['average'] for model in models]
    
    # Plot QCMR comparison
    ax1.bar(models, qcmr_values, color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Query-Code Match Rate (%)')
    ax1.set_title('QCMR Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot REI comparison
    ax2.bar(models, rei_values, color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Reasoning Efficiency Index')
    ax2.set_title('REI Comparison')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()
    
    # Create figure for complexity analysis (DeepSeek-V3)
    v3_data = metrics['DeepSeek-V3']
    categories = list(v3_data['categories'].keys())
    categories.insert(0, 'Average')
    
    # Prepare data
    qcmr_by_category = [v3_data['qcmr']['average']]
    qcmr_by_category.extend([v3_data['categories'][cat]['qcmr'] for cat in categories[1:]])
    
    rei_by_category = [v3_data['rei']['average']]
    rei_by_category.extend([v3_data['categories'][cat]['rei'] for cat in categories[1:]])
    
    # Plot category comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.bar(categories, qcmr_by_category, color='#2ecc71')
    ax1.set_ylabel('Query-Code Match Rate (%)')
    ax1.set_title('QCMR by Problem Category (DeepSeek-V3)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.set_ylim(80, 90)
    
    ax2.bar(categories, rei_by_category, color='#2ecc71')
    ax2.set_ylabel('Reasoning Efficiency Index')
    ax2.set_title('REI by Problem Category (DeepSeek-V3)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('category_analysis.png', dpi=300)
    
    return True

# Example usage with dummy data
# In practice, you would load real data from your experiments
model_results = {
    'DeepSeek-V3': {
        'initial_distances': np.random.uniform(0.4, 0.6, 100),
        'final_distances': np.random.uniform(0.1, 0.15, 100),
        'reasoning_steps': np.random.randint(3, 6, 100),
        'complexity_indices': {
            'simple': list(range(0, 33)),
            'medium': list(range(33, 66)),
            'complex': list(range(66, 100))
        },
        'category_indices': {
            'Arrays': list(range(0, 35)),
            'Binary Trees': list(range(35, 70)),
            'Dynamic Programming': list(range(70, 100))
        }
    },
    'DeepSeek-R1': {
        'initial_distances': np.random.uniform(0.4, 0.6, 100),
        'final_distances': np.random.uniform(0.15, 0.22, 100),
        'reasoning_steps': np.random.randint(4, 8, 100),
        'complexity_indices': {
            'simple': list(range(0, 33)),
            'medium': list(range(33, 66)),
            'complex': list(range(66, 100))
        }
    }
}

# Calculate and visualize metrics
metrics = calculate_metrics(model_results)
visualize_results(metrics)

# Print results in a format that could be used for the LaTeX table
print("Model Results Summary:")
for model in ['DeepSeek-V3', 'DeepSeek-R1']:
    print(f"\n{model}:")
    print(f"Average QCMR: {metrics[model]['qcmr']['average']:.1f}%")
    print(f"Average REI: {metrics[model]['rei']['average']:.3f}")
    print("QCMR by Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        print(f"  {complexity.capitalize()}: {metrics[model]['qcmr_by_complexity'][complexity]:.1f}%")

if 'categories' in metrics['DeepSeek-V3']:
    print("\nDeepSeek-V3 Category Breakdown:")
    for category in metrics['DeepSeek-V3']['categories']:
        cat_data = metrics['DeepSeek-V3']['categories'][category]
        print(f"\n{category}:")
        print(f"QCMR: {cat_data['qcmr']:.1f}%")
        print(f"REI: {cat_data['rei']:.3f}")