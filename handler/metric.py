import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

class CodeReasoningMetrics:
    """
    A class implementing metrics for evaluating code generation reasoning patterns,
    visualization quality, and embedding effectiveness.
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, 
                 gamma: float = 0.5, delta: float = 0.5):
        """
        Initialize metrics with configurable weights.
        
        Args:
            alpha, beta: Weights for VAS calculation
            gamma, delta: Weights for EQM calculation
        """
        # Validate weights sum to 1
        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta must equal 1"
        assert abs(gamma + delta - 1.0) < 1e-6, "gamma + delta must equal 1"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def reasoning_step_distribution(self, phase_steps: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate Reasoning Step Distribution (RSD) for each phase.
        
        Args:
            phase_steps: Dictionary mapping phase names to step counts
            
        Returns:
            Dictionary of phase distribution percentages
        """
        total_steps = sum(phase_steps.values())
        return {phase: (steps/total_steps)*100 
                for phase, steps in phase_steps.items()}

    def phase_transition_clarity(self, 
                               predicted_boundaries: List[int],
                               actual_boundaries: List[int],
                               total_steps: int) -> float:
        """
        Calculate Phase Transition Clarity (PTC) score.
        
        Args:
            predicted_boundaries: Model-predicted phase transition points
            actual_boundaries: Ground truth phase transition points
            total_steps: Total number of reasoning steps
            
        Returns:
            PTC score between 0 and 1
        """
        if len(predicted_boundaries) != len(actual_boundaries):
            raise ValueError("Predicted and actual boundaries must have same length")
            
        differences = [abs(p - a) for p, a in zip(predicted_boundaries, actual_boundaries)]
        avg_error = sum(differences) / (len(differences) * total_steps)
        return 1 - avg_error

    def visualization_accuracy_score(self,
                                   embeddings: np.ndarray,
                                   phase_labels: List[int],
                                   predicted_boundaries: List[int],
                                   actual_boundaries: List[int]) -> float:
        """
        Calculate Visualization Accuracy Score (VAS).
        
        Args:
            embeddings: Matrix of embedded vectors
            phase_labels: Phase labels for each point
            predicted_boundaries: Predicted phase transitions
            actual_boundaries: Actual phase transitions
            
        Returns:
            Combined VAS score
        """
        # Calculate cluster separation using silhouette score
        cluster_separation = silhouette_score(embeddings, phase_labels)
        
        # Calculate boundary precision using PTC
        boundary_precision = self.phase_transition_clarity(
            predicted_boundaries, 
            actual_boundaries,
            len(embeddings)
        )
        
        return self.alpha * cluster_separation + self.beta * boundary_precision

    def semantic_coherence_index(self, 
                               step_embeddings: np.ndarray,
                               similarity_threshold: float = 0.7) -> float:
        """
        Calculate Semantic Coherence Index (SCI).
        
        Args:
            step_embeddings: Embedded vectors for sequential steps
            similarity_threshold: Threshold for semantic connection
            
        Returns:
            SCI score as percentage
        """
        # Calculate cosine similarity between consecutive steps
        similarities = cosine_similarity(step_embeddings[:-1], step_embeddings[1:])
        connected_steps = np.sum(similarities > similarity_threshold)
        return (connected_steps / (len(step_embeddings) - 1)) * 100

    def embedding_quality_measure(self,
                                original_distances: np.ndarray,
                                embedded_distances: np.ndarray,
                                k_neighbors: int = 5) -> float:
        """
        Calculate Embedding Quality Measure (EQM).
        
        Args:
            original_distances: Pairwise distances in original space
            embedded_distances: Pairwise distances in embedded space
            k_neighbors: Number of neighbors for local structure
            
        Returns:
            Combined EQM score
        """
        # Local structure preservation (k-nearest neighbors preservation)
        def get_knn_indices(distances: np.ndarray, k: int) -> np.ndarray:
            return np.argsort(distances)[:, 1:k+1]
        
        original_knn = get_knn_indices(original_distances, k_neighbors)
        embedded_knn = get_knn_indices(embedded_distances, k_neighbors)
        
        local_preservation = np.mean([
            len(set(o) & set(e)) / k_neighbors
            for o, e in zip(original_knn, embedded_knn)
        ])
        
        # Global distance maintenance (correlation between distance matrices)
        global_correlation = np.corrcoef(
            original_distances.flatten(),
            embedded_distances.flatten()
        )[0, 1]
        
        return self.gamma * local_preservation + self.delta * global_correlation

    def evaluate_all_metrics(self,
                           phase_steps: Dict[str, int],
                           embeddings: np.ndarray,
                           phase_labels: List[int],
                           predicted_boundaries: List[int],
                           actual_boundaries: List[int],
                           original_distances: np.ndarray,
                           embedded_distances: np.ndarray) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Returns:
            Dictionary containing all metric scores
        """
        return {
            'rsd': self.reasoning_step_distribution(phase_steps),
            'ptc': self.phase_transition_clarity(
                predicted_boundaries, actual_boundaries, len(embeddings)
            ),
            'vas': self.visualization_accuracy_score(
                embeddings, phase_labels, predicted_boundaries, actual_boundaries
            ),
            'sci': self.semantic_coherence_index(embeddings),
            'eqm': self.embedding_quality_measure(
                original_distances, embedded_distances
            )
        }
    




