# CodeGenReasoning
A Framework for Comparing Reasoning between ChatGPT and DeepSeek during Code Generation Tasks

## Overview
This project provides a framework to analyze and visualize the Chain of Thought (CoT) reasoning processes between different large language models (LLMs) during code generation tasks. Currently, it supports comparison between ChatGPT and DeepSeek models.

## Visualizing Model Reasoning

### Frames of Mind: Animating Thought Processes
We visualize the "thought process" by:
- Capturing chains of thought as text from DeepSeek and GPT4
- Converting the text to embeddings using:
  - OpenAI API
  - CodeBERT embeddings
- Visualizing the embeddings sequentially using t-SNE dimensionality reduction

Example visualization:
| |
|---------|
| ![Thought Process Animation](dynamic-img-withrefcode/deepseek/codebert/0/simple_animation.gif) |

### Consecutive Distance Analysis
To understand the cognitive "jumps" between consecutive thoughts, we analyze the distance between sequential embedding pairs:

| |
|---------|
| ![Distance Analysis](dynamic-img-withrefcode/deepseek/codebert/0/distance.gif) |

The distances are calculated using cosine similarity between embeddings and normalized to [0,1] scale across all consecutive steps. This helps identify significant transitions in the reasoning process.

### Combined Visualization
For comprehensive analysis, we provide a combined view of both thought process and distance metrics:

| |
|---------|
| ![Combined Analysis](dynamic-img-withrefcode/deepseek/codebert/0/dual_animation.gif) |

## Project Setup

### Data Sources
- Primary dataset: DS1000
  - Contains diverse programming prompts
  - Includes reference code solutions
  - Covers various programming tasks and domains

### Components
1. **Data Generation**
   - Pre-generated Chain of Thought processes using:
     - OpenAI's ChatGPT
     - DeepSeek's model
   - Stored in the `data` directory

2. **Analysis Tools**
   - Embedding generation
   - Visualization utilities
   - Comparative analysis tools

3. **Visualization Features**
   - Interactive thought process animations
   - Distance metric visualizations
   - Combined analysis views

## Usage
1. Clone the repository
2. Install dependencies (requirements.txt)
3. Configure API keys for OpenAI and DeepSeek
4. Run analysis scripts for your chosen prompts

## Future Work
- Support for additional LLMs
- Enhanced visualization options
- Extended analysis metrics
- Integration with more datasets

## Citation
If you find this work useful in your research, please consider citing it.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.