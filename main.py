import json
#
from utils.tsne_show import *
from utils.distance import *
from utils.text_embedding import TextEmbedder
import os

COT_END_OFFSET = 2

##### Load

def get_cot(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

        # Find where the <next> tag is
        delimiter_index = None
        for i, text in enumerate(data):
            if "<next>" in text:
                delimiter_index = i

        # If we found a delimiter, only take texts before it
        if delimiter_index is not None:
            return [text for text in data[:delimiter_index + COT_END_OFFSET] if text.strip() and "<next>" not in text]

        # If no delimiter found, return all non-empty texts
        return [text for text in data if text.strip() and "<next>" not in text]

###### Distances

def aggregate_distances(filepaths):

    distances = []

    for filepath in filepaths:
        # get cot for filepath
        texts = get_cot(filepath)
        # get embeddings
        reference_dict = TextEmbedder().get_reference_dictionary(texts)
        # get dict
        visualizer = AnimatedTSNEVisualizer()
        visualizer.from_dict(reference_dict)
        # get distances
        distances_for_text = visualizer.calculate_consecutive_distances(
            metric="cosine",
            normalization="maxunit"
        )
        # parse distances
        distances_parsed = [distance_for_text["distance"] for distance_for_text in distances_for_text]
        # append
        distances.append(distances_parsed)

    plot_normalized_sequences(distances, show_individual=True)



# get CoT
filepath = "data/chains/1.json"
texts = get_cot(filepath)
reference_dict = TextEmbedder().get_reference_dictionary(texts)

# make visualizaer
visualizer = AnimatedTSNEVisualizer()
visualizer.from_dict(reference_dict)

if not os.path.exists("dynamic-img"):
    os.makedirs("dynamic-img")

# simple CoT viz
visualizer.create_animation('dynamic-img/simple_animation.gif', show_line=True)

# distance bars
visualizer.create_distance_animation('dynamic-img/distance.gif', metric="cosine", normalization="maxunit")

# combined t-SNE and distance bars
visualizer.create_combined_animation('dynamic-img/dual_animation.gif', show_line=True)

# aggregate distances
# filepaths=["data/chains/1.json", "data/chains/2.json", "data/chains/3.json", "data/chains/4.json", "data/chains/5.json", "data/chains/6.json", "data/chains/7.json", "data/chains/8.json", "data/chains/9.json", "data/chains/10.json"]
# aggregate_distances(filepaths)


if not os.path.exists("static-image"):
    os.makedirs("static-image")

# 保存t-SNE静态图序列
visualizer.save_static_tsne('static-image/tsne.png', show_line=True)

# 保存距离条形图静态图序列
visualizer.save_static_distance('static-image/distance.png', metric="cosine", normalization="maxunit")

# 保存组合视图静态图序列
visualizer.save_static_combined('static-image/combined.png', show_line=True)
