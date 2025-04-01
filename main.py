import json
#
from utils.tsne_show import *
from utils.distance import *
from utils.text_embedding import TextEmbedder
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def get_cot_from_json(filepath):
    cot_dict = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            cot_list = []
            id = item['id']
            cot = item['reasoning_chains']
            prompt = item['prompt']
            reference_code = item['reference_code']
            cot_list.append(prompt)
            cot_list.extend(cot)
            cot_list.append(reference_code)
            cot_dict[id] = cot_list

    return cot_dict


# get CoT
filepath = f"data/chains/{os.getenv('COT_MODEL')}/ds{os.getenv('LIMIT')}_cot.json"
cot_dict = get_cot_from_json(filepath)

for id, cot_list in tqdm(cot_dict.items(), desc="Generating visualizations"):
    reference_dict = TextEmbedder().get_reference_dictionary(cot_list)

    # make visualizaer
    visualizer = AnimatedTSNEVisualizer()
    visualizer.from_dict(reference_dict)

    embedding_model = os.getenv("EMBEDDING_MODEL")
    cot_model = os.getenv("COT_MODEL")

    if not os.path.exists(f"dynamic-img/{cot_model}/{embedding_model}/{id}"):
        os.makedirs(f"dynamic-img/{cot_model}/{embedding_model}/{id}")

    # simple CoT viz
    visualizer.create_animation(f'dynamic-img/{cot_model}/{embedding_model}/{id}/simple_animation.gif', show_line=True)

    # distance bars
    visualizer.create_distance_animation(f'dynamic-img/{cot_model}/{embedding_model}/{id}/distance.gif', metric="cosine", normalization="maxunit")




    # combined t-SNE and distance bars
    visualizer.create_combined_animation(f'dynamic-img/{cot_model}/{embedding_model}/{id}/dual_animation.gif', show_line=True)


    if not os.path.exists(f"static-image/{cot_model}/{embedding_model}/{id}"):
        os.makedirs(f"static-image/{cot_model}/{embedding_model}/{id}")

    # 保存t-SNE静态图序列
    visualizer.save_static_tsne(f'static-image/{cot_model}/{embedding_model}/{id}/tsne.png', show_line=True)

    # 保存距离条形图静态图序列
    visualizer.save_static_distance(f'static-image/{cot_model}/{embedding_model}/{id}/distance.png', metric="cosine", normalization="maxunit")

    # 保存组合视图静态图序列
    visualizer.save_static_combined(f'static-image/{cot_model}/{embedding_model}/{id}/combined.png', show_line=True)

    distances = visualizer.calculate_consecutive_distances( metric="euclidean", normalization="maxunit")

    # 保存距离值
    if not os.path.exists(f"metrics/{cot_model}/{embedding_model}/{id}"):
        os.makedirs(f"metrics/{cot_model}/{embedding_model}/{id}")

    # 保存距离值
    with open(f'metrics/{cot_model}/{embedding_model}/{id}/distances.json', 'w', encoding='utf-8') as f:
        json.dump(distances, f, ensure_ascii=False, indent=4)