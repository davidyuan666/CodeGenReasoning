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

    distances = visualizer.calculate_consecutive_distances( metric="euclidean", normalization="maxunit")
    print(distances)

    # 保存距离值
    if not os.path.exists(f"metrics/{cot_model}/{embedding_model}/{id}"):
        os.makedirs(f"metrics/{cot_model}/{embedding_model}/{id}")

    # 保存距离值
    with open(f'metrics/{cot_model}/{embedding_model}/{id}/distances.json', 'w', encoding='utf-8') as f:
        json.dump(distances, f, ensure_ascii=False, indent=4)