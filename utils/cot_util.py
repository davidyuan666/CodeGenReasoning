import json
import os
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel

load_dotenv()

class OutputFormat(BaseModel):
    step: str
    description: str



class CoTUtil:
    def __init__(self, data_path: str = "data/ds1000.json", output_dir: str = "data/chains"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        self.limit = int(os.getenv("LIMIT"))
        
    def read_dataset(self) -> List[Dict]:
        """读取数据集文件"""
        try:
            codelist = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                json_list = f.readlines()
                for index, json_str in tqdm(enumerate(json_list), desc="Reading dataset", total=min(0, len(json_list))):
                    if index > self.limit:
                        break
                    json_obj = json.loads(json_str)
                    json_obj['reference_code'] = json_obj['reference_code'].replace('\n', '')
                    json_obj['prompt'] = json_obj['prompt'].replace('\n', '')
                    codelist.append(json_obj)
            return codelist
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise

    def read_dataset_random(self) -> List[Dict]:
        """随机采样读取数据集文件
        
        Args:
            sample_size: 采样数量，默认为None，则使用self.limit
            
        Returns:
            随机采样的数据列表
        """
        import random
        
        sample_size = self.limit
            
        try:
            all_data = []
            with open(self.data_path, 'r', encoding='utf-8') as f:
                json_list = f.readlines()
                for json_str in tqdm(json_list, desc="Loading full dataset"):
                    json_obj = json.loads(json_str)
                    json_obj['reference_code'] = json_obj['reference_code'].replace('\n', '')
                    json_obj['prompt'] = json_obj['prompt'].replace('\n', '')
                    all_data.append(json_obj)
                    
            # 随机采样
            if sample_size >= len(all_data):
                return all_data
            
            sampled_data = random.sample(all_data, sample_size)
            print(f"Randomly sampled {len(sampled_data)} examples from {len(all_data)} total examples")
            return sampled_data
            
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            raise

    
    def generate_chain_of_thought_by_deepseek(self, prompt: str, reference_code: str) -> List[Dict]:
        """生成思维链推理过程"""
        # Please install OpenAI SDK first: `pip3 install openai`
        client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url=os.getenv("DEEPSEEK_BASE_URL"))
        '''
        with reference code
        '''
        # self.messages=[
        #     {"role": "system", "content": "You are a helpful programming assistant. Please explain your thinking step by step and format your response as a JSON object with a 'steps' array. Each step should have a 'step' number and a 'description' field."},
        #     {"role": "user", "content": f"Problem: {prompt}\nReference solution: {reference_code}\nPlease explain how to solve this step by step and return the steps in JSON format with 'step' and 'description' fields."}
        # ]

        '''
        without reference code
        '''
        self.messages=[
            {"role": "system", "content": "You are a helpful programming assistant. Please explain your thinking step by step and format your response as a JSON object with a 'steps' array. Each step should have a 'step' number and a 'description' field."},
            {"role": "user", "content": f"Problem: {prompt}\nPlease explain how to solve this step by step and return the steps in JSON format with 'step' and 'description' fields."}
        ]
        response = client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL"),
            messages=self.messages,
            stream=False
        )

        reasoning = response.choices[0].message.content
        print('reasoning:', reasoning)

          # Parse the JSON response
        try:
            reasoning = reasoning.replace('```json', '').replace('```', '')
            reasoning_json = json.loads(reasoning)
            steps = reasoning_json.get('steps', [])
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            steps = []
        
        # Format the steps - update to handle the new structure
        formatted_steps = []
        for i, step in enumerate(steps):
            # Convert the step to use consistent keys
            if isinstance(step, dict):
                description = step.get('explanation', step.get('description', ''))
                formatted_step = {
                    'step': i + 1,
                    'description': description
                }
                formatted_steps.append(formatted_step)
        
        reasoning_chain = []    
        for step in formatted_steps:
            reasoning_chain.append(step['description'])
        
        print('reasoning_chain:', reasoning_chain)
        return reasoning_chain



    def generate_chain_of_thought_by_gpt(self, prompt: str, reference_code: str) -> List[Dict]:
        """生成思维链推理过程"""
        '''
        with reference code
        '''
        # self.messages = [
        #     {"role": "system", "content": "You are a helpful programming assistant. Please explain your thinking step by step and format your response as a JSON object with a 'steps' array. Each step should have a 'step' number and a 'description' field."},
        #     {"role": "user", "content": f"Problem: {prompt}\nReference solution: {reference_code}\nPlease explain how to solve this step by step and return the steps in JSON format with 'step' and 'description' fields."}
        # ]

        '''
        without reference code
        '''
        self.messages = [
            {"role": "system", "content": "You are a helpful programming assistant. Please explain your thinking step by step and format your response as a JSON object with a 'steps' array. Each step should have a 'step' number and a 'description' field."},
            {"role": "user", "content": f"Problem: {prompt}\nPlease explain how to solve this step by step and return the steps in JSON format with 'step' and 'description' fields."}
        ]

        self.struct_model = "gpt-4o-2024-08-06"

        completion = self.client.beta.chat.completions.parse(
            model=self.struct_model,
            messages=self.messages,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        reasoning = completion.choices[0].message.content
        print('reasoning:', reasoning)
        
        # Parse the JSON response
        try:
            reasoning_json = json.loads(reasoning)
            steps = reasoning_json.get('steps', [])
        except json.JSONDecodeError:
            print("Failed to parse JSON response")
            steps = []
        
        # Format the steps - update to handle the new structure
        formatted_steps = []
        for i, step in enumerate(steps):
            # Convert the step to use consistent keys
            if isinstance(step, dict):
                description = step.get('explanation', step.get('description', ''))
                formatted_step = {
                    'step': i + 1,
                    'description': description
                }
                formatted_steps.append(formatted_step)
        
        reasoning_chain = []    
        for step in formatted_steps:
            reasoning_chain.append(step['description'])
        return reasoning_chain
    
    
    def process_and_save(self):
        """处理数据集并保存推理链结果"""
        # codelist = self.read_dataset()
        codelist = self.read_dataset_random()
        results = []
        
        for idx, item in tqdm(enumerate(codelist), desc="Generating reasoning chains", total=len(codelist)):
            prompt = item['prompt']
            reference_code = item['reference_code']
            
            # 生成推理链
            if os.getenv("COT_MODEL") == "chatgpt":
                reasoning_chains = self.generate_chain_of_thought_by_gpt(prompt, reference_code)
            elif os.getenv("COT_MODEL") == "deepseek" or os.getenv("COT_MODEL") == "deepseekr1":
                reasoning_chains = self.generate_chain_of_thought_by_deepseek(prompt, reference_code)
            
            result = {
                "id": idx,
                "prompt": prompt,
                "reference_code": reference_code,
                "reasoning_chains": reasoning_chains
            }
            results.append(result)
        
        # 保存结果
        if not os.path.exists(os.path.join(self.output_dir,os.getenv("COT_MODEL"))):
            os.makedirs(os.path.join(self.output_dir,os.getenv("COT_MODEL")))
            
        output_path = os.path.join(self.output_dir,os.getenv("COT_MODEL"), f"ds{self.limit}_cot.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    cot_util = CoTUtil()
    cot_util.process_and_save()

if __name__ == "__main__":
    main()