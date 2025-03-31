import json
import os
from openai import OpenAI
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

class CoTUtil:
    def __init__(self, data_path: str = "data/ds1000.json", output_dir: str = "chains"):
        self.data_path = data_path
        self.output_dir = output_dir
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
    def read_dataset(self) -> List[Dict]:
        """读取数据集文件"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_chain_of_thought(self, prompt: str, reference_code: str) -> List[Dict]:
        """生成思维链推理过程"""
        messages = [
            {"role": "system", "content": "You are a helpful programming assistant. Please explain your thinking step by step."},
            {"role": "user", "content": f"Problem: {prompt}\nReference solution: {reference_code}\nPlease explain how to solve this step by step."}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # 或其他适合的模型
            messages=messages,
            temperature=0.7
        )
        
        # 解析响应中的推理步骤
        reasoning = response.choices[0].message.content
        steps = self._parse_reasoning_steps(reasoning)
        
        return steps
    
    def _parse_reasoning_steps(self, reasoning: str) -> List[Dict]:
        """将推理过程解析为步骤列表"""
        # 这里可以根据实际输出格式进行调整
        steps = []
        current_step = ""
        
        for line in reasoning.split('\n'):
            if line.startswith('Step '):
                if current_step:
                    steps.append({"step": current_step})
                current_step = line
            else:
                current_step += '\n' + line if current_step else line
                
        if current_step:
            steps.append({"step": current_step})
            
        return steps
    
    def process_and_save(self):
        """处理数据集并保存推理链结果"""
        dataset = self.read_dataset()
        results = []
        
        for idx, item in enumerate(dataset):
            if idx > 10:
                break
            prompt = item['prompt']
            reference_code = item['reference_code']
            
            # 生成推理链
            reasoning_chain = self.generate_chain_of_thought(prompt, reference_code)
            
            result = {
                "id": idx,
                "prompt": prompt,
                "reference_code": reference_code,
                "reasoning_chain": reasoning_chain
            }
            results.append(result)
        
        # 保存结果
        output_path = os.path.join(self.output_dir, "ds1000_cot.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

def main():
    cot_util = CoTUtil()
    cot_util.process_and_save()

if __name__ == "__main__":
    main()