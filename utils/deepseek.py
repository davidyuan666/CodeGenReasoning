# Please install OpenAI SDK first: `pip3 install openai`

import os
from openai import OpenAI

class DeepSeek:
    def __init__(self, api_key=None, base_url=os.getenv("DEEPSEEK_BASE_URL")):
        """
        初始化DeepSeek客户端
        
        参数:
            api_key: DeepSeek API密钥，如果为None，则从环境变量DEEPSEEK_API_KEY获取
            base_url: DeepSeek API的基础URL
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API密钥未提供，请设置DEEPSEEK_API_KEY环境变量或在初始化时提供")
        
        self.client = OpenAI(api_key=self.api_key, base_url=base_url)
    
    def chat(self, prompt, system_prompt="You are a helpful assistant",  stream=False):
        """
        使用DeepSeek API进行聊天
        
        参数:
            prompt: 用户输入的提示
            system_prompt: 系统提示
            model: 使用的模型名称
                  - 'deepseek-chat': DeepSeek-V3 (默认)，通用对话模型
                  - 'deepseek-reasoner': DeepSeek-R1，专门用于推理任务的模型
            stream: 是否流式输出
            
        返回:
            聊天回复的内容
        """
        response = self.client.chat.completions.create(
            model=os.getenv("DEEPSEEK_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=stream
        )
        
        if stream:
            return response
        else:
            return response.choices[0].message.content

# 使用示例
if __name__ == "__main__":
    deepseek = DeepSeek()  # 默认从环境变量DEEPSEEK_API_KEY获取API密钥
    
    # 使用DeepSeek-V3（通用对话模型）
    response_v3 = deepseek.chat("Hello", model="deepseek-chat")
    print("DeepSeek-V3 回复:", response_v3)
    
    # 使用DeepSeek-R1（推理模型）
    response_r1 = deepseek.chat("请解决这个数学问题: 如果x^2 + 5x + 6 = 0，求x的值", model="deepseek-reasoner")
    print("DeepSeek-R1 回复:", response_r1)