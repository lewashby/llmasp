from openai import OpenAI
from typing import Optional

class LLMHandler:
    
    def __init__(self, model_name: str="ollama", server_url: str="http://localhost:11434/v1", api_key:str="ollama", timeout:int=3600, max_retries:int=4):
        self.client = OpenAI(base_url=server_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model_name

    
    def call(self, messages: list, temperature:float=0, stream: bool=False, max_tokens:Optional[int] = None):
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            stream=stream,
            max_tokens=max_tokens
        )

        if stream == True:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    return chunk.choices[0].delta.content
        else:
            completion = response.choices[0].message.content
            meta = response.usage
            return completion, meta