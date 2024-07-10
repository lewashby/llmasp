from openai import OpenAI

class LLMHandler:
    
    def __init__(self, model_name: str, server_url: str, api_key:str="ollama"):
        self.client = OpenAI(base_url=server_url, api_key=api_key)
        self.model = model_name

    
    def call(self, messages: list, temperature:float=0, stream: bool=False):
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
            stream=stream
        )

        if stream == True:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    return chunk.choices[0].delta.content
        else:
            return response.choices[0].message.content