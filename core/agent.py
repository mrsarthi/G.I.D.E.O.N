import ollama
from config.settings import MODEL, SYSTEM_PROMPT
from core.registry import TOOLS, TOOL_REGISTRY

class GideonAgent:
    def __init__(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def chat(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        
        response = ollama.chat(
            model=MODEL,
            messages=self.messages,
            tools=TOOLS,
            options={'temperature': 0}
        )
        
        if response["message"].get("tool_calls"):
            return self._handle_tools(response)
        
        self.messages.append(response["message"])
        return response["message"]["content"]

    def _handle_tools(self, response):
        while response["message"].get("tool_calls"):
            for tool_call in response["message"]["tool_calls"]:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]
                
                if name in TOOL_REGISTRY:
                    result = TOOL_REGISTRY[name](**args)
                else:
                    result = f"Unknown tool: {name}"
                
                self.messages.append({"role": "tool", "content": str(result)})
            
            response = ollama.chat(
                model=MODEL, 
                messages=self.messages, 
                tools=TOOLS,
                options={'temperature': 0}
            )
            
        self.messages.append(response["message"])
        return response["message"]["content"]

    def stream_chat(self, user_input):
        content = self.chat(user_input)
        for char in content:
            yield char
