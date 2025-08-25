import google.generativeai as genai
import json
import asyncio
from typing import Dict, List, Optional, Any
from config.settings import settings

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
        
    async def generate(self, system_prompt: str, user_message: str, 
                      response_format: str = "text") -> str:
        """Generate response using Gemini API"""
        try:
            full_prompt = f"{system_prompt}\n\nUser Query: {user_message}"
            
            if response_format == "json":
                full_prompt += "\n\nTrả về kết quả dưới dạng valid JSON."
            
            response = await asyncio.to_thread(
                self.model.generate_content, 
                full_prompt
            )
            
            response = response.text.strip()
            
            import re
            # Nếu response bọc trong code block markdown ```json ... ```
            if response.startswith("```"):
                response = re.sub(r"^```[a-zA-Z]*\n", "", response)
                response = response.rstrip("`").strip()
            
            return response
            
        except Exception as e:
            print(f"Gemini API Error: {e}")
            return ""
    
    async def generate_with_functions(self, system_prompt: str, user_message: str, 
                                    functions: List[Dict]) -> Dict:
        """Generate response with function calling"""
        try:
            # Format functions for Gemini
            function_descriptions = self._format_functions(functions)
            
            full_prompt = f"""
            {system_prompt}

            Available Functions:
            {function_descriptions}

            User Query: {user_message}

            Phân tích query và quyết định function nào cần gọi. Trả về JSON format:
            {{
                "functions_to_call": [
                    {{
                        "function_name": "function_name",
                        "parameters": {{...}}
                    }}
                ],
                "reasoning": "Giải thích tại sao chọn functions này"
            }}
            """
            
            response = await self.generate(full_prompt, "", "json")
            return json.loads(response)
            
        except json.JSONDecodeError:
            return {"functions_to_call": [], "reasoning": "Failed to parse response"}
        except Exception as e:
            print(f"Function calling error: {e}")
            return {"functions_to_call": [], "reasoning": f"Error: {e}"}
    
    def _format_functions(self, functions: List[Dict]) -> str:
        """Format functions for prompt"""
        formatted = []
        for func in functions:
            formatted.append(f"""
                Function: {func['name']}
                Description: {func['description']}
                Parameters: {json.dumps(func['parameters'], indent=2)}
            """)
        return "\n".join(formatted)
    
    async def embed_text(self, text: str) -> List[float]:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embedding_model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL_NAME,
            google_api_key=settings.GOOGLE_API_KEY
        )
        embedding = embedding_model.embed_query(text, output_dimensionality=512)
        return embedding
        