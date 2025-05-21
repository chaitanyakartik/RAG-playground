from langchain_core.language_models.llms import LLM
from typing import Any, Dict, Iterator, List, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import google.generativeai as genai


class GeminiLLM(LLM):
    """A custom wrapper for the Gemini model."""
    api_key: str
    model_name: str

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input."""

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        response = model.generate_content(prompt)
        return response.text  # Adjust based on actual response structure

    @property
    def _llm_type(self) -> str:
        return "gemini"
    
#-------------------------------------------------------------------------------------
###Gemini API
import google.generativeai as genai
import os

class Gemini_Model:
    """
    model list: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
    """

    def __init__(self, model_name="gemini-2.0-flash"):
        self.model_name = model_name

        Gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not Gemini_api_key:
            raise ValueError("Gemini_api_key not found in environment variables.")
        
        genai.configure(api_key=Gemini_api_key)

        generation_config = {
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 64,
            "max_output_tokens": 8192,
        }
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
        )

    def generate(self,prompt: str):
        response = self.model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text