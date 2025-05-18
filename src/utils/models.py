from dotenv import load_dotenv
import os
load_dotenv()

###Gemini API
import google.generativeai as genai

class Gemini_Model:
    """
    model list: gemini-2.0-flash, gemini-2.5-flash, gemini-2.5-pro
    """

    def __init__(self, model_name="gemini-2.0-turbo"):
        self.model_name = model_name

        Gemini_api_key = os.getenv("Gemini_api_key")
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
            model_name="gemini-2.0-flash",
            generation_config=generation_config,
        )

    def generate(self,prompt: str):
        response = self.model.generate_content([prompt])
        return response
    
            
#-------------------------------------------------------------------------------------
###Gemini with vertex
import vertexai
import vertexai.generative_models
import json

class Gemini_Model_VertexAI():

    def __init__(self,model_name="gemini-1.5-pro"):
        vertexai.init()
        self.model = vertexai.generative_models.GenerativeModel(
            model_name=model_name,
            safety_settings={
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
            },
        )
        
    def generate(self,prompt: str):
        generation_config={
            "temperature": 0.5,
            "response_mime_type": "application/json",
        }
        response=self.model.generate_content(
            [prompt],
            generation_config=generation_config
        )
        response = json.loads(response.candidates[0].content.parts[0].text)
        return response
    

class Gemini_Model_VertexAI_With_History():
    """
    This is done using chat sessions (gemini multiturn)
    Example chat history

    gemini_history = [
        {role: "user", parts: "What is the capital of France?"},
        {role: "model", parts: "The capital of France is Paris."},
    """
    def __init__(self,model_name="gemini-1.5-pro",chat_history=[]):
        vertexai.init()
        self.model = vertexai.generative_models.GenerativeModel(
            model_name=model_name,
            safety_settings={
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
                vertexai.generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: vertexai.generative_models.HarmBlockThreshold.BLOCK_NONE,
            },
        )
        self.current_chat_session = vertexai.generative_models.ChatSession(model=self.model,history=chat_history)
        
    def generate(self,prompt: str):
        generation_config={
            "temperature": 0.5,
            "response_mime_type": "application/json",
        }

        response=self.current_chat_session.send_message(
            [prompt],
            generation_config=generation_config
        )

        response = json.loads(response.candidates[0].content.parts[0].text)
        return response


#-------------------------------------------------------------------------------------
###OpenAI API
from openai import OpenAI

class OpenAI_Model():
    """
    Natively supports chat history
    Example chat history

    openai_history = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "system", "content": "The capital of France is Paris."},
    """
    def __init__(self,model_name="gpt-3.5-turbo",chat_history=[]):
        OpenAI_api_key = os.getenv("OpenAI_api_key")
        if not OpenAI_api_key:
            raise ValueError("OpenAI_api_key not found in environment variables.")
        
        self.client = OpenAI(api_key=OpenAI_api_key)
        self.model_name = model_name
        self.chat_history = chat_history

    def generate(self,prompt: str):
        user_message = {"role": "user", "content": prompt}
        chat_history = self.chat_history + [user_message]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=chat_history,  ##This is a list
            temperature=0.7,
        )
        result = response.choices[0].message.content

        return result
    

#-------------------------------------------------------------------------------------
###Anthropic with vertex
import anthropic

class Anthropic_Model():
    """
    Natively supports chat history
    example chat history
    anthropic_history = [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the capital of France?"
            }
        ]
    }, {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The capital of France is Paris."
            }
        ]
    }]
    """
    def __init__(self,model_name="claude-3-sonnet-20240229",chat_history=[]):
        Claude_api_key = os.getenv("Claude_api_key")
        self.client = anthropic.Anthropic(
            api_key=Claude_api_key,
        )
        self.model_name = model_name

    def generate(self,prompt: str):

        messages = self.chat_history.copy()
        # Add the new user message
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })

        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=0.3,
            messages=messages
        )

        return response.content.text