from src.utils.models import Gemini_Model

def generate_gemini_response(prompt: str):
    """
    Generates a response using the Gemini model.

    Args:
        prompt (str): The input prompt for the Gemini model.

    Returns:
        str: The generated response.
    """
    gemini_model = Gemini_Model(model_name="gemini-2.0-flash")
    response = gemini_model.generate(prompt)
    return response

if __name__ == "__main__":
    # Test the function with a sample prompt
    test_prompt = "Explain the significance of forest fires in 2018."
    print(f"Input Prompt: {test_prompt}")
    
    response = generate_gemini_response(test_prompt)
    print(f"Generated Response: {response}")
