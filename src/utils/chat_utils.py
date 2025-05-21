import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from src.utils.models import GeminiLLM  # Replace with your actual import

def generate_gemini_response(query: str, retriever, gemini_api_key: str) -> str:
    prompt_template = PromptTemplate(
        input_variables=["query", "text", "images"],
        template=(
            "This is the query: {query}. Answer only using the provided context which consists of text and images.\n\n"
            "Text: {text}\nImages: {images}\n\n"
            "If there are no images, use only the text."
        )
    )

    llm = GeminiLLM(api_key=gemini_api_key, model_name='gemini-2.0-flash')
    chain = LLMChain(llm=llm, prompt=prompt_template)

    docs = retriever.get_relevant_documents(query)
    images, text = split_image_text_types(docs)

    response = chain.run(
        query=query,
        text=" ".join(text),
        images=" ".join(images)
    )
    return response


def split_image_text_types(docs):
    """
    Split image file paths and text content.

    Args:
        docs (list[str]): List of strings that are either text or image file paths.

    Returns:
        tuple: (image_paths, texts)
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_paths = []
    text = []

    for doc in docs:
        if isinstance(doc, str) and os.path.splitext(doc)[1].lower() in image_extensions:
            image_paths.append(doc)
        else:
            text.append(doc)

    return image_paths, text
