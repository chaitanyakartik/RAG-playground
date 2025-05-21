import pdfplumber
import fitz  # PyMuPDF
import os
import os
from docx import Document
from PIL import Image

from src.utils.prompts import DESCRIBE_IMAGE_PROMPT,SUMMARIZE_TEXT_PROMPT
from src.utils.models import Gemini_Model


def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text


def extract_images_from_pdf(pdf_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    images_list = []  # List to store image file paths

    # Loop through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load each page
        image_list = page.get_images(full=True)  # Extract images

        # Loop through all the images on the page
        for img_index, img in enumerate(image_list):
            xref = img[0]  # Image index reference
            base_image = pdf_document.extract_image(xref)  # Extract the image
            image_bytes = base_image["image"]  # Image data
            image_ext = base_image["ext"]  # Image file extension

            # Save the image to the specified directory
            image_filename = os.path.join(output_dir, f"image_page{page_num + 1}_{img_index + 1}.{image_ext}")
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)

            images_list.append(image_filename)  # Add file path to the list

            print(f"Extracted: {image_filename}")

    return images_list  # Return the list of images


def extract_text_from_docx(docx_file):
    document = Document(docx_file)
    text = ''
    for paragraph in document.paragraphs:
        text += paragraph.text + '\n'
    return text

def get_image_descriptions(image_path: str, history: list[dict]=[]):
    #Get image description
    template= DESCRIBE_IMAGE_PROMPT["template"]
    image_model = Gemini_Model(model_name="gemini-2.0-flash")

    
    img = Image.open(image_path)
    input_object=[template, img]
    image_response=image_model.generate(input_object)
    return image_response

def split_string_into_chunks(long_text, max_chunk_size=3000):
    # Create a list to hold the chunks
    chunks = []

    # Iterate over the text, slicing it into chunks
    for start_index in range(0, len(long_text), max_chunk_size):
        end_index = start_index + max_chunk_size
        chunk = long_text[start_index:end_index]  # Extract the chunk
        chunks.append(chunk)  # Add the chunk to the list

    return chunks

def create_text_summaries(text_chunks, model):
    text_summaries = []
    query = SUMMARIZE_TEXT_PROMPT["template"]

    text_model = Gemini_Model(model_name="gemini-2.0-flash")

    for text in text_chunks:
        result = text_model.generate(text + query)
        text_summaries.append(text)
    return text_summaries 

def another_function():
    pass