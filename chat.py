from src.db.chroma.helpers import (
    create_embedding_model,
    create_vectorstore,
    create_retriever,
    create_documents,
)

from src.utils.data_processing_helpers import (
    extract_text_from_pdf,
    extract_images_from_pdf,
    extract_text_from_docx,
    get_image_descriptions,
    split_string_into_chunks,
    create_text_summaries,
)
import logging
from langchain.schema import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def doc_ingestion_pipe(
    doc_path: str,
    persist_path: str = "./chroma_db",
    include_raw_text: bool = True,
    include_summaries: bool = True,
    include_images: bool = True,
):
    """
    Ingests a document (PDF or DOCX) and stores its content in a vector database.

    Args:
        doc_path (str): Path to the document.
        persist_path (str): Path to the directory where the vector database will be stored.
        include_raw_text (bool): Whether to include raw text chunks in the vectorstore.
        include_summaries (bool): Whether to include text summaries in the vectorstore.
        include_images (bool): Whether to include image descriptions in the vectorstore.

    Returns:
        None
    """
    logger.info(f"Starting ingestion for: {doc_path}")

    # Create embedding model
    logger.info("Creating embedding model...")
    embedding_model = create_embedding_model()

    # Create vectorstore
    logger.info("Creating vectorstore...")
    vectorstore = create_vectorstore(embedding_model, persist_path)

    # Create retriever
    logger.info("Creating retriever...")
    retriever = create_retriever(vectorstore)

    # Extract text and images
    logger.info("Extracting content from document...")
    if doc_path.endswith(".pdf"):
        text = extract_text_from_pdf(doc_path)
        images = extract_images_from_pdf(doc_path, output_dir=persist_path)
        logger.info(f"Extracted text and {len(images)} images from PDF.")
    elif doc_path.endswith(".docx"):
        text = extract_text_from_docx(doc_path)
        images = []
        logger.info("Extracted text from DOCX.")
    else:
        logger.error("Unsupported file format. Must be PDF or DOCX.")
        raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")

    # Process content
    logger.info("Processing text chunks...")
    text_chunks = split_string_into_chunks(text)
    logger.info(f"Split text into {len(text_chunks)} chunks.")

    # Choose whether to include raw text, summaries, or both based on parameters
    if include_raw_text:
        # Store the raw text chunks directly in the vector store
        raw_text_docs = [Document(page_content=chunk, 
                                 metadata={"source": doc_path, "type": "text", "chunk_id": f"chunk_{i}"}) 
                        for i, chunk in enumerate(text_chunks)]
        
        # Add the raw chunks directly to the vectorstore
        logger.info("Adding raw text chunks to vectorstore...")
        vectorstore.add_documents(raw_text_docs)
    
    if include_summaries:
        # Process summaries
        logger.info("Generating text summaries...")
        text_summaries = create_text_summaries(text_chunks, embedding_model)
        
        # Create documents with summaries
        text_summaries_docs = [Document(page_content=summary, 
                                      metadata={"source": doc_path, "type": "summary", "chunk_id": f"summary_{i}"}) 
                             for i, summary in enumerate(text_summaries)]
        
        # Store in vectorstore
        logger.info("Adding text summaries to vectorstore...")
        vectorstore.add_documents(text_summaries_docs)
    
    if include_images and images:
        # Process images
        logger.info("Generating image descriptions...")
        image_descriptions = [get_image_descriptions(img) for img in images]
        
        # Create documents with image descriptions
        visual_descriptions_docs = [Document(page_content=desc, 
                                          metadata={"source": img_path, "type": "image", "image_id": f"image_{i}"}) 
                                  for i, (desc, img_path) in enumerate(zip(image_descriptions, images))]
        
        # Store in vectorstore
        logger.info("Adding image descriptions to vectorstore...")
        vectorstore.add_documents(visual_descriptions_docs)
    
    vectorstore.persist()
    logger.info("Ingestion completed successfully.")
