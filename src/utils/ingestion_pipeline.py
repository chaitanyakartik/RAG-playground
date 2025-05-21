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

def doc_ingestion_pipe(
    doc_path: str,
    persist_path: str = "./chroma_db",
):
    """
    Ingests a document (PDF or DOCX) and stores its content in a vector database.

    Args:
        doc_path (str): Path to the document.
        persist_path (str): Path to the directory where the vector database will be stored.

    Returns:
        None
    """
    # Create embedding model
    embedding_model = create_embedding_model()

    # Create vectorstore
    vectorstore = create_vectorstore(embedding_model, persist_path)
    retriever = create_retriever(vectorstore)

    # Extract text and images from the document
    if doc_path.endswith(".pdf"):
        text = extract_text_from_pdf(doc_path)
        images = extract_images_from_pdf(doc_path)
    elif doc_path.endswith(".docx"):
        text = extract_text_from_docx(doc_path)
        images = []
    else:
        raise ValueError("Unsupported file format. Please provide a PDF or DOCX file.")

    # Process images and text
    text_chunks = split_string_into_chunks(text)

    image_descriptions = [get_image_descriptions(img) for img in images]
    text_summaries = create_text_summaries(text_chunks, embedding_model)

    # Create documents for text and images
    text_summaries_docs, text_summaries_id = create_documents(text_summaries, "text_id")
    visual_descriptions_docs, visual_descriptions_id = create_documents(
        image_descriptions, "image_id"
    )

    retriever.vectorstore.add_documents(text_summaries_docs)
    retriever.docstore.mset(list(zip(text_summaries_id, text_chunks)))

    retriever.vectorstore.add_documents(visual_descriptions_docs)
    retriever.docstore.mset(list(zip(visual_descriptions_id, images)))