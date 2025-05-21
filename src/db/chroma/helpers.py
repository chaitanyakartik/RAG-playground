import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.storage import LocalFileStore


import uuid
from langchain.schema import Document

def create_embedding_model():
    return SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

def create_vectorstore(embedding_model, persist_path="./chroma_db"):
    return Chroma(
        collection_name="multi_modal_rag",
        embedding_function=embedding_model,
        persist_directory=persist_path
    )

def create_retriever(vectorstore, id_key="doc_id", persist_path="./chroma_docstore"):
    store = LocalFileStore(persist_path)
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
        search_kwargs={"k": 5}  # Increase number of results returned
    )

def create_documents(texts: list[str], id_key: str) -> tuple[list[Document], list[str]]:
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    documents = [
        Document(
            page_content=text, 
            metadata={
                id_key: doc_id,
                "source": "document"  # Add source metadata to help with filtering
            }
        )
        for text, doc_id in zip(texts, doc_ids)
    ]
    return documents, doc_ids
