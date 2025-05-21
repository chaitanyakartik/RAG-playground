from chat import doc_ingestion_pipe

doc_ingestion_pipe(
    doc_path="/Users/chaitanyakartik/Downloads/example.pdf",
    persist_path="./chroma_db"
)

