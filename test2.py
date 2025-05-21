def test_ingestion(persist_path: str, query: str):
    from src.db.chroma.helpers import (
        create_embedding_model,
        create_vectorstore,
        create_retriever
    )

    # Load embedding + vectorstore + retriever
    embedding_model = create_embedding_model()
    vectorstore = create_vectorstore(embedding_model, persist_path)
    
    # Debug: Check collection stats
    print(f"Collection stats: {vectorstore._collection.count()} documents")
    
    # Try direct similarity search first to see if the vectors are working
    print("\nDirect similarity search:")
    direct_results = vectorstore.similarity_search(query, k=5)
    print(f"Direct search returned {len(direct_results)} results")
    
    for i, doc in enumerate(direct_results):
        print(f"[{i+1}] Document metadata: {doc.metadata}")
        print(f"[{i+1}] Content preview: {doc.page_content[:200]}...\n---")
    
    # Now try with the retriever
    retriever = create_retriever(vectorstore)
    
    # Run a query
    docs = retriever.get_relevant_documents(query)

    print(f"\nRetriever results:")
    print(f"🔍 Query: {query}")
    print(f"📄 Retrieved {len(docs)} document(s):\n")
    for i, doc in enumerate(docs):
        print(f"[{i+1}] Document metadata: {doc.metadata}")
        print(f"[{i+1}] Content preview: {doc.page_content[:200]}...\n---")

    return docs

# Try with different queries
print("\nTEST QUERY 1:")
docs = test_ingestion(persist_path="./chroma_db", query="What happened in 2018 regarding forest fires?")

# print("\nTEST QUERY 2:")
# docs = test_ingestion(persist_path="./chroma_db", query="forest fires")


from src.utils.chat_utils import generate_gemini_response

text = generate_gemini_response()

