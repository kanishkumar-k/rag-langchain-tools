from langchain.tools import Tool
from langchain_core.tools import tool
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

# Configuration
DOCUMENT_PATH = './data/sample.txt'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CONNECTION_STRING = "postgresql+psycopg://kanishkumar:Kanish%40123@localhost:5432/user-db"
COLLECTION_NAME = "my_docs"

vector_store = None

def build_or_load_vector_store():
    global vector_store

    ## Step-3 Embeddings 

    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)

    ## Step-4 Vector Store

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )

    test_result = vector_store.similarity_search("test", k=1)
    if not test_result:

        ## Step-1 Loading document 

        loader = TextLoader(DOCUMENT_PATH, encoding='utf-8')
        documents = loader.load()
        
        ## Step-2 Split into RecursiveCharacter TextSplit

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = splitter.split_documents(documents)

        ids = [str(i) for i in range(len(texts))]
        vector_store.add_documents(documents=texts, ids=ids)

    return vector_store

def run_rag_pipeline(query: str = "") -> str:
    
    ## Step-5 Query

    if not query:
        return "No query provided."

    store = build_or_load_vector_store()
    results = store.similarity_search(query=query, k=5)

    if not results:
        return "Sorry, I couldn't find anything relevant."

    ## Step-6 Retrival

    output = ""
    for i, doc in enumerate(results):
        output += f"[Result {i+1}]\n{doc.page_content}\n\n"
    return output.strip()


rag_tool = Tool.from_function(
    func=run_rag_pipeline,
    name="RAGSearchTool",
    description="Searches sample.txt using vector similarity. Provide 1 line query to retrieve matching content."
)
