import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.config import settings
from backend.db.database import Database
from backend.rag.ingestion import IngestionPipeline
from backend.rag.engine import AdvancedRAGEngine

def test_flow():
    db = Database(db_path=settings.db_path)
    
    print("--- 1. Testing Document Ingestion ---")
    pipeline = IngestionPipeline()
    sample_path = ROOT / "README.md"
    print(f"Indexing sample document: {sample_path}")
    
    num_chunks = pipeline.ingest_file(sample_path)
    print(f"Document indexed successfully! Chunks created: {num_chunks}")

    print("\n--- 2. Testing RAG Engine Query ---")
    engine = AdvancedRAGEngine(db=db)
    question = "What is this project about and what technologies does it use?"
    print(f"Querying: {question}")

    user = db.get_user_by_username("test_user") or db.create_user("test_user")
    conversation = db.create_conversation(user["user_id"], "Test Conversation")
    
    response = engine.chat(
        user_id=user["user_id"],
        conversation_id=conversation["conversation_id"],
        message=question
    )
    print("\n--- Answer ---")
    print(response.get("reply"))
    print("\n--- Sources ---")
    print(response.get("sources"))

if __name__ == "__main__":
    test_flow()
