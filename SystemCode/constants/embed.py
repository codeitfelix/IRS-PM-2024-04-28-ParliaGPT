import os
from constants.directories import MODEL_DIR
from chromadb.utils import embedding_functions

EMBEDDING_MODEL = os.path.join(MODEL_DIR, "thenlper_gte-small")
RERANKING_MODEL = os.path.join(MODEL_DIR, "BAAI_bge-reranker-base")
LLM_MODEL = os.path.join(MODEL_DIR, "TheBloke_Mistral-7B-OpenOrca-GPTQ")

SENT_EF = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

ALPHA = 0.75
CE_THRESHOLD = 0.01
