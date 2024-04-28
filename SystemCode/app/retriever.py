import os, re
import duckdb
import spacy
import numpy as np

from rank_bm25 import BM25Plus
from sklearn.preprocessing import normalize
from sentence_transformers import CrossEncoder

import chromadb
from chromadb.config import Settings

from constants.embed import SENT_EF, ALPHA, CE_THRESHOLD, RERANKING_MODEL
from constants.directories import RELATIONAL_DIR, VECTOR_DIR


class FusedRetriever:
    def __init__(self, query:str, n_results:int=5):
        self.tokenized_query = None
        self.sparse_res = None
        self.dense_res = None
        self.hybrid_res = None
        self.ce_list = None
        
        self.__lemmatize_query(query)
        self.__retrieve_sparse_data(self.tokenized_query, n_results)
        self.__retrieve_dense_data(query, n_results)
        self.__retrieve_hybrid(self.dense_res, self.sparse_res)
        self.__rerank(query, self.hybrid_res)
    
    
    def __lemmatize_query(self, query):
        nlp = spacy.load("en_core_web_sm")
        self.tokenized_query = [token.lemma_.lower() for token in nlp(query.lower()) if token.text.isalnum() and not token.is_space and not token.is_stop and not token.is_punct and not token.is_digit]


    def __retrieve_sparse_data(self, tokenized_query, num_results):
        connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))
        db_info = connection.sql("SELECT id, lemmas FROM parliament_debate").fetchall()
        tokenized_corpus = [row[-1].split(" ") for row in db_info]
        bm25 = BM25Plus(tokenized_corpus)
        
        corpus_ids = [row[0] for row in db_info]
        top_n_docs = bm25.get_top_n(tokenized_query, corpus_ids, n=num_results) # Get TOP N corpus_ids
        doc_scores = normalize([bm25.get_scores(tokenized_query)], norm='max')[0].tolist() # Normalize results
        
        scores = [doc_scores[int(re.sub('id_', '', doc))-1] for doc in top_n_docs]
        texts = [connection.sql(f"SELECT text FROM parliament_debate WHERE id = '{doc}'").fetchone()[0] for doc in top_n_docs]

        self.sparse_res = {'ids': [top_n_docs],
                           'scores': [scores],
                           'documents': [texts]
                           }
        
        connection.close()
        
        
    def __retrieve_dense_data(self, query, num_results):
        client = chromadb.PersistentClient(VECTOR_DIR, Settings(anonymized_telemetry=False))
        collection = client.get_collection('parliament_debate', embedding_function=SENT_EF)
        self.dense_res = collection.query(query_texts=[query], n_results=num_results)
        
        
    def __retrieve_hybrid(self, dense_res, sparse_res):
        all_ids = list(set(dense_res['ids'][0] + sparse_res['ids'][0]))

        hyb_list = []
        for id in all_ids:
            if id in sparse_res['ids'][0]:
                idx = sparse_res['ids'][0].index(id)
                sparse_score = sparse_res['scores'][0][idx]
            else:
                sparse_score = 0
        
            if id in dense_res['ids'][0]:
                idx = dense_res['ids'][0].index(id)
                dense_score = 1 - dense_res['distances'][0][idx]
            else:
                dense_score = 0

            hybrid_score = ((1 - ALPHA) * sparse_score) + (ALPHA * dense_score)
    
            if hybrid_score > 0.62:
                hyb_list.append([id, hybrid_score])

        hyb_list.sort(key=lambda x: x[1], reverse=True)
        self.hybrid_res = hyb_list


    def __rerank(self, query, hyb_list):
        connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))
        cross_encoder = CrossEncoder(RERANKING_MODEL, max_length=512, device='cpu')
        text_pairs = [[query, connection.sql(f"SELECT text FROM parliament_debate WHERE id = '{id}'").fetchone()[0]] for (id, _) in hyb_list]
        ce_scores = cross_encoder.predict(text_pairs,
                                          batch_size=1,
                                          show_progress_bar=True,
                                          ).tolist()
        ce_list = [(hyb_list[i][0], ce_scores[i]) for i in range(len(ce_scores)) if ce_scores[i] > CE_THRESHOLD]
        self.ce_list = ce_list
        connection.close()
