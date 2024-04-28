import os
from tqdm import tqdm
import pandas as pd

import spacy
import duckdb
import chromadb
from chromadb.config import Settings

from unstructured.partition.auto import partition_pdf
from unstructured.staging.base import convert_to_dataframe
from unstructured.chunking.title import chunk_by_title

from constants.directories import SRC_DIR, VECTOR_DIR, RELATIONAL_DIR
from constants.embed import SENT_EF


# nlp = spacy.load("en_core_web_sm")

# connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))

# pdf_files = sorted(os.listdir(SRC_DIR))
# data_df = pd.DataFrame()

# # Extracts chunks from files and insert into database
# for file in tqdm(pdf_files):
#     elements = partition_pdf(os.path.join(SRC_DIR, file),
#                              strategy='hi_res',
#                              hi_res_model_name='yolox',
#                              infer_table_structure=False
#                              )
#     e_df = convert_to_dataframe(elements)
#     # idx = e_df[e_df['text'].isin(['ORAL ANSWERS TO QUESTIONS', "PRESIDENT'S ADDRESS"])].index.tolist()[0]
    
#     chunks = chunk_by_title(elements,
#                             new_after_n_chars=1500,
#                             max_characters=2000,
#                             overlap=200)
    
#     page_list = []
#     for chunk in chunks:
#         page_list.append(', '.join([str(el) for el in sorted(list(set([e.metadata.page_number for e in chunk.metadata.orig_elements])))]))
        
#     chunks_df = convert_to_dataframe(chunks)
#     chunks_df['page_number'] = page_list

#     # Ingest data into vectorDB (ChromaDB)
#     for row in chunks_df.iterrows():
#         collection.add(
#             documents=row[1]['text'],
#             metadatas=[{'filename': row[1]['filename'],
#                         "page_number": row[1]['page_number']}],
#             ids=[f'id_{collection.count() + 1}']
#             )
    
#     # Ingest data as csv
#     data_df = pd.concat([data_df, chunks_df])
#     data_df.to_csv(os.path.join(RELATIONAL_DIR, 'data.csv'))


# # Ingest data into relationalDG (DuckDB)
# connection.sql("CREATE TABLE parliament_debate AS SELECT text, filename, page_number, id FROM data_df")
# connection.sql('ALTER TABLE parliament_debate ADD COLUMN lemmas VARCHAR')

# # Perform NLP on extracted text with spacy
# for idx, sent in enumerate(tqdm(data_df['text'])):
#     sent = nlp(sent[0].lower())
#     filtered_words = ' '.join([token.lemma_.lower() for token in sent if token.text.isalnum() and not token.is_space and not token.is_stop and not token.is_punct and not token.is_digit])
#     connection.sql(f"UPDATE parliament_debate SET lemmas = '{filtered_words}' WHERE id = 'id_{idx+1}'")

# connection.close()


def chunk_data(dir:str=None):
    '''
    Converts PDF data into data chunks and adds it into a
    vector database and a flat file
    '''
    print("Creating Chunks from PDF Files...")
    pdf_files = sorted(os.listdir(dir))
    for file in tqdm(pdf_files):
        elements = partition_pdf(os.path.join(dir, file),
                                 strategy='hi_res',
                                 hi_res_model_name='yolox',
                                 infer_table_structure=False
                                 )
        
        chunks = chunk_by_title(elements,
                                new_after_n_chars=1500,
                                max_characters=2000,
                                overlap=200
                                )
        
        page_list = []
        for chunk in chunks:
            page_list.append(', '.join([str(el) for el in sorted(list(set([e.metadata.page_number for e in chunk.metadata.orig_elements])))]))

        chunks_df = convert_to_dataframe(chunks)
        chunks_df['page_number'] = page_list
        
        ingest_vdb(chunks_df)
        ingest_csv(chunks_df)
    
    print("Chunking Complete!")
    print("Creating Relational Database...")
    ingest_rdb()
    print("Relational Database Created!")


def ingest_vdb(chunks_df:pd.DataFrame):
    '''
    Creates a persistent vector database (if not present),
    Adds chunked data into vector database (if present)
    '''
    client = chromadb.PersistentClient(VECTOR_DIR, Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection('parliament_debate', embedding_function=SENT_EF)
    
    for row in chunks_df.iterrows():
        collection.add(
            documents=row[1]['text'],
            metadatas=[{'filename': row[1]['filename'],
                        "page_number": row[1]['page_number']}],
            ids=[f'id_{collection.count() + 1}']
            )    
    

def ingest_csv(chunks_df:pd.DataFrame):
    '''
    Adds chunked data into a csv flatfile
    '''
    data_dir = os.path.join(RELATIONAL_DIR, 'data.csv')
    
    if os.path.isdir(data_dir):
        data_df = pd.read_csv(data_dir)
        data_df = pd.concat([data_df, chunks_df])
        data_df.to_csv(data_dir, index=False)
        
    else:
        chunks_df.to_csv(data_dir, index=False)


def ingest_rdb():
    '''
    Converts csv flatfile into a persistent relational database
    '''
    nlp = spacy.load("en_core_web_sm")
    
    data_df = pd.read_csv(os.path.join(RELATIONAL_DIR, 'data.csv'))
    data_df['id'] = [f'id_{i}' for i in range(len(data_df))]
    
    lemmas = []
    for sent in tqdm(data_df['text']):
        sent = nlp(sent.lower())
        filtered_words = ' '.join([token.lemma_.lower() for token in sent if token.text.isalnum() and not token.is_space and not token.is_stop and not token.is_punct and not token.is_digit])
        lemmas.append(filtered_words)
    
    data_df['lemmas'] = lemmas
    
    connection = duckdb.connect(os.path.join(RELATIONAL_DIR, 'data.db'))
    connection.sql("CREATE TABLE parliament_debate AS SELECT text, filename, page_number, id, lemmas FROM data_df")
    connection.close()
