import os
from huggingface_hub import snapshot_download
from constants.directories import MODEL_DIR

snapshot_download(repo_id='TheBloke/Mistral-7B-OpenOrca-GPTQ',
                  revision='main',
                  local_dir=os.path.join(MODEL_DIR, 'TheBloke_Mistral-7B-OpenOrca-GPTQ'),
                  local_dir_use_symlinks=False
                  )

snapshot_download(repo_id='thenlper/gte-small',
                  local_dir=os.path.join(MODEL_DIR, 'thenlper_gte-small'),
                  local_dir_use_symlinks=False
                  )

snapshot_download(repo_id='BAAI/bge-reranker-base',
                  local_dir=os.path.join(MODEL_DIR, 'BAAI_bge-reranker-base'),
                  local_dir_use_symlinks=False
                  )
