from process_data.create_dbs import chunk_data
from process_data.pdf2img import extract_images

from constants.directories import SRC_DIR

extract_images(SRC_DIR)
chunk_data(SRC_DIR)
