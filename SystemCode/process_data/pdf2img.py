import os
import fitz
from tqdm import tqdm

from constants.directories import SRC_DIR, IMG_DIR


def extract_images(dir:str=None):
    '''
    Extracts pages from PDF files as an image
    '''
    files = sorted(os.listdir(dir))
    
    for file in tqdm(files):
        file_folder_name = os.path.join(IMG_DIR, f'{os.path.splitext(file)[0]}')

        if not os.path.isdir(file_folder_name):
            os.mkdir(file_folder_name)
        else:
            continue
        
        # Generate images into folder
        doc = fitz.open(os.path.join(os.path.join(dir, file)))
        for page in doc:
            pix = page.get_pixmap()
            pix.save(os.path.join(file_folder_name, f'{page.number + 1}.jpg'))
