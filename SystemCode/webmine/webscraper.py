import time, json, glob, os, sys
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import InvalidArgumentException
from selenium.webdriver.common.by import By

from constants.directories import SRC_DIR


settings = {"recentDestinations": [{"id": "Save as PDF",
                                    "origin": "local",
                                    "account": "",
                                    }],
            "selectedDestinationId": "Save as PDF",
            "isCssBackgroundEnabled": False,
            "isHeaderFooterEnabled" : False,
            "version": 2
            }

prefs = {'printing.print_preview_sticky_settings.appState': json.dumps(settings),
         'savefile.default_directory': f'{SRC_DIR}'
         }

chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
chrome_options.add_experimental_option('prefs', prefs) 
chrome_options.add_argument('--kiosk-printing')
chrome_options.add_argument("--log-level=3")


def get_pdf_document(weburl:str=None):
    '''
    Saves the user defined webpage into a PDF document 
    '''
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
        driver.implicitly_wait(2)
        driver.get(weburl)
        _ = driver.find_element(By.CLASS_NAME, 'printIcon')
    except Exception:    
        driver.quit()
    else:
        time.sleep(2)
        driver.execute_script('window.print();')
        driver.quit()
        time.sleep(1)
        
        pdf_files = glob.glob(os.path.join(SRC_DIR, "*.pdf"))
        # pdf_files.sort(key=os.path.getctime, reverse=True)
        latest_created_file = pdf_files[-1]
        savefile_name = datetime.strptime(weburl.split("=")[1], f'%d-%m-%Y').strftime(f'%Y%m%d')
        os.rename(latest_created_file, os.path.join(SRC_DIR, f'{savefile_name}.pdf'))
