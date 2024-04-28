from tqdm import tqdm
from datetime import date
from dateutil.rrule import rrule, DAILY

from webmine.webscraper import get_pdf_document

#############################################
# To edit in the format of YEAR, MONTH, DAY #
#############################################
START_DATE = date(2023, 2, 21)
END_DATE = date(2024, 4, 19)
#############################################


for d in tqdm(rrule(DAILY, dtstart=START_DATE, until=END_DATE)):
    d = d.strftime(f'%d-%m-%Y')
    get_pdf_document(f'https://sprs.parl.gov.sg/search/#/fullreport?sittingdate={d}')
