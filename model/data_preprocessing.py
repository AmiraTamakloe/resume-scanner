from PyPDF2 import PdfReader

import re

def clean_document(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText) 
    resumeText = re.sub('@\S+', '  ', resumeText) 
    resumeText = re.sub('#\S+\s', ' ', resumeText) 
    resumeText = re.sub('RT|cc', ' ', resumeText) 
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText) 
    resumeText = re.sub(r'[^\x00-\x7f]',' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  
    return resumeText.lower()


def transform_pdf_to_text(file):
    cv_text = PdfReader(file).pages[0].extract_text().lower()
    return cv_text