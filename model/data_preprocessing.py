from PyPDF2 import PdfReader

import re

def clean_document(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('#\S+\s', ' ', resumeText)  # remove hashtags
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',' ', resumeText) #remove non-ascii characters
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText.lower()


def transform_pdf_to_text(file):
    cv_text = PdfReader(file).pages[0].extract_text().lower()
    return cv_text