import pandas as pd
from model.data_preprocessing import clean_document, transform_pdf_to_text
from model.model_training import Categorizer, Model_Training

def predict_category(resume):

    df = pd.read_csv("./UpdatedResumeDataSet.csv")
    categorizer = Categorizer(df)
    trained_model = Model_Training(categorizer.df, categorizer.mapping)

    cv_text = transform_pdf_to_text(resume)
    cleaned_resume = clean_document(cv_text)  # Clean the input resume
    input_features = trained_model.tfidf.transform([cleaned_resume]) # Transform the cleaned resume using the trained TfidfVectorizer into sparse matrix
    prediction_id = trained_model.clf.predict(input_features)[0] # Make the prediction using the loaded classifier
    return categorizer.mapping[prediction_id]

