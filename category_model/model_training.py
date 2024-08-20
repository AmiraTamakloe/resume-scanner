import pandas as pd

from category_model.data_preprocessing import clean_document
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
from category_model.data_preprocessing import clean_document, transform_pdf_to_text


class Categorizer(object):
    def __init__(self, df):
        self.df = self.cleanup_dataset(df)
        self.mapping = {}
        self.label = df['Category'].unique()
        self.labelize()
        self.create_mapping()

    def cleanup_dataset(self, df):
        df['Resume'] = df['Resume'].apply(lambda x: clean_document(x))
        return df

    def labelize(self):
        le = LabelEncoder()

        encoded_labels = le.fit(self.df['Category'])
        self.df['Category'] = le.transform(self.df['Category']) 
        self.df.Category.unique() 
        self.df['Encoded_Category'] = le.fit_transform(self.df['Category'])


    def create_mapping(self):
        unique_labels = self.df['Encoded_Category'].unique()
        for index, label_val in enumerate(unique_labels):
            category_name = self.label[index]
            self.mapping[label_val] = category_name

        # print("Encoded Label : Category Name")
        # for label_val, category_name in self.mapping.items():
        #     print(f"{label_val} : {category_name}")


class Model_Training(object):
    def __init__(self, df, categories_mapping):
        self.df = df
        self.categories_mapping = categories_mapping
        self.tfidf = TfidfVectorizer()
        self.clf = None
        self.required_text = []
        self.vectorize()
        self.train_model()

    def vectorize(self):
        self.tfidf = TfidfVectorizer(stop_words='english') 

        self.tfidf.fit(self.df['Resume']) 
        self.required_text = self.tfidf.transform(self.df['Resume']) 

        matrix = self.required_text.todense()
        dfMatrix = pd.DataFrame(matrix)
        # print(dfMatrix[:5])
        dfMatrix.to_csv('output.csv', index=False)
        return self.tfidf

    def train_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.required_text, self.df['Category'], random_state=0, test_size=0.2)    
        self.clf = OneVsRestClassifier(KNeighborsClassifier())
        self.clf.fit(X_train,Y_train) 



def predict_category(resume):

    df = pd.read_csv("./UpdatedResumeDataSet.csv")
    categorizer = Categorizer(df)
    trained_model = Model_Training(categorizer.df, categorizer.mapping)

    cv_text = transform_pdf_to_text(resume)
    cleaned_resume = clean_document(cv_text) 
    input_features = trained_model.tfidf.transform([cleaned_resume])
    prediction_id = trained_model.clf.predict(input_features)[0] 
    return categorizer.mapping[prediction_id]
