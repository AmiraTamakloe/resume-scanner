import pandas as pd

from model.data_preprocessing import clean_document
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle


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
        self.df['Category'] = le.transform(self.df['Category']) # assign a number to each category
        self.df.Category.unique() # check the unique numbers assigned to each category
        self.df['Encoded_Category'] = le.fit_transform(self.df['Category'])


    def create_mapping(self):
        # Retrieve the unique numerical labels assigned to each category
        unique_labels = self.df['Encoded_Category'].unique()
        # Create a mapping between encoded labels and their original category names
        for index, label_val in enumerate(unique_labels):
            category_name = self.label[index]
            self.mapping[label_val] = category_name

        # Print the mapping between encoded labels and category names
        print("Encoded Label : Category Name")
        for label_val, category_name in self.mapping.items():
            print(f"{label_val} : {category_name}")


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
        self.tfidf = TfidfVectorizer(stop_words='english') # remove stop words ex: the, a, an, is, are, etc. to keep only important words

        self.tfidf.fit(self.df['Resume']) # kind of a dictionary of all the words in the dataset without the stop words
        self.required_text = self.tfidf.transform(self.df['Resume'])  # converts the resumes in the dataset into a matrix of numbers (row = resume, column = word)

        matrix = self.required_text.todense()
        dfMatrix = pd.DataFrame(matrix)
        print(dfMatrix[:5])
        dfMatrix.to_csv('output.csv', index=False)
        return self.tfidf

    def train_model(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.required_text, self.df['Category'], random_state=0, test_size=0.2)    
        self.clf = OneVsRestClassifier(KNeighborsClassifier()) # one vs rest classifier to classify the resumes into multiple categories
        self.clf.fit(X_train,Y_train) # train the model using the training set
        ypred = self.clf.predict(X_test) # predict the categories of the resumes in the testing set
        print(accuracy_score(Y_test,ypred)) 
        Y_test_pred = self.clf.predict(X_train) # predict the categories of the resumes in the training set

        # pickle.dump(self.tfidf,open('tfidf.pkl','wb')) # save the tfidf model to use it later to convert the resumes into a matrix of numbers
        # pickle.dump(clf, open('clf.pkl', 'wb')) # save the model to use it later to predict the categories of the resumes


def predict_category(resume, tfidf):
    clf = pickle.load(open('clf.pkl', 'rb')) # Load the trained classifier
    cleaned_resume = clean_document(resume)  # Clean the input resume
    input_features = tfidf.transform([cleaned_resume]) # Transform the cleaned resume using the trained TfidfVectorizer into sparse matrix
    prediction_id = clf.predict(input_features)[0] # Make the prediction using the loaded classifier
    return category_mapping[prediction_id]
