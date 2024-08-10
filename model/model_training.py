import pandas as pd

from data_preprocessing import clean_document
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
import pickle
    

def cleanup_dataset(df):
    df['Resume'] = df['Resume'].apply(lambda x: clean_document(x))

def label_category(df):
    le = LabelEncoder()

    label = df['Category'].unique() # unique categories
    encoded_labels = le.fit(df['Category'])
    df['Category'] = le.transform(df['Category']) # assign a number to each category
    df.Category.unique() # check the unique numbers assigned to each category
    df['Encoded_Category'] = le.fit_transform(df['Category'])

    return {'df': df, 'label': label}


def create_categories(df, label):
    # Retrieve the unique numerical labels assigned to each category
    unique_labels = df['Encoded_Category'].unique()
    # Create a mapping between encoded labels and their original category names
    category_mapping = {}
    for index, label_val in enumerate(unique_labels):
        category_name = label[index]
        category_mapping[label_val] = category_name

    # Print the mapping between encoded labels and category names
    print("Encoded Label : Category Name")
    for label_val, category_name in category_mapping.items():
        print(f"{label_val} : {category_name}")

def vectorize(df):
    tfidf = TfidfVectorizer(stop_words='english') # remove stop words ex: the, a, an, is, are, etc. to keep only important words

    tfidf.fit(df['Resume']) # kind of a dictionary of all the words in the dataset without the stop words
    requiredText = tfidf.transform(df['Resume'])  # converts the resumes in the dataset into a matrix of numbers (row = resume, column = word)

    matrix = requiredText.todense()
    dfMatrix = pd.DataFrame(matrix)
    print(dfMatrix[:5])
    dfMatrix.to_csv('output.csv', index=False)

    return tfidf

def train_model(required_text, tfidf):
    X_train, X_test, Y_train, Y_test = train_test_split(required_text, df['Category'], random_state=0, test_size=0.2)    
    clf = OneVsRestClassifier(KNeighborsClassifier()) # one vs rest classifier to classify the resumes into multiple categories
    clf.fit(X_train,Y_train) # train the model using the training set
    ypred = clf.predict(X_test) # predict the categories of the resumes in the testing set
    print(accuracy_score(Y_test,ypred)) 
    Y_test_pred = clf.predict(X_train) # predict the categories of the resumes in the training set

    pickle.dump(tfidf,open('tfidf.pkl','wb')) # save the tfidf model to use it later to convert the resumes into a matrix of numbers
    pickle.dump(clf, open('clf.pkl', 'wb')) # save the model to use it later to predict the categories of the resumes

def predict_category(resume, tfidf):
    clf = pickle.load(open('clf.pkl', 'rb')) # Load the trained classifier
    cleaned_resume = clean_document(resume)  # Clean the input resume
    input_features = tfidf.transform([cleaned_resume]) # Transform the cleaned resume using the trained TfidfVectorizer into sparse matrix
    prediction_id = clf.predict(input_features)[0] # Make the prediction using the loaded classifier
    return category_mapping[prediction_id]
    

df = pd.read_csv("./UpdatedResumeDataSet.csv")