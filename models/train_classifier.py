import pandas as pd
import numpy as np
import sys  # Import sys for command line arguments
import re
import pickle  # for ML-model export as a pickle file
from sqlalchemy import create_engine  # to save the clean dataset into an sqlite database
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline  # For creating the pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # For text processing
from sklearn.multioutput import MultiOutputClassifier  # For multi-output classification
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest classifier
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import classification_report  # For evaluating the model
from sklearn.model_selection import GridSearchCV        #for using GridSearchCV, to improve model with Grid Search

nltk.download('wordnet')  # lexical database of English
nltk.download('punkt')  # tokenizer model split text into words
nltk.download('stopwords')  # list of common stopwords to remove them


import pandas as pd
import numpy as np
import nltk
import re
import sys  # Import sys for command line arguments
import pickle  # for ML-model export as a pickle file
from sqlalchemy import create_engine  # to save the clean dataset into an sqlite database
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline  # For creating the pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # For text processing
from sklearn.multioutput import MultiOutputClassifier  # For multi-output classification
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest classifier
from sklearn.model_selection import train_test_split  # For splitting the dataset
from sklearn.metrics import classification_report  # For evaluating the model
from sklearn.model_selection import GridSearchCV        #for using GridSearchCV, to improve model with Grid Search


# NLTK downloads
nltk.download('wordnet')  # lexical database of English
nltk.download('punkt')  # tokenizer model split text into words
nltk.download('stopwords')  # list of common stopwords to remove them


def load_data(database_filepath):
    """
    Loads the data from the database at the specified file path.

    Parameters:
    database_filepath (str): The file path to the database containing the messages and categories.
    
    Returns:
    tuple: A tuple consisting of:
        - X (pd.Series): The messages (feature).
        - y (pd.DataFrame): The categories (labels), split into separate columns.
        - category_names (Index): The names of the categories.
    """
    # Load the database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM DisasterResponses', engine)
        
    # Define features and label arrays
    X = df['message']  # Use df['message'] to access the 'message' column
    Y = df.iloc[:, 4:]  # Use iloc to select all rows and columns from index 4 onwards
    category_names = Y.columns  # This line is fine if you need the category names later
    
    return X, Y, category_names


def tokenize(text):    
    """
    Tokenizes and lemmatizes the input text.

    Parameters:
    text (str): The input text to be tokenized.

    Returns:
    list: A list of cleaned tokens.
    """
    # text processing: tokenization function to process data
    # Define a regex pattern to detect URLs
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'  
    text = re.sub(url_regex, "urlplaceholder", text)  # Replace URLs with a placeholder
    tokens = word_tokenize(text.lower())  # Normalize and tokenize text
    tokens = [w for w in tokens if w not in stopwords.words("english") and w.isalpha()]  # Remove stopwords
        
    lemmatizer = WordNetLemmatizer()  # Initiate lemmatizer
    clean_tokens = []
    for tok in tokens:  # Iterate through each token
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
     
    return clean_tokens    


def build_model():
    """
    Builds a machine learning pipeline for multi-output classification.

    This function creates a pipeline that processes text data using a 
    CountVectorizer to convert text into a matrix of token counts, 
    followed by a TfidfTransformer to transform the count matrix to 
    a normalized term-frequency or TF-IDF representation. Finally, 
    it applies a MultiOutputClassifier with a RandomForestClassifier 
    as the base estimator to handle multi-label classification tasks.

    Returns:
        Pipeline: A scikit-learn Pipeline object that encapsulates the 
        text processing and classification steps.
    """
    # Build a machine learning pipeline
    machine_learning_pipeline = Pipeline([
        ('cvect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators': [100],           # used here for shorter run time  
    #   'clf__estimator__n_estimators': [100, 200],      # better but runs longer     
        'clf__estimator__min_samples_split': [2, 3],
    }

    # Create grid search object
    model = GridSearchCV(machine_learning_pipeline, param_grid=parameters, return_train_score=True, verbose=2)
        
    #model = cv.best_estimator_
    
    return model
    

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the performance of a trained machine learning model on the provided test data.

    This function generates predictions using the provided model and compares them to the true labels 
    in the test dataset. It prints a classification report for each category, which includes metrics 
    such as precision, recall, and F1-score.

    Parameters:
    model: The trained machine learning model to be evaluated.
    X_test (pd.DataFrame or pd.Series): The input features for testing, which should match the format 
                                         used during training.
    Y_test (pd.DataFrame): The true labels for the test data, with each column representing a category.
    category_names (Index): The names of the categories corresponding to the columns in Y_test.

    Returns:
    None: This function does not return a value. It prints the classification report for each category 
          to the console.
    """
    # Generate predictions y_pred and print out classification_report for all 36 categories 
    y_pred = model.predict(X_test)
    
    # Iterate through each column and print the classification report
    for i in range(Y_test.shape[1]):  # Use the number of columns in Y_test
        print(i+1,")","#########################", Y_test.columns[i], "#########################")
        print(classification_report(Y_test.iloc[:, i], y_pred[:, i]))        
           

def save_model(model, model_filepath):
    """
    Saves the given machine learning model to a specified file path using pickle.

    Parameters:
    model: The machine learning model to be saved.
    model_filepath (str): The file path where the model will be saved, including the file name and extension.

    Returns:
    None: This function does not return a value. It saves the model to the specified file path.
    """
    # Export model as a pickle file
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)    
        

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
                   
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    