#!/usr/bin/env python
# coding: utf-8

# ### ETL Pipeline Preparation

# In[1]:


# import libraries
import pandas as pd
from sqlalchemy import create_engine # to save the clean dataset into an sqlite database


# In[2]:


# 1. load datasets from csv files.

# load messages dataset
messages = pd.read_csv(r"messages.csv") 
# messages.head()


# load categories dataset
categories = pd.read_csv(r"categories.csv")
# categories.head()



# 2. Merge datasets

# merge datasets messages and categories datasets using the common id
df = pd.merge(messages, categories, left_on='id', right_on='id', how='inner') 
# df.head()



# 3. Split categories into separate category columns

# create a dataframe of the 36 individual category columns 
# split the 'categories' column into separate columns

categories = df["categories"].str.split(';', expand=True)
# categories.head()

 
## select the first row of the categories dataframe
#row = categories[0:1]
#
## use this row to extract a list of new column names for categories.
## one way is to apply a lambda function that takes everything 
## up to the second to last character of each string with slicing
#category_col = row.apply(lambda x: x.str[:-2]).values.tolist()
#print(category_col)
#

# Better Alternative: 
# Extract new column names directly from the first row
categories.columns = categories.iloc[0].str[:-2].values

# Drop the first row since it was used for column names
categories = categories[1:]

# Display the first few rows of the categories DataFrame
# categories.head()



## rename the columns of `categories`
#categories.columns = category_col
#categories.head()
#



# 4. Convert category values to just numbers 0 or 1.

for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].str[-1]
    
    # convert column from string to numeric
    categories[column] = pd.to_numeric(categories[column])
# categories.head()

## Better Alternative
## Optimized extraction of the last character and conversion to numeric
#categories = categories.apply(lambda x: pd.to_numeric(x.str[-1]))
#
## Display the first few rows of the updated DataFrame
#categories.head()



# 5. Replace categories column in df with new category columns.

# drop the original categories column from `df`

df.drop(['categories'], axis=1, inplace = True)
#df.head()



# concatenate the original dataframe with the new `categories` dataframe
df = pd.concat([df, categories], axis=1)
# df.head()



# 6. Remove duplicates.

# check number of duplicates
# df.duplicated().sum()

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
#df.duplicated().sum()



# 7. Save the clean dataset into an sqlite database.

engine = create_engine('sqlite:///DisasterResponseProject.db')
df.to_sql('DisasterResponses', engine, index=False, if_exists='replace')


# ### Machine Learning Pipeline Preparation
# 

# In[27]:


# import libraries

# 1) read SQL
import pandas as pd
from sqlalchemy import create_engine

# 2) tokenization function 
import nltk
import re
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#3) machine learning pipeline
from sklearn.pipeline import Pipeline  # For creating the pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # For text processing
from sklearn.multioutput import MultiOutputClassifier  # For multi-output classification
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest classifier

#4) training of pipeline
from sklearn.model_selection import train_test_split

#5 test training model
from sklearn.metrics import classification_report
import numpy as np

#8 Improve model with Grid Search
from sklearn.model_selection import GridSearchCV        #for using GridSearchCV

#9 Export your model as a pickle file
import pickle     # for ML-model export s a pickle file


# In[4]:


# 1. load data from database.

# IMPORTANT: Notebook from 5.5 ETL has to be run bevore!
# load data from database

engine = create_engine('sqlite:///DisasterResponseProject.db')
df = pd.read_sql('SELECT * FROM DisasterResponses', engine)
X = df['message']
y = df.iloc[:,4:]


# In[5]:


df.head() #check df structure.


# In[6]:


# Check text in column 'message'
list(df['message'][:20])


# In[7]:


# 2. Write a tokenization function to process your text data


# Check text in column 'message'
list(df['message'][:20])

def tokenize(text):
    # Define a regex pattern to detect URLs
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace URLs with a placeholder
    text = re.sub(url_regex, "urlplaceholder", text)
    
    # Normalize and tokenize text
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english") and w.isalpha()]
    
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# In[8]:


# Check text tokens in column 'message'
for message in X[:20]:
    tokens = tokenize(message)
    print(tokens,'\n')
 


# In[9]:


# Replace all NaN values with 0

df.fillna(0, inplace=True)

# split the dataset

X = df.message
y = df.iloc[:,4:]
category_names = y.columns


# In[10]:


#3. Build a machine learning pipeline

machine_learning_pipeline = Pipeline([
    ('cvect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# In[11]:


# 4. Train pipeline

## train test split
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=55) # Split data into train and test sets
#machine_learning_pipeline.fit(X_train, y_train) # train test split

X_train, X_test, y_train, y_test = train_test_split(X, y)
machine_learning_pipeline.fit(X_train, y_train)

#y_pred = machine_learning_pipeline.predict(X_test)
#y_pred[55].shape


# In[12]:


# 5. Test your model

y_pred = machine_learning_pipeline.predict(X_test)

# Test shape of y_test and y_pred
print("Shape of y_test:", y_test.values.shape)
print("Shape of y_pred:", y_pred.shape)

# Test unique values in y_test and y_pred
print("Unique values in y_test:", np.unique(y_test))
print("Unique values in y_pred:", np.unique(y_pred))


# In[13]:


# Generate predictions y_pred and print out classification_report
y_pred = machine_learning_pipeline.predict(X_test)

# Iterate through each column and print the classification report
for i in range(y_test.shape[1]):  # Use the number of columns in y_test
    print(i,")","#########################", y_test.columns[i], "#########################")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    


# In[14]:


#Alternative:
#
## Assuming y_test is a DataFrame and y_pred is a NumPy array
#y_test_array = y_test.values  # Convert y_test to a NumPy array if it's a DataFrame
#
## Iterate through each class and print the classification report
#for i in range(y_test_array.shape[1]):  # Use the number of columns in y_test
#    #print("=======================", y_test.columns[i], "======================")
#    print(classification_report(y_test_array[:, i], y_pred[:, i], target_names=[y_test.columns[i]]))


# In[15]:


# 6. Improve your model with Grid Search to find better parameters

machine_learning_pipeline.get_params() # Displays the parameters of machine_learning_pipeline


# In[16]:


## specify parameters for grid search
#   
#parameters = {
#    'clf__n_estimators': [100, 200],
#    'clf__min_samples_split': [2, 3],
#}

# create grid search object
#cv = GridSearchCV(machine_learning_pipeline, param_grid=parameters)


# In[17]:


# Specify parameters for grid search
parameters = {
    'clf__estimator__n_estimators': [100],           # used for shorter run time  
#   'clf__estimator__n_estimators': [100, 200],      # better but runs longer     
    'clf__estimator__min_samples_split': [2, 3],
}

# Create grid search object
#cv = GridSearchCV(machine_learning_pipeline, param_grid=parameters)  #with warnings

cv = GridSearchCV(machine_learning_pipeline, param_grid=parameters, return_train_score=True, verbose=2)

cv.fit(X_train, y_train)


# In[18]:


cv.cv_results_


# In[19]:


# Identification of the best parameters from GritSearch analysis

print(cv.best_params_)


# In[23]:


#building new model

machine_learning_pipeline_optimized = cv.best_estimator_
print (cv.best_estimator_)


# In[24]:


# 7. test of the optimized machine learning model

# Generate predictions y_pred and print out classification_report
y_pred = machine_learning_pipeline_optimized.predict(X_test)

# Iterate through each column and print the classification report
for i in range(y_test.shape[1]):  # Use the number of columns in y_test
    print(i,")","#########################", y_test.columns[i], "#########################")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))


# In[28]:


# 9. Export your model as a pickle file
pickle.dump(machine_learning_pipeline_optimized, open('model.pkl', 'wb'))

