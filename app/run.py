import json
import plotly
import pandas as pd
import re
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords 

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponseProject.db')
df = pd.read_sql_table('DisasterResponses', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

    # Figure 1 - Distribution of Message Genres 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #Figure 2 - Distribution of Top Words in Disaster Messages
    message_text = ""
    for i,j in df.iterrows():
        message_text+=j["message"]
   
    # Normalize text by changing to lower case and removing punctuation 
    text = re.sub(r"[^a-zA-Z0-9]",  " ",message_text)
    text = text.lower().strip()
    # split() returns list of words in the text 
    word_list = text.split()

    # Python Counter as a container that contains the count of each of word in word_list 
    word_container = Counter(word_list) 
  
    # Return a list of the 50 top words used in Disater Messages and their counts from the most common to the least. 
    top_words  = word_container.most_common(50) 
    top_words_clean = [w for w in top_words if w[0] not in stopwords.words("english")]
    
    x_val = [x[0] for x in top_words_clean[:10]]
    y_val = [x[1] for x in top_words_clean[:10]]

    #Figure 3 - Distribution of Message Categories
    category_count = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
    category_names = list(category_count.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts,
                    marker=dict(color='#476D2D')
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Genres',
                    'font': {
                        'size':24,
                        'weight': 'bold'
                    }
                },
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Message Genre",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_val,
                    y=y_val,
                    marker=dict(color='#476D2D')
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Top Words in Disaster Messages',
                    'font': {
                        'size': 24,
                        'weight': 'bold'
                    }
                },
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Words in Messages",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_count,
                    marker=dict(color='#476D2D')
                )
            ],
            'layout': {
                'title': {
                    'text': 'Distribution of Message Categories',
                    'font': {
                        'size': 24,
                        'weight': 'bold'
                    }
                },
                'yaxis': {
                    'title': {
                        'text': "Count",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                },
                'xaxis': {
                    'title': {
                        'text': "Message Category",
                        'font': {
                            'size': 18,
                            'weight': 'bold'
                        }
                    }
                }
            }
        } 
    ]
        
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
    