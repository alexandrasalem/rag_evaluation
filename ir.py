from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import os
from joblib import Memory

memory = Memory(location='cache_dir', verbose=0)


@memory.cache
def load_tfidf(list_of_docs):
    """
    This function creates a tfidf vectorizer and document-term matrix.

    Takes as input:
    :param list_of_docs: corpus data, loaded as a dataframe, with 'abstract' column.
    And then outputs:
    :return: the vectorizer for the tfidf (vectorizer), and the document-term matrix (X)
    """

    #corpus = list(data['abstract']) #[item['abstract'].lower() for item in data]
    corpus = [item.lower() for item in list_of_docs]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    vectorizer.get_feature_names_out()
    X = normalize(X, axis=1, norm="l1")
    return vectorizer, X

def ir_single_query_top_docs_bio(question,  corpus_csv='pubmed_abstracts.csv', k = 5):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question: the string of the current question
    :param corpus_csv: location of csv file with docs
    :return: a string with the text of the doc with the highest cosine similarity to the query.
    """
    question = question.lower()
    docs = pd.read_csv(corpus_csv)

    vectorizer, X = load_tfidf(docs['abstract'])
    query = [question]
    query_vector = vectorizer.transform(query)
    res = cosine_similarity(X, query_vector).squeeze()

    top_idx = np.argpartition(res, -k)[-k:]
    res_data = docs.loc[top_idx]
    return res_data

def ir_single_query_top_docs_trivia(question,  corpus_location = 'triviaqa-rc/evidence/wikipedia', k = 1):
    """
    This function pulls the doc with the highest cosine similarity to the query and returns it.

    Takes as input:
    :param question: the string of the current question
    :param corpus_csv: location of csv file with docs
    :return: a string with the text of the doc with the highest cosine similarity to the query.
    """
    question = question.lower()
    doc_texts = []
    doc_filenames = []
    filenames = os.listdir(corpus_location)
    for filename in filenames:
        if filename.endswith('.txt'):
            with open(corpus_location + '/' + filename, 'r') as f:
                doc = f.read()
                doc_texts.append(doc)
                doc_filenames.append(filename)
    docs = pd.DataFrame({'id': doc_filenames, 'text': doc_texts})

    vectorizer, X = load_tfidf(docs['text'])
    query = [question]
    query_vector = vectorizer.transform(query)
    res = cosine_similarity(X, query_vector).squeeze()

    top_idx = np.argpartition(res, -k)[-k:]
    res_data = docs.loc[top_idx]
    return res_data
