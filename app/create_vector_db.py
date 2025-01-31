import pandas as pd
import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

DATA_PATH = 'data/final_cocktails.csv'
INDEX_PATH = 'data/cocktail_index.pkl'
EMBEDDINGS_PATH = 'data/cocktail_embeddings.pkl'

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data():
    df = pd.read_csv(DATA_PATH)
    
    df['description'] = (
        'Name: ' + df['name'] + 
        '. Ingredients: ' + df['ingredients'] + ' (' + df['ingredientMeasures'] + ') ' +
        '. Instructions: ' + df['instructions']
    )
    
    return df

def build_vector_store():
    df = load_data()

    embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(index, f)
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(df, f)

    print('vector database built successfully!')

if __name__ == '__main__':
    build_vector_store()