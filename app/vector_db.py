import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = 'data/cocktail_index.pkl'
EMBEDDINGS_PATH = 'data/cocktail_embeddings.pkl'
USER_MEMORY_PATH = 'data/user_memory_embeddings.pkl'

model = SentenceTransformer('all-MiniLM-L6-v2')

def load_embeddings(file_path):
    try:
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        return embeddings
    except FileNotFoundError:
        return {}
    except EOFError:
        return {}

def save_embeddings(file_path, embeddings):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)

def get_user_preferences(user_memory_embeddings):
    if not user_memory_embeddings:
        return np.zeros_like(next(iter(user_memory_embeddings.values()), np.zeros(512)))
    return list(user_memory_embeddings.values())[0]

def update_user_memory(new_memory_vector, user_memory_embeddings):
    user_memory_embeddings.clear()
    user_memory_embeddings['shared_memory'] = new_memory_vector 
    save_embeddings(USER_MEMORY_PATH, user_memory_embeddings)


def load_vector_store():
    with open(INDEX_PATH, 'rb') as f:
        index = pickle.load(f)
    with open(EMBEDDINGS_PATH, 'rb') as f:
        df = pickle.load(f)
    return index, df

index, df = load_vector_store()

def search_cocktails(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    results = df.iloc[indices[0]].to_dict(orient='records')
    return results