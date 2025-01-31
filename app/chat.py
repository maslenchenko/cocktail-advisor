import openai
import numpy as np
from app.vector_db import search_cocktails, get_user_preferences, update_user_memory
from app.vector_db import load_embeddings, save_embeddings
from sentence_transformers import SentenceTransformer

openai.api_key = 'API-KEY'

cocktail_embeddings = load_embeddings('data/cocktail_embeddings.pkl')
user_memory_embeddings = load_embeddings('data/user_memory_embeddings.pkl')

def get_user_memory_vector():
    return get_user_preferences(user_memory_embeddings)

def update_user_memory_vector(new_memory_vector: np.ndarray):
    update_user_memory(new_memory_vector, user_memory_embeddings)

def generate_response(query: str, top_k=5):
    user_memory_vector = get_user_memory_vector()

    cocktail_results = search_cocktails(query, top_k)

    cocktail_info = '\n'.join([ 
        f"Name: {cocktail['name']}\nIngredients: {', '.join(cocktail['ingredients'])}\nInstructions: {cocktail['instructions']}\n"
        for cocktail in cocktail_results
    ])

    prompt = (
        f'You are a cocktail expert. The user has preferences stored in their memory:\n\n'
        f'User Memory: {user_memory_vector}\n\n'
        f'The user asked the following question:\n\n'
        f'User Query: {query}\n\n'
        f"Here are some cocktail recommendations based on the user's query and memory:\n{cocktail_info}\n\n"
        f"Please explain how these cocktails relate to the user's query and provide a detailed and helpful response."
        f'Talk to the user.'
    )

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=[
            {'role': 'system', 'content': 'You are a helpful cocktail expert.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=700, 
        temperature=0.7,
        top_p=1,      
        n=1,             
        stop=None       
    )

    response_text = response['choices'][0]['message']['content'].strip()

    selected_cocktail = cocktail_results[0]
    new_memory_vector = generate_memory_vector(selected_cocktail, query)

    update_user_memory_vector(new_memory_vector)

    return response_text

def generate_memory_vector(selected_cocktail, user_query):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cocktail_ingredients = ', '.join(selected_cocktail['ingredients'])
    combined_text = f"User Query: {user_query} Ingredients: {cocktail_ingredients}"
    memory_vector = model.encode([combined_text], convert_to_numpy=True)
    return memory_vector
