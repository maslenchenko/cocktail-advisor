# Cocktail Advisor Chat
Project structure:
```
cocktail_advisor/
│── data/
│   ├── final_cocktails.csv  
│   ├── cocktail_embeddings.pkl
│   ├── cocktail_index.pkl  
│   ├── user_memory_embeddings.pkl  
│── app/
│   ├── main.py       
│   ├── chat.py     
│   ├── vector_db.py  
│   ├── create_vector_db.py  
│── templates/
│   ├── index.html 
│── static/
│   ├── style.css
│   ├── script.js
│── requirements.txt
│── README.md
```

## To start and use the application:
1. Clone the repo
2. Install `requirements.txt`
```
pip install -r requirements.txt
```
4. Enter your OpenAI API Key in `app/chat/py`
5. Run
 ```
 uvicorn app.main:app --reload
 ```
 from the `app` directory

## Preview:
![image_2025-01-31_19-04-52](https://github.com/user-attachments/assets/e3813f37-ed89-46a9-8937-eae75a4a01b2)

## What need to be improved
Due to the lack of time, I wasn't able to fully implement a robust user memory system. Currently, the system saves both the user's queries and the ingredients of the suggested cocktails.
