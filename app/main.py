from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.chat import generate_response
from pydantic import BaseModel

class QueryModel(BaseModel):
    query: str

app = FastAPI()

app.mount('/static', StaticFiles(directory='app/static'), name='static')

templates = Jinja2Templates(directory='app/templates')

@app.get('/', response_class=HTMLResponse)
async def get_index(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})

@app.post('/chat')
async def chat(query: QueryModel):
    response = generate_response(query.query)

    if isinstance(response, str):
        return JSONResponse(content={'response': response})
    else:
        return JSONResponse(content={'response': str(response)})