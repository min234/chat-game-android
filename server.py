# ai_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from ve_db import deep
from correct import correct


class ChatRequest(BaseModel):
    text: str
    style: str

class ChatResponse(BaseModel):
    reply: str

app = FastAPI()

class CorrectionRequest(BaseModel):
    text: str

class CorrectionResponse(BaseModel):
    correctedText: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    print("POST:: /chat")
    print("TEXT: " + req.text)
    print("STYLE: " + req.style)
    global history
    try:
        history  # 이미 정의된 경우
    except NameError:
        history = []

    # deep() 는 이제 문자열을 반환합니다.
    reply_text = await deep(user_query=req.text, history=history, style=req.style)
    print("AI reply:", reply_text)
    a = await correct(query=req.text,style=req.style)
    print(a)
 
    return ChatResponse(reply=reply_text)
# @app.post("/chat", response_model=CorrectionResponse)
# async def correct(req:str,style):
   
#     cor = await correct(query,style)
#     print("고치는 문장:",cor)
#     print("style:",style)
#     return cor

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
