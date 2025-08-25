from pydantic import BaseModel
from fastapi import FastAPI
from deep import convert_styles
from de import search

async def correct(query:str ,style:str):
    # 프롬프트: 간단히 교정만
    system_prompt = "You are a teacher who corrects English grammar."
    sim = search(query,k=1)[0]
    sim_sent = sim.get(style, sim["text"])
    prompt = f"""
Please edit this sentence to fit your style without further explanation. also check the similarity  
sentence : \"{query}\"
style :\"{style}\"
similarity :\"{sim_sent}\"
"""
    resp = await convert_styles(system_prompt = system_prompt, user_prompt = prompt)
    print(resp)
    return resp
