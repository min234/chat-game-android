# chat_tutor.py
import json, faiss, numpy as np, asyncio
from pathlib import Path
from sentence_transformers import SentenceTransformer
 
from dotenv import load_dotenv
from deep import convert_styles
from de import search

load_dotenv(dotenv_path=".env", override=True)

 
async def deep(
    user_query: str,
    history: list[dict],
    style: str = "Black"
) -> list[dict]:
    user_prompt = f"""
    When writing an answer, please only display AI answers.
User: "{user_query}"
ai reply:
"""
    if not any(m["role"] == "system" for m in history):
        top = search(user_query, k=1)[0]
        sim_sent = top.get(style, top["text"])
        
        # ① f-string으로 변수 삽입
        system_prompt = f"""
   You are a friendly chat companion who speaks in {style} English.
   Use the example below only to capture the style—do NOT summarize or explain it.

    Look at {sim_sent} and think of the most similar answer.
ABSOLUTE RULES:
1. Reply in a natural, conversational style—do NOT describe or summarize the user’s feelings.
2. Speak directly to the user, using “you” and “I” as if you’re chatting face-to-face.
Now continue the conversation with the user’s last message.
    """
        # ② 실제 변환 지시 (user)
    
            # ③ convert_styles에 넘기기
        history.insert(0, {"role": "system", "content": system_prompt})

    # ——— 유저 메시지 추가 ———
    history.append({"role": "user", "content": user_query})

    # ——— AI 호출 ———
    response = convert_styles(
        history[0]["content"],  # system prompt
        user_prompt              # latest user message
    )
    clean = response.replace("\n", " ")

    # 2) 양쪽 공백(스페이스·탭 등) 전부 제거
    clean = clean.strip()  # → "Hi, thank y'all for likin' me."

    # 필요하다면 접두어가 있을 때만 잘라내기
    prefix = "AI reply:"
    if clean.startswith(prefix):
        clean = clean[len(prefix):].strip()
    # ——— 어시스턴트 응답 추가 & 리턴 ———
    history.append({"role": "assistant", "content": clean})
    return clean

