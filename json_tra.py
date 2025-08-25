# scripts/build_styled_json.py
import re, json, time, pathlib, pandas as pd, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.document_loaders import CSVLoader, PyPDFLoader
# 1. 엑셀 로드 (H열 = index 7)

pdf_loader = PyPDFLoader(file_path='EventGameDocument.pdf')

# 문서 로드
pdf_documents = pdf_loader.load()
cleaned_pdf_content = pdf_documents[0].page_content.replace('\n', ' ')

# 로드된 문서 확인

# 2. 모델 로드 (Chat 버전 필수)
model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)

# 3. Qwen 템플릿 함수
def qwen_chat(system, user):
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
        {"role": "assistant", "content": ""}
    ]
    tpl = tok.apply_chat_template(messages, tokenize=False)
    return tok(tpl, return_tensors="pt").to(model.device)

SYS_PROMPT = (
   """ 
You are a document parser. 
Given raw text tagged by page boundaries, split it into sentences and output a JSON array.
Each item must have:
- title: text title
- page: integer page number
- text: the sentence text

Only output valid JSON; do not include any additional explanation.
Example (do NOT wrap in ```json):
{"THỂ LỆ CHUNG":{"title": "Điều kiện tham gia" ,"text":"Tất cả người chơi cần đăng ký tài khoản bằng email hoặc số điện thoại"...}....}
"""
)

json_re = re.compile(r"\{[\s\S]*?\}", re.S)

def convert_styles(sentence: str):
    usr = f'Input: "{cleaned_pdf_content}"\nReturn the JSON.'
    inp = qwen_chat(SYS_PROMPT, usr)
    prompt_len = inp["input_ids"].shape[-1]

    out_ids = model.generate(
        input_ids      = inp["input_ids"],
        attention_mask = inp["attention_mask"],
        max_new_tokens = 128,
        temperature    = 0.7
    )

    text = tok.decode(out_ids[0][prompt_len:], skip_special_tokens=True)
    match = json_re.search(text)
    print(text)
    print(match)
    if not match:
        print("⚠️  JSON not found:", sentence[:40])
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        print("⚠️  Parse error:", sentence[:40])
        return None

# # 4. 메인 루프
# results = []
# for i, sent in enumerate(cleaned_pdf_content, 1):
#     print(f"🔄 ({i}/{len(cleaned_pdf_content)}) {sent[:40]}...")
#     var = convert_styles(sent)
#     if var:
#         results.append({
#             "{sent}":{
#               {
#                 "title": {"q": var.get("Black English", "")},
#                 "White English": {"q": var.get("White English", "")},
#                 "British English": {"q": var.get("British English", "")}
#             }}
#         })
#     time.sleep(0.4)
# out_path = pathlib.Path("styled_questions.json")

# # 5. 저장
# if out_path.exists():
#     try:
#         existing = json.loads(out_path.read_text(encoding="utf-8"))
#         if isinstance(existing, list):
#             combined = existing + results
#         else:
#             # 기존 파일이 리스트가 아니면 백업하고 새로 시작
#             backup = out_path.with_suffix(".backup.json")
#             out_path.rename(backup)
#             print(f"⚠️ 기존 JSON이 리스트가 아닙니다. 백업: {backup.name}")
#             combined = results
#     except json.JSONDecodeError:
#         print("⚠️ 기존 JSON 파싱 실패. 덮어쓰기하지 않고 새로 생성합니다.")
#         combined = results
# else:
#     combined = results

# # 덮어쓰기 저장
# out_path.write_text(
#     json.dumps(combined, ensure_ascii=False, indent=2),
#     encoding="utf-8"
# )
# print(f"✅ styled_questions.json 업데이트 완료! 총 항목: {len(combined)} 문장")
