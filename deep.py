from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="cuda", torch_dtype=torch.bfloat16
)

def qwen_chat(system_prompt, user_prompt):
    msgs = [
        {"role": "system",    "content": system_prompt},
        {"role": "user",      "content": user_prompt},
      
    ]
    tpl = tok.apply_chat_template(msgs, tokenize=False, add_special_tokens=False)
    return tok(tpl, return_tensors="pt").to(model.device)

json_re = re.compile(r"\{[\s\S]*?\}")

def convert_styles(system_prompt, user_prompt):
    inp = qwen_chat(system_prompt, user_prompt)
    prompt_len = inp["input_ids"].shape[-1]
    
    out_ids = model.generate(
        **inp,
        max_new_tokens = 128,
        temperature    = 0.3,
        eos_token_id   = tok.eos_token_id,
    
    )

    text = tok.decode(out_ids[0][prompt_len:], skip_special_tokens=True)

    # 백틱 제거
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text).rstrip("`").strip()

    # ─── 여기에 think 태그도 제거 ───
    text = re.sub(r"</?think>", "", text)

    sent_match = re.match(r'^(.*?[\.!?])', text)
    if sent_match:
        return sent_match.group(1).strip()
    return text
