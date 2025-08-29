# app.py  — Kutubbia Chatbot

import os
import gradio as gr
from huggingface_hub import InferenceClient

# الموديل الافتراضي
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

# التوكن من HuggingFace
HF_TOKEN = os.environ.get("HF_TOKEN")

SYSTEM_PROMPT = (
    "أجب بالعربية الفصحى باختصار ووضوح. "
    "عند الشك أو غياب المعلومة قل: لست متأكدًا. "
    "تجنّب اختلاق المعلومات."
)

# تهيئة العميل
def get_client():
    if not HF_TOKEN:
        raise RuntimeError("لم يتم العثور على HF_TOKEN. أضفه في Settings → Secrets → New secret → HF_TOKEN")
    return InferenceClient(model=MODEL_ID, token=HF_TOKEN)

client = None

def stream_chat(message, history):
    global client
    if client is None:
        client = get_client()

    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        last_user, last_bot = history[-1]
        if last_user:
            msgs.append({"role": "user", "content": last_user})
        if last_bot:
            msgs.append({"role": "assistant", "content": last_bot})

    msgs.append({"role": "user", "content": message})

    partial = ""
    try:
        for chunk in client.chat_completion(
            messages=msgs,
            max_tokens=180,
            temperature=0.2,
            top_p=0.9,
            stream=True
        ):
            delta = getattr(chunk.choices[0], "delta", None)
            token = ""
            if delta and hasattr(delta, "get"):
                token = delta.get("content", "")
            elif hasattr(chunk.choices[0], "message"):
                token = chunk.choices[0].message.get("content", "")
            if token:
                partial += token
                yield partial

    except Exception as e:
        yield f"حدث خطأ: {e}"

# واجهة Gradio
demo = gr.ChatInterface(
    fn=stream_chat,
    title="🤖 Kutubbia Chatbot",
    description="بوت كوتوبيا يدعم العربية ويجيب بالفصحى بسرعة وبأقل هلوسة",
    examples=["السلام عليكم", "من هو مؤسس فيسبوك", "اكتب لي حكمة عربية قصيرة"]
)

if __name__ == "__main__":
    demo.launch()
