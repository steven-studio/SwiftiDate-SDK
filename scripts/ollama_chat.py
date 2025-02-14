import requests

OLLAMA_URL = "http://127.0.0.1:11411"
MODEL_NAME = "llama3:8b"

# 在這裡定義我們的系統提示，描述兩個角色
system_prompt = """你是一個負責模擬對話的模型，對話中有兩個角色：

角色1（男生）：性格自大、愛炫耀，嘗試「把妹」。
角色2（女生）：個性比較冷靜，對男生的炫耀抱持懷疑或輕蔑的態度。

請在接下來的對話中，依照這兩種角色的個性與語氣，生成中文對話。請不要跳出角色設定。"""

# 開始時，messages 裡放一條 "system" 訊息（OpenAI風格）
messages = [
    {"role": "system", "content": system_prompt}
]

def ollama_generate(messages, model=MODEL_NAME):
    """
    向 Ollama 發送 /api/generate，傳入 messages 進行推理，取得回應文字。
    messages 格式為類似 OpenAI ChatGPT 的 {role, content} 陣列。
    """
    # 組合成 chat 格式
    # Ollama 的 /api/generate 大多會接受 prompt: <str>
    # 我們需要手動把 system/user/assistant 的對話串接成一個大 prompt
    prompt_text = ""
    for msg in messages:
        if msg["role"] == "system":
            prompt_text += f"[SYSTEM]\n{msg['content']}\n"
        elif msg["role"] == "user":
            prompt_text += f"[USER]\n{msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt_text += f"[ASSISTANT]\n{msg['content']}\n"

    # 在 prompt 最後加入標記，提示我們要讓模型開始輸出回答
    prompt_text += "[ASSISTANT]\n"

    payload = {
        "model": model,
        "prompt": prompt_text,
        # 可以依照需要加一些推理參數，如 temperature, top_k, max_tokens 等
        "temperature": 0.7,
        "max_tokens": 200
    }

    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)

    # Ollama 默認會用 SSE (Server-Sent Event) 流回傳，需要逐行解析
    full_text = ""
    for chunk in response.iter_lines(decode_unicode=True):
        if chunk:
            # chunk是一個JSON字串，如 {"token":"你好"}
            # 解析出裡面的token
            try:
                data = json.loads(chunk)
                if "done" in data and data["done"]:
                    break
                if "token" in data:
                    full_text += data["token"]
            except:
                # 如果解析失敗就忽略
                pass

    return full_text.strip()

def simulate_dialog():
    """
    模擬對打：由男生先講一句，再由模型回應女生、然後再男生，如此輪流數輪。
    這裡只是範例，可自行擴充。
    """
    # 假設我們手動控制輪次
    total_rounds = 4

    # 男生、女生台詞可手動輸入(模擬)或隨機
    male_lines = [
        "嘿，美女，等很久了嗎？我剛剛開著跑車去吃米其林餐廳，晚了一點。",
        "你知道嗎，我投資了好多股票，每天都在賺錢耶！"
    ]
    female_lines = [
        "呵，看不出來你竟然會遲到？不過算了。",
        "投資那麼厲害，那你現在可以退休了嗎？"
    ]

    # 先由男生說話 (角色1)
    for i in range(total_rounds):
        # 男生先發話
        male_text = male_lines[i % len(male_lines)]
        messages.append({"role": "user", "content": f"【男生】：{male_text}"})

        # 取得模型回應(假設是女生視角回應，但由assistant產生)
        assistant_reply = ollama_generate(messages)
        # 把回應記錄下來
        messages.append({"role": "assistant", "content": assistant_reply})

        print(f"\n--- 第 {i+1} 輪 ---")
        print(f"(男生) {male_text}")
        print(f"(女生) {assistant_reply}")

        # 接著，再由女生（我們這裡也模擬手動输入）
        if i < total_rounds - 1:  # 避免最後一次又多一輪
            female_text = female_lines[i % len(female_lines)]
            messages.append({"role": "user", "content": f"【女生】：{female_text}"})
            
            # assistant 再回應(男生視角)
            assistant_reply_2 = ollama_generate(messages)
            messages.append({"role": "assistant", "content": assistant_reply_2})

            print(f"(女生) {female_text}")
            print(f"(男生) {assistant_reply_2}")

if __name__ == "__main__":
    import json
    simulate_dialog()
    print("\n對話結束。")
