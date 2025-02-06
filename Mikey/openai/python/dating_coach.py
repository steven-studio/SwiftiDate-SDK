import openai
import json
import numpy as np
from collections import deque

# 設定 API 金鑰
openai.api_key = "your_openai_api_key"

# **IOI（興趣指標）語句**
IOI_examples = [
    "你在幹嘛", "你週末有空嗎", "你平常喜歡做什麼", "你有沒有喜歡的餐廳", "你是不是很會聊天",
    "哈哈哈", "好好笑", "怎麼這麼可愛", "我們下次去哪", "我也想試試", "我可以嗎"
]

# **IOD（冷淡指標）語句**
IOD_examples = [
    "嗯", "好哦", "知道了", "不一定哦", "都可以",
    "你決定吧", "我現在不想聊", "沒什麼", "隨便", "..."
]

# 不想見面 IOD
meeting_iod_examples = [
    "再看看吧", 
    "我再想想"
]

shit_test_examples = [
    "你該不會只會聊天吧",
    "也就這樣啊",
    "你是不是很孤單",
    "你沒別的本事嗎",
    "我看你其實也還好嘛",
    "你是不是在炫耀啊",
    "也沒多厲害嘛",
    "就這樣？"
]

# **爆炸語錄（舔狗語錄）**
red_flags = [
    "我可以為你做任何事",
    "妳開心就好",
    "我沒關係",
    "我不值得",
    "對不起我錯了",
    "妳是我的全部",
    "求妳了",
    "妳說什麼我都聽",
    "只要妳開心，我什麼都願意",
    "拜託妳理我"
]

# **聊色行為（NSFW 語錄）**
nsfw_flags = [
    "你的身材好辣", "你穿什麼顏色的內衣", "晚上可以來我家嗎", "我們來點刺激的", "你睡覺會穿什麼",
    "要不要一起洗澡", "妳是不是很騷", "我們直接開房吧", "今晚要不要來點特別的"
]

# **獲取 Embeddings**
def get_embedding(text):
    """使用 OpenAI 的 text-embedding-ada-002 轉換文字為向量"""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"])

# **預先計算 IOI/IOD 向量**
IOI_embeddings = {phrase: get_embedding(phrase) for phrase in IOI_examples}
IOD_embeddings = {phrase: get_embedding(phrase) for phrase in IOD_examples}
meeting_IOD_embeddings = {phrase: get_embedding(phrase) for phrase in meeting_iod_examples}
red_flag_embeddings = {phrase: get_embedding(phrase) for phrase in red_flags}
nsfw_embeddings = {phrase: get_embedding(phrase) for phrase in nsfw_flags}
shit_test_embeddings = {phrase: get_embedding(phrase) for phrase in shit_test_examples}

# **記錄對話歷史**
conversation_history = deque(maxlen=10)  # 只記錄最近 10 次對話
no_reply_count = 0  # 記錄女生不回應的天數

def cosine_similarity(vec1, vec2):
    """計算餘弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 讀取 37 套聊天案例
with open("chat_cases.json", "r", encoding="utf-8") as file:
    chat_cases = json.load(file)

# 將 37 套案例轉成一個長 Prompt，讓 AI 記住並模仿
case_prompt = "你是一位戀愛高手，專門指導男生聊天。請嚴格按照以下 37 套聊天案例的風格來回應任何女生的對話。\n\n"
for i, case in enumerate(chat_cases, 1):
    case_prompt += f"案例 {i}:\n"
    for d in case["dialogue"]:
        case_prompt += f"{d['role']}: {d['content']}\n"
    case_prompt += "\n"

def classify_response(response_text):
    """判斷女生回應是 IOI（有興趣）還是 IOD（沒興趣）"""
    response_embedding = get_embedding(response_text)
    
    max_ioi_sim = max([cosine_similarity(response_embedding, emb) for emb in IOI_embeddings.values()])
    max_iod_sim = max([cosine_similarity(response_embedding, emb) for emb in IOD_embeddings.values()])
    max_meeting_iod_sim = max([cosine_similarity(response_embedding, emb) for emb in meeting_IOD_embeddings.values()])
    max_shit_test_sim = max([cosine_similarity(response_embedding, emb) for emb in shit_test_embeddings.values()])
    
    # 先判斷 Shit Test
    if max_shit_test_sim > 0.85:
        return "SHIT_TEST"

    # 依序判斷 IOI / IOD / MEETING_IOD / 中性
    if max_ioi_sim > 0.85:
        return "IOI" # （女生對你有興趣）
    elif max_iod_sim > 0.85:
        return "IOD" # （女生對你沒興趣）
    elif max_meeting_iod_sim > 0.85:
        return "MEETING IOD" # (女生對你沒興趣)
    else:
        return "中性" # （無法確定）

def analyze_conversation_trend():
    """分析長期 IOI/IOD 趨勢"""
    ioi_count = sum(1 for msg in conversation_history if "IOI" in msg)
    iod_count = sum(1 for msg in conversation_history if "IOD" in msg)

    if iod_count >= 2:
        return "測試：女生連續兩次冷淡回應，請直接詢問她的真實想法" 
    elif ioi_count > iod_count:
        return "趨勢：女生對你有興趣，繼續推進！"
    elif iod_count > ioi_count:
        return "趨勢：女生對你冷淡，可能要調整策略！"
    else:
        return "趨勢：女生反應一般，保持框架，觀察變化。"
    
# **測試對話**
test_responses = [
    "你週末有空嗎",  # IOI
    "我再看看吧",  # IOD
    "哈哈哈",  # IOI
    "隨便",  # IOD
    "嗯"  # IOD
]

for response in test_responses:
    result = classify_response(response)
    conversation_history.append(result)
    print(f"女生：{response}")
    print(f"分析結果：{result}")
    
# **查看長期趨勢**
print(analyze_conversation_trend())

def is_exploded(response_text):
    """判斷男生是否過度迎合（舔狗行為）"""
    response_embedding = get_embedding(response_text)  # 轉換用戶輸入成向量
    
    for phrase, flag_embedding in red_flag_embeddings.items():
        if cosine_similarity(response_embedding, flag_embedding) > 0.85:
            return "舔狗爆炸！🔥 不要把睪丸放到女生手上，保持框架！"

    for phrase, flag_embedding in nsfw_embeddings.items():
        if cosine_similarity(response_embedding, flag_embedding) > 0.85:
            return "你爆了！🔥 不要急著聊色，這樣會掉價！"
        
    return None  # 沒有爆炸

def handle_no_reply():
    """處理女生不回應的情況"""
    global no_reply_count
    no_reply_count += 1

    if no_reply_count == 1:
        return "不理人？"
    elif no_reply_count >= 2:
        return "等幾天後再找她，不要急著催她。"
    return ""

def generate_soi():
    """產生 SOI（表明意圖）"""
    value_statements = [
        "剛開完會，",
        "今天健身完超累，",
        "我剛試了一家新餐廳，"
    ]
    return np.random.choice(value_statements) + " 晚點一起吃個飯？"

# === 以下為假示範函式，實際上你有自己的實作 ===

def detect_no_self_worth(user_input):
    """
    偵測女生是否「沒有配得感」的簡易示範，
    例如女生說「我好醜」「我不值得」等。
    """
    no_self_worth_phrases = ["我很醜", "我不值得", "配不上", "我什麼都不會"]
    return any(phrase in user_input for phrase in no_self_worth_phrases)

def generate_ioi_response(no_self_worth, user_input):
    """
    呼叫 OpenAI ChatCompletion，讓 AI 根據 no_self_worth 決定要給 '強 IOI' 還是 '弱 IOI'
    """

    system_message = """
你是一位戀愛教練，請產生一段回應給女生。 
- 如果 no_self_worth = True，表示女生配得感很低，需要你給 "強 IOI" 來帶領她，並適度安撫或鼓勵她。
- 如果 no_self_worth = False，表示正常狀態，你只需要給 '弱 IOI'，不要表現得比對方更熱情。
- 回應風格參考你的 37 套聊天案例，語氣自然，不要過度浮誇，也不要舔狗。
"""

    user_prompt = f"""
女生的訊息: {user_input}
no_self_worth = {no_self_worth}

請用 1~2 句話，給出對應的 IOI 回應：
- 強 IOI：展現更高的熱情和帶領感
- 弱 IOI：保持輕鬆有趣，但不會過度熱情
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",     # 你實際使用的模型
        messages=messages,
        temperature=0.7,    # 可自行調整
    )

    ai_response = response["choices"][0]["message"]["content"]
    return ai_response

def generate_iod_response(user_input):
    """
    讓 AI 產生一個「低投資、保持框架」的回應。
    使用 ChatCompletion、搭配 37 套聊天案例風格。
    """

    system_message = """
你是一位戀愛教練，當女生回應冷淡 (IOD) 時，
你要維持框架、不要過度投資，也不要顯得討好或生氣。
請參考 37 套聊天案例的風格，語氣自然，簡短有禮。
"""

    user_prompt = f"""
女生的訊息: {user_input}
她的回應是 IOD（冷淡指標），請用 1~2 句話回應她，
保持禮貌又不失框架，語氣不要過度熱情。
"""

    messages = [
        {"role": "system", "content": case_prompt + "\n\n" + system_message},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",     # 你的Chat模型
        messages=messages,
        temperature=0.5     # 可自行調整：溫度低→更嚴謹穩定
    )
    
    ai_response = response["choices"][0]["message"]["content"]
    return ai_response

def generate_shit_test_response(user_input):
    """
    讓 AI 產生針對『廢物測試』(Shit Test) 的幽默回應。
    不要掉價，不要被激怒，能適度展現自信。
    """

    system_message = """
你是一位戀愛教練，當女生對你進行 '廢物測試(Shit Test)'，你要用幽默、自信的方式應對。
- 不要過度解釋或道歉
- 不要卑微，維持框架
- 語氣可輕鬆反問或帶點調侃，讓女生感到有趣
- 請參考 37 套聊天案例風格，不要過度攻擊或情緒化
"""

    user_prompt = f"""
女生的訊息: {user_input}
她對你有些挑釁或嘲諷，你要如何回應才不顯得掉價？
請用 1~2 句話展現幽默與自信。
"""

    messages = [
        {"role": "system", "content": case_prompt + "\n\n" + system_message},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7
    )
    
    ai_response = response["choices"][0]["message"]["content"]
    return ai_response

def chat_coach(user_input):
    """
    AI 依照 37 套聊天案例的風格回應。
    當女生顯示 IOI 時，由 AI 自行判斷「真IOI / 假IOI」，並決定回應「弱IOI / 強IOI」。
    除非偵測到女生沒有配得感，才給強IOI，否則預設給弱IOI。
    其餘 IOD / 中性 仍按原邏輯處理。
    """
    
    # 1. 偵測爆炸（舔狗 or 聊色）
    explosion = is_exploded(user_input)
    if explosion:
        return explosion
    
    # 2. 如果女生沒輸入任何東西(空字串)，處理不回應
    if user_input.strip() == "":
        return handle_no_reply()
    
    # 3. 分析這次女生訊息是 IOI / IOD / 中性
    cls_result = classify_response(user_input)
    conversation_history.append(cls_result)

    # 示範：偽代碼 - 假設我們有以下資訊
    #   - no_self_worth: 是否偵測到女生「沒有配得感」 (True / False)
    no_self_worth = detect_no_self_worth(user_input)  # 你可以自行實作
    
    #   - case_prompt: 你原本的 37 套聊天案例 Prompt
    #   - 其餘可依需求添加
    
    # **檢查對話歷史來判斷整體趨勢**
    ioi_count = sum(1 for msg in conversation_history if msg == "IOI")
    iod_count = sum(1 for msg in conversation_history if msg == "IOD")
    
    # **如果女生連續兩次說「再看看吧」，直接測試她**
    if conversation_history.count("MEETING_IOD") >= 2:
        return "你是不太想和我見面嗎？"
    
    # **處理女生不回應**
    if user_input.strip() == "":
        return handle_no_reply()
    
    # **產生 SOI**
    if np.random.rand() > 0.8:  # 偶爾加入 SOI
        return generate_soi()
    
    # **處理 IOI（興趣指標）**
    if cls_result == "IOI":
        # 這裡呼叫 OpenAI API，讓 AI 決定該回「強 IOI」或「弱 IOI」
        return generate_ioi_response(no_self_worth, user_input)
    
    # **處理 IOD（冷淡指標）**
    elif cls_result == "IOD":
        # 讓 AI 動態生成「冷淡回應」
        return generate_iod_response(user_input)
    
    """讓 AI 依照 37 套聊天案例的風格回應，並判斷是否爆了"""
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": case_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    ai_response = response["choices"][0]["message"]["content"]

    return ai_response

# 測試女生的回應
examples = [
    "今天有點累耶😩",  # AI 必須用 37 套案例的方式回應
    "最近好無聊喔😏",
    "好像很久沒出去玩了🤔",
    "妳開心就好，我沒關係🥺",  # 測試爆炸行為
    "妳說什麼我都聽，妳是我的全部🥺"
]

for example in examples:
    print(f"女生：{example}")
    print(f"戀愛教練 AI：{chat_coach(example)}\n")

