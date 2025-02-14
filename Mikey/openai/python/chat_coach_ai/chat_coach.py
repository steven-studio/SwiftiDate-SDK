import openai
import json
import numpy as np
from collections import deque

from embedding_utils import (
    get_embedding,
    cosine_similarity,
    red_flag_embeddings,
    nsfw_embeddings,
)
from prompt_generators import (
    generate_soi,
    generate_ioi_response,
    generate_iod_response,
)

# === 全域變數 ===
IS_GIRL_INITIATED = False  # 表示女生是否有「主動開話題」的狀態
conversation_history = deque(maxlen=10)  # 記錄最近 10 次對話分類結果
no_reply_count = 0  # 記錄女生不回應的次數

# === 37 套聊天案例所組成的 Prompt ===
case_prompt = (
    "你是一位戀愛高手，專門指導男生聊天。"
    "請嚴格按照以下 37 套聊天案例的風格來回應任何女生的對話。\n\n"
)

# === 判斷是否女生主動的範例函式 ===
def should_mark_girl_initiated(user_input: str) -> bool:
    """
    依據某些條件，判斷女生是否主動。
    以下示範：若對話歷史為空 (表示剛開始),
    或本專案規則(可自行擴充),
    就標記 True。
    """
    if len(conversation_history) == 0:
        return True
    # 其他邏輯亦可自行擴充
    return False

def is_exploded(response_text: str) -> str or None: # type: ignore
    """判斷男生是否過度迎合（舔狗行為）或聊色爆炸"""
    response_embedding = get_embedding(response_text)  # 轉換用戶輸入成向量
    
    for phrase, flag_embedding in red_flag_embeddings.items():
        if cosine_similarity(response_embedding, flag_embedding) > 0.85:
            return "舔狗爆炸！🔥 不要把睪丸放到女生手上，保持框架！"

    for phrase, flag_embedding in nsfw_embeddings.items():
        if cosine_similarity(response_embedding, flag_embedding) > 0.85:
            return "你爆了！🔥 不要急著聊色，這樣會掉價！"
        
    return None  # 沒有爆炸

def handle_no_reply() -> str:
    """處理女生不回應的情況"""
    global no_reply_count
    no_reply_count += 1

    if no_reply_count == 1:
        return "不理人？"
    elif no_reply_count >= 2:
        return "等幾天後再找她，不要急著催她。"
    return ""

def detect_no_self_worth(user_input: str) -> bool:
    """
    偵測女生是否「沒有配得感」的簡易示範，
    例如女生說「我好醜」「我不值得」等。
    """
    no_self_worth_phrases = ["我很醜", "我不值得", "配不上", "我什麼都不會"]
    return any(phrase in user_input for phrase in no_self_worth_phrases)

# === 以下四個變數/Embeddings，可在其他檔案 (e.g. constants.py 或 embedding_utils.py) 定義後 import ===
# 這裡假設你已經在別處定義：
#   IOI_embeddings, IOD_embeddings, meeting_IOD_embeddings, shit_test_embeddings

def classify_response(response_text: str) -> str:
    """
    判斷女生回應是:
    - SHIT_TEST
    - IOI (有興趣)
    - IOD (冷淡)
    - MEETING IOD (不想見面)
    - 中性
    """
    from embedding_utils import (
        IOI_embeddings, IOD_embeddings,
        meeting_IOD_embeddings, shit_test_embeddings,
    )
    response_embedding = get_embedding(response_text)
    
    max_ioi_sim = max(
        cosine_similarity(response_embedding, emb)
        for emb in IOI_embeddings.values()
    )
    max_iod_sim = max(
        cosine_similarity(response_embedding, emb)
        for emb in IOD_embeddings.values()
    )
    max_meeting_iod_sim = max(
        cosine_similarity(response_embedding, emb)
        for emb in meeting_IOD_embeddings.values()
    )
    max_shit_test_sim = max(
        cosine_similarity(response_embedding, emb)
        for emb in shit_test_embeddings.values()
    )
    
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

def chat_coach(user_input: str) -> str:
    """
    AI 依照 37 套聊天案例的風格回應。
    當女生顯示 IOI 時，由 AI 自行判斷「真IOI / 假IOI」，並決定回應「弱IOI / 強IOI」。
    除非偵測到女生沒有配得感，才給強IOI，否則預設給弱IOI。
    其餘 IOD / 中性 仍按原邏輯處理。
    """

    global IS_GIRL_INITIATED

    # 1. 偵測爆炸（舔狗 or 聊色）
    explosion = is_exploded(user_input)
    if explosion:
        return explosion
    
    # 2. 如果女生沒輸入任何東西(空字串)，處理不回應
    if user_input.strip() == "":
        return handle_no_reply()
    
    # 3. 判斷女生是否主動
    if should_mark_girl_initiated(user_input):
        IS_GIRL_INITIATED = True
    
    # 4. 分析回應類型
    cls_result = classify_response(user_input)
    conversation_history.append(cls_result)

    # 5. 若連續兩次 "MEETING_IOD"
    if conversation_history.count("MEETING_IOD") >= 2:
        return "你是不太想和我見面嗎？"

    # **檢查對話歷史來判斷整體趨勢**
    ioi_count = sum(1 for msg in conversation_history if msg == "IOI")
    iod_count = sum(1 for msg in conversation_history if msg == "IOD")
    
    # 6. 碰到空字串也再次檢查(以防邏輯)
    if user_input.strip() == "":
        return handle_no_reply()
    
    # 7. 隨機機率加入 SOI
    if np.random.rand() > 0.8:  # 偶爾加入 SOI
        return generate_soi()
    
    # 8. 根據分類結果決定回應
    no_self_worth = detect_no_self_worth(user_input)  # 你可以自行實作
    if cls_result == "IOI":
        # 這裡呼叫 OpenAI API，讓 AI 決定該回「強 IOI」或「弱 IOI」
        return generate_ioi_response(no_self_worth, user_input)
    elif cls_result == "IOD":
        # 讓 AI 動態生成「冷淡回應」
        return generate_iod_response(user_input)
    
    # 9. 如果都不符合 (含「中性」or 其他)
    #   就用最初的 system prompt 讓 GPT-4o 產生回覆
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": case_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    
    return response["choices"][0]["message"]["content"]