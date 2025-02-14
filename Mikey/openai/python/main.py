import openai
import json
import numpy as np
from collections import deque
from embedding_utils import get_embedding, cosine_similarity
from constants import IOI_EXAMPLES, IOD_EXAMPLES, MEETING_IOD_EXAMPLES, SHIT_TEST_EXAMPLES, RED_FLAGS, NSFW_FLAGS
from chat_coach import chat_coach

# 設定 API 金鑰
openai.api_key = "your_openai_api_key"

# **記錄對話歷史**
no_reply_count = 0  # 記錄女生不回應的天數

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

conversation_history = deque(maxlen=10)  # 只記錄最近 10 次對話

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

# for response in test_responses:
#     result = classify_response(response)
#     conversation_history.append(result)
#     print(f"女生：{response}")
#     print(f"分析結果：{result}")
    
# **查看長期趨勢**
# print(analyze_conversation_trend())

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
