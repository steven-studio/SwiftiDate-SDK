import openai
import json
import numpy as np
from collections import deque

case_prompt = "你是一位戀愛高手，專門指導男生聊天。請嚴格按照以下 37 套聊天案例的風格來回應任何女生的對話。\n\n"

def generate_soi():
    """產生 SOI（表明意圖）"""
    value_statements = [
        "剛開完會，",
        "今天健身完超累，",
        "我剛試了一家新餐廳，"
    ]
    return np.random.choice(value_statements) + " 晚點一起吃個飯？"

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