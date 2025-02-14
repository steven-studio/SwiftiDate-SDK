import openai
import json
import numpy as np
from collections import deque
from constants import IOI_EXAMPLES, IOD_EXAMPLES, MEETING_IOD_EXAMPLES, shit_test_examples, red_flags, nsfw_flags

# **獲取 Embeddings**
def get_embedding(text):
    """使用 OpenAI 的 text-embedding-ada-002 轉換文字為向量"""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response["data"][0]["embedding"])

def cosine_similarity(vec1, vec2):
    """計算餘弦相似度"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# **預先計算 IOI/IOD 向量**
IOI_embeddings = {phrase: get_embedding(phrase) for phrase in IOI_examples}
IOD_embeddings = {phrase: get_embedding(phrase) for phrase in IOD_examples}
meeting_IOD_embeddings = {phrase: get_embedding(phrase) for phrase in meeting_iod_examples}
red_flag_embeddings = {phrase: get_embedding(phrase) for phrase in red_flags}
nsfw_embeddings = {phrase: get_embedding(phrase) for phrase in nsfw_flags}
shit_test_embeddings = {phrase: get_embedding(phrase) for phrase in shit_test_examples}
