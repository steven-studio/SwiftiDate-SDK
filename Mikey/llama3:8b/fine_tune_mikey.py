# fine_tune_mikey.py

import os
import json
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel

# 1. 指定基底模型與 LoRA 設定
BASE_MODEL_NAME = "decapoda-research/llama-7b-hf"  # 只作示例, 需有合法權限存取
MICKEY_DATA_PATH = os.path.join(os.path.dirname(__file__), "mikey_success.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mikey_lora_out")

lora_config = LoraConfig(
    r=8,                # rank
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 2. 準備資料集
def load_mikey_data(file_path):
    """
    讀取 jsonl 格式資料, 返回 list of dict.
    每行格式: {"instruction": "...", "input": "...", "output": "..."}
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            data_list.append(item)
    return data_list

# 簡易 Dataset wrapper
class MikeyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, tokenizer, max_length=512):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        # instruction + input + output 拼接成一個 prompt/label
        # 以下為示例, 您可自行設計 prompt 格式
        instruction = record.get("instruction", "")
        user_input = record.get("input", "")
        output = record.get("output", "")

        # Alpaca-style prompt:
        # <s>Instruction: {instruction}\nInput: {user_input}\nResponse: {output}</s>
        prompt_text = f"指令: {instruction}\n" \
                      f"輸入: {user_input}\n" \
                      f"回覆: "

        # 模型要產生 output, 所以將 output 設為label
        # 先 tokenize prompt + output
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length)
        output_ids = self.tokenizer(output, truncation=True, max_length=self.max_length)

        # 為了簡化, 這裡直接把 prompt 和 output 接起來
        input_ids = prompt_ids["input_ids"] + output_ids["input_ids"][1:]  # 移除第一個[CLS]/[BOS]
        # attention_mask 也同樣處理
        attention_mask = prompt_ids["attention_mask"] + output_ids["attention_mask"][1:]

        # 創建 labels, 其中 prompt 的部分可標記為 -100 以忽略梯度, 只訓練 output
        # 這裡是簡化示例, 若要更精細可看各種 Alpaca-LoRA 實作
        labels = [-100]*len(prompt_ids["input_ids"]) + output_ids["input_ids"][1:]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def data_collator(batch):
    # 動態填充
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def main():
    # 1. 載入原始模型 & Tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL_NAME)
    # 如果要 int8 加速, 需安裝 bitsandbytes 等
    base_model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )

    # 2. 准備LoRA
    model = prepare_model_for_int8_training(base_model)
    model = get_peft_model(model, lora_config)

    # 3. 載入資料 & 建立 Dataset
    data_list = load_mikey_data(MICKEY_DATA_PATH)
    dataset = MikeyDataset(data_list, tokenizer, max_length=512)

    # 這裡沒拆 train / val, 只是簡化演示
    train_dataset = dataset

    # 4. 設定 TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,  # 示範只跑1 epoch
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=1e-4,
        fp16=True,
        optim="adamw_torch"
    )

    # 5. 建立 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # 6. 開始訓練
    trainer.train()

    # 7. 存下 LoRA 權重 (只會存 LoRA 差分)
    model.save_pretrained(OUTPUT_DIR)
    print(f"LoRA fine-tune complete. Weights saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
