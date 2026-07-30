"""
Script fine-tune mô hình reranker (Cross-Encoder) sử dụng LoRA.
Hỗ trợ thiết bị CUDA và MPS (Apple Silicon).
"""

import os
os.environ["USE_TF"] = "0"        # Bỏ qua TensorFlow, chỉ dùng PyTorch
os.environ["USE_FLAX"] = "0"

import json
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from peft import LoraConfig, TaskType, get_peft_model
from datasets import Dataset


def load_data(data_path: str):
    """
    Tải dữ liệu huấn luyện từ file JSONL.
    
    Args:
        data_path: Đường dẫn tới file JSONL chứa dữ liệu huấn luyện.
        
    Returns:
        queries, passages, labels
    """
    queries = []
    passages = []
    labels = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(data['query'])
            passages.append(data['passage'])
            labels.append(float(data['label']))
            
    return queries, passages, labels


def compute_metrics(p: EvalPrediction):
    """
    Tính toán các metrics đánh giá cho mô hình.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Áp dụng sigmoid để chuyển đổi logits thành xác suất
    probs = 1.0 / (1.0 + np.exp(-preds.flatten()))
    
    # Tính ROC-AUC
    try:
        auc = roc_auc_score(p.label_ids, probs)
    except ValueError:
        auc = 0.0  # Xảy ra khi chỉ có 1 class trong batch
        
    # Dự đoán class (ngưỡng 0.5)
    pred_labels = (probs > 0.5).astype(int)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        p.label_ids, pred_labels, average='binary', zero_division=0
    )
    acc = accuracy_score(p.label_ids, pred_labels)
    
    return {
        'roc_auc': auc,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def find_all_linear_names(model):
    """
    Tìm tất cả các lớp tuyến tính (Linear layers) trong mô hình để áp dụng LoRA.
    """
    import torch.nn as nn
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            
    # Xóa các lớp output/classification ra khỏi mục tiêu LoRA
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    if 'score' in lora_module_names:
        lora_module_names.remove('score')
    if 'classifier' in lora_module_names:
        lora_module_names.remove('classifier')
        
    return list(lora_module_names)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Cross-Encoder Reranker với LoRA")
    parser.add_argument("--data_path", type=str, default="training_data/reranker_pairs.jsonl", 
                        help="Đường dẫn file dữ liệu huấn luyện")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="Tên mô hình gốc (base model)")
    parser.add_argument("--output_dir", type=str, default="models/reranker_lora",
                        help="Thư mục lưu mô hình sau khi fine-tune")
    parser.add_argument("--epochs", type=int, default=3, help="Số epoch huấn luyện")
    parser.add_argument("--batch_size", type=int, default=2, help="Kích thước batch (MPS: 2, CUDA: 8-16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=256, help="Độ dài chuỗi tối đa (256 tiết kiệm VRAM)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (8 tiết kiệm VRAM, 16 expressive hơn)")
    
    args = parser.parse_args()
    
    # Thiết lập seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Xác định device (CUDA, MPS, hoặc CPU)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Sử dụng thiết bị: {device}")
    
    print("Đang tải dữ liệu...")
    try:
        queries, passages, labels = load_data(args.data_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {args.data_path}")
        return
        
    print(f"Tổng số mẫu dữ liệu: {len(queries)}")
    
    # Chia dữ liệu train/val (80/20) stratified
    q_train, q_val, p_train, p_val, y_train, y_val = train_test_split(
        queries, passages, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    
    print("Đang tải tokenizer và xử lý dữ liệu...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_func(queries_list, passages_list):
        return tokenizer(
            queries_list,
            passages_list,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors=None
        )
        
    train_encodings = tokenize_func(q_train, p_train)
    val_encodings = tokenize_func(q_val, p_val)
    
    # Tạo HuggingFace Datasets
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': y_train
    })
    
    val_dataset = Dataset.from_dict({
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': y_val
    })
    
    print("Đang khởi tạo mô hình...")
    # num_labels=1 cho bài toán phân loại nhị phân (Binary Classification)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,  # MPS không hỗ trợ float16
    )
    # Bật gradient checkpointing — giảm VRAM ~40% (trade-off: chậm hơn ~20%)
    model.gradient_checkpointing_enable()
    
    # Cấu hình LoRA
    target_modules = find_all_linear_names(model)
    print(f"Các module mục tiêu cho LoRA: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,  # alpha = 2 * r (scaling)
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Tính toán số bước tích lũy gradient
    # Giả sử batch size hiệu dụng mong muốn là 16
    effective_batch_size = 16
    grad_accum_steps = max(1, effective_batch_size // args.batch_size) if args.batch_size < effective_batch_size else 1
    
    # Cấu hình tham số huấn luyện
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),  # eval cũng giảm batch
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        fp16=(device == "cuda"),  # MPS không hỗ trợ fp16
        logging_dir="./logs",
        logging_steps=10,
        gradient_accumulation_steps=grad_accum_steps,
        gradient_checkpointing=True,  # Tiết kiệm VRAM
        report_to="none",
        dataloader_pin_memory=False,  # Tắt pin_memory cho MPS
    )
    
    class CustomTrainer(Trainer):
        """
        Trainer tùy chỉnh để sử dụng BCEWithLogitsLoss cho Cross-Encoder binary scoring.
        """
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            # Sử dụng BCEWithLogitsLoss thay vì MSELoss mặc định của num_labels=1
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float().view(-1))
            return (loss, outputs) if return_outputs else loss
            
    print("Bắt đầu huấn luyện...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    print("Đang lưu mô hình (adapter)...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Hoàn thành! Mô hình LoRA đã được lưu tại {args.output_dir}")


if __name__ == "__main__":
    main()
