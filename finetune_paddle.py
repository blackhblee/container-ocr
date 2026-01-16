"""
PaddleOCR-VL LoRA 파인튜닝 스크립트
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from PIL import Image
from pathlib import Path
from typing import Dict, List
import numpy as np


class ContainerOCRDataset(Dataset):
    """컨테이너 OCR 데이터셋 (PaddleOCR-VL용)"""
    
    def __init__(self, json_file: str, processor, base_dir: str = "."):
        """
        Args:
            json_file: 데이터셋 JSON 파일
            processor: PaddleOCR-VL processor
            base_dir: 이미지 파일의 기본 디렉토리
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.base_dir = Path(base_dir)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 이미지 로드
        image_path = self.base_dir / item["image"]
        image = Image.open(image_path).convert('RGB')
        
        # 프롬프트와 정답
        task_prompt = "OCR:"
        answer = item["suffix"]  # "TEMU 1234567 0"
        
        # Chat template 사용 - User(image+text) / Assistant 형식
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task_prompt},
                ]
            },
            {
                "role": "assistant", 
                "content": [
                    {"type": "text", "text": answer}
                ]
            }
        ]
        
        # Chat template 적용 (processor.apply_chat_template 사용)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Labels 생성: assistant 메시지만 학습
        # User 메시지만으로 길이 확인
        user_only_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": task_prompt},
                ]
            }
        ]
        user_inputs = self.processor.apply_chat_template(
            user_only_messages,
            tokenize=True,
            add_generation_prompt=True,  # assistant 시작 부분까지 포함
            return_dict=True,
            return_tensors="pt"
        )
        user_length = user_inputs["input_ids"].shape[-1]
        
        # Labels 복사 후 user 부분 마스킹
        labels = inputs["input_ids"].clone()
        labels[:, :user_length] = -100
        
        inputs["labels"] = labels
        
        return inputs


def collate_fn(batch):
    """데이터 배치 생성 (batch_size=1 전용)"""
    # PaddleOCR-VL의 dynamic resolution 때문에 batch_size=1만 지원
    if len(batch) != 1:
        raise ValueError("PaddleOCR-VL 파인튜닝은 batch_size=1만 지원합니다")
    
    # 단일 샘플 반환 (이미 배치 차원 포함)
    return batch[0]


def setup_lora_model(model_name: str, device: str = "cuda"):
    """
    LoRA를 적용한 PaddleOCR-VL 모델 준비
    
    Args:
        model_name: 모델 이름
        device: 디바이스
        
    Returns:
        모델, processor
    """
    
    # Processor 로드
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=8,  # LoRA rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Attention layer에 적용
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # LoRA 적용
    model = get_peft_model(model, lora_config)
    
    # 학습 가능한 파라미터 출력
    model.print_trainable_parameters()
    
    return model, processor


def train_paddle(
    model_name: str = "PaddlePaddle/PaddleOCR-VL",
    train_file: str = "dataset/train.json",
    val_file: str = "dataset/val.json",
    output_dir: str = "./paddle-container-lora",
    num_epochs: int = 7,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    device: str = "cuda"
):
    """
    PaddleOCR-VL LoRA 파인튜닝
    
    Args:
        model_name: 기본 모델 이름
        train_file: 학습 데이터 JSON
        val_file: 검증 데이터 JSON
        output_dir: 출력 디렉토리
        num_epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        device: 디바이스
    """
    
    print("="*60)
    print("PaddleOCR-VL LoRA 파인튜닝 시작")
    print("="*60)
    
    # 모델 및 processor 준비
    print("\n1. 모델 로딩 중...")
    
    # 디바이스 자동 감지
    if device == "cuda":
        if torch.cuda.is_available():
            actual_device = "cuda"
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚠️  CUDA 요청했지만 사용 불가, CPU로 대체")
            actual_device = "cpu"
    elif device == "mps":
        if torch.backends.mps.is_available():
            actual_device = "mps"
            print("   디바이스: MPS (Apple Silicon)")
        else:
            print("   ⚠️  MPS 요청했지만 사용 불가, CPU로 대체")
            actual_device = "cpu"
    else:
        actual_device = "cpu"
        print("   디바이스: CPU")
    
    model, processor = setup_lora_model(model_name, actual_device)
    model = model.to(actual_device)
    
    # 데이터셋 준비
    print("\n2. 데이터셋 로딩 중...")
    train_dataset = ContainerOCRDataset(train_file, processor)
    val_dataset = ContainerOCRDataset(val_file, processor)
    
    print(f"   학습 데이터: {len(train_dataset)}개")
    print(f"   검증 데이터: {len(val_dataset)}개")
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=True if actual_device == "cuda" else False,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # 학습 시작
    print("\n3. 학습 시작...")
    trainer.train()
    
    # 모델 저장
    print(f"\n4. 모델 저장 중: {output_dir}")
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60)
    print(f"저장 위치: {output_dir}")
    print("\n사용 방법:")
    print(f"  from peft import PeftModel")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{model_name}', trust_remote_code=True)")
    print(f"  model = PeftModel.from_pretrained(model, '{output_dir}')")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PaddleOCR-VL LoRA 파인튜닝")
    parser.add_argument("--model", type=str, default="PaddlePaddle/PaddleOCR-VL",
                        help="기본 모델 이름")
    parser.add_argument("--train", type=str, default="dataset/train.json",
                        help="학습 데이터 JSON")
    parser.add_argument("--val", type=str, default="dataset/val.json",
                        help="검증 데이터 JSON")
    parser.add_argument("--output", type=str, default="./paddle-container-lora",
                        help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=7,
                        help="에포크 수")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="배치 크기 (PaddleOCR-VL은 1만 지원)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="학습률")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="디바이스")
    
    args = parser.parse_args()
    
    train_paddle(
        model_name=args.model,
        train_file=args.train,
        val_file=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
