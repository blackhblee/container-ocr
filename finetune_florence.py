"""
Florence-2 LoRA 파인튜닝 스크립트
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
    """컨테이너 OCR 데이터셋"""
    
    def __init__(self, json_file: str, processor, base_dir: str = "."):
        """
        Args:
            json_file: 데이터셋 JSON 파일
            processor: Florence-2 processor
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
        prefix = item["prefix"]  # "<OCR_WITH_REGION>"
        suffix = item["suffix"]  # "TEMU 1234567 0"
        
        # Processor로 입력 준비
        inputs = self.processor(
            text=prefix,
            images=image,
            return_tensors="pt"
        )
        
        # 배치 차원 제거
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # suffix(정답)를 labels로 토크나이징
        labels = self.processor.tokenizer(
            suffix,
            return_tensors="pt",
            padding=False,
            truncation=True
        )
        
        inputs["labels"] = labels["input_ids"].squeeze(0)
        
        return inputs


def collate_fn(batch):
    """데이터 배치 생성"""
    # 모든 항목을 하나의 딕셔너리로 합치기
    keys = batch[0].keys()
    
    collated = {}
    for key in keys:
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor):
            # 패딩이 필요한 경우
            if key in ['input_ids', 'attention_mask', 'labels']:
                # 최대 길이로 패딩
                max_len = max(v.shape[0] for v in values)
                padded = []
                for v in values:
                    pad_len = max_len - v.shape[0]
                    if pad_len > 0:
                        # labels는 -100으로 패딩 (loss 계산 시 무시됨)
                        if key == 'labels':
                            padding = torch.full((pad_len,), -100, dtype=v.dtype)
                        else:
                            padding = torch.zeros(pad_len, dtype=v.dtype)
                        v = torch.cat([v, padding])
                    padded.append(v)
                collated[key] = torch.stack(padded)
            else:
                collated[key] = torch.stack(values)
        else:
            collated[key] = values
    
    return collated


def setup_lora_model(model_name: str, device: str = "mps"):
    """
    LoRA를 적용한 Florence-2 모델 준비
    
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
        attn_implementation="eager"
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


def train_florence(
    model_name: str = "microsoft/Florence-2-large",
    train_file: str = "dataset/train.json",
    val_file: str = "dataset/val.json",
    output_dir: str = "./florence-container-lora",
    num_epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    device: str = "mps"
):
    """
    Florence-2 LoRA 파인튜닝
    
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
    print("Florence-2 LoRA 파인튜닝 시작")
    print("="*60)
    
    # 모델 및 processor 준비
    print("\n1. 모델 로딩 중...")
    model, processor = setup_lora_model(model_name, device)
    
    # 디바이스로 이동 (자동 감지)
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    elif device == "mps" and torch.backends.mps.is_available():
        model = model.to("mps")
        print("   디바이스: MPS (Apple Silicon)")
    else:
        model = model.to("cpu")
        print("   디바이스: CPU")
    
    
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
        fp16=False,  # MPS는 fp16 미지원
        report_to="none",  # wandb 등 사용 안 함
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
    
    parser = argparse.ArgumentParser(description="Florence-2 LoRA 파인튜닝")
    parser.add_argument("--model", type=str, default="microsoft/Florence-2-large",
                        help="기본 모델 이름")
    parser.add_argument("--train", type=str, default="dataset/train.json",
                        help="학습 데이터 JSON")
    parser.add_argument("--val", type=str, default="dataset/val.json",
                        help="검증 데이터 JSON")
    parser.add_argument("--output", type=str, default="./florence-container-lora",
                        help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, default=10,
                        help="에포크 수")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="배치 크기")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="학습률")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "mps", "cpu"],
                        help="디바이스")
    
    args = parser.parse_args()
    
    train_florence(
        model_name=args.model,
        train_file=args.train,
        val_file=args.val,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )
