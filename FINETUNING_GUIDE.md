# Florence-2 컨테이너 OCR 파인튜닝 가이드

## 📋 준비 사항

Florence-2를 컨테이너 번호 인식에 특화시키기 위한 LoRA 파인튜닝 가이드입니다.

## 🎯 1단계: 필요한 패키지 설치

```bash
pip install -r requirements_finetune.txt
```

## 📝 2단계: 데이터셋 준비

### 2-1. Annotation 템플릿 생성

```bash
python prepare_dataset.py --mode create_template --image_folder ./image --annotation_file annotations.json
```

이 명령은 `annotations.json` 파일을 생성합니다. 각 이미지에 대해 다음과 같은 형식으로 되어 있습니다:

```json
{
  "images": [
    {
      "image_path": "image/000207_LEFT_1.jpg",
      "container_number": "TODO",
      "owner_code": "TODO",
      "serial_number": "TODO",
      "check_digit": "TODO"
    }
  ]
}
```

### 2-2. 수동으로 라벨링

`annotations.json` 파일을 열고 각 이미지를 보면서 실제 컨테이너 번호를 입력합니다:

```json
{
  "image_path": "image/000207_LEFT_1.jpg",
  "container_number": "TEMU 1234567 0",
  "owner_code": "TEMU",
  "serial_number": "1234567",
  "check_digit": "0"
}
```

**중요**:

- 정확한 라벨링이 학습 성능의 핵심입니다
- 최소 50-100개의 이미지를 라벨링하는 것을 권장합니다
- 다양한 각도와 조명 조건의 이미지를 포함시키세요

### 2-3. 학습 데이터셋 생성

```bash
python prepare_dataset.py --mode prepare --annotation_file annotations.json --output_dir ./dataset
```

이 명령은:

- `dataset/train.json`: 학습 데이터 (80%)
- `dataset/val.json`: 검증 데이터 (20%)

를 생성합니다.

## 🚀 3단계: LoRA 파인튜닝

### 기본 학습

```bash
python finetune_florence.py \
  --train dataset/train.json \
  --val dataset/val.json \
  --output ./florence-container-lora \
  --epochs 10 \
  --batch_size 2 \
  --lr 1e-4 \
  --device mps
```

### 파라미터 설명

- `--epochs`: 학습 에포크 수 (기본: 10)
- `--batch_size`: 배치 크기 (메모리에 따라 조정, 기본: 2)
- `--lr`: 학습률 (기본: 1e-4)
- `--device`: 디바이스 (cuda/mps/cpu)

### 예상 학습 시간

- 100개 이미지, 10 epochs, MPS: 약 30-60분
- CUDA GPU 사용 시 더 빠름

## 🎓 4단계: 파인튜닝된 모델 사용

`container_ocr.py`를 수정하여 파인튜닝된 모델을 사용합니다:

```python
from peft import PeftModel

# 기본 모델 로드
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-large",
    trust_remote_code=True,
    attn_implementation="eager"
)

# LoRA 어댑터 로드
model = PeftModel.from_pretrained(base_model, "./florence-container-lora")
model = model.to(device)
```

또는 새로운 클래스 만들기:

```python
class FinetunedContainerOCR(ContainerOCR):
    def __init__(self, lora_path: str = "./florence-container-lora"):
        """파인튜닝된 모델 사용"""
        # 부모 클래스 초기화 스킵하고 직접 구현
        from peft import PeftModel

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True
        )

        # 기본 모델 로드
        base_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
            attn_implementation="eager"
        )

        # LoRA 적용
        self.model = PeftModel.from_pretrained(base_model, lora_path)
        self.model = self.model.to(self.device)
        self.model.eval()
```

## 📊 5단계: 성능 평가

파인튜닝 후 성능을 비교합니다:

```bash
# 기본 모델
python batch_process.py ./image --output results_base.txt

# 파인튜닝 모델 (container_ocr.py 수정 후)
python batch_process.py ./image --output results_finetuned.txt
```

## 💡 팁

### 데이터셋 크기

- 최소: 50개 이미지
- 권장: 100-200개 이미지
- 최적: 500개 이상 이미지

### 데이터 다양성

- 다양한 각도 (정면, 측면, 위)
- 다양한 조명 (밝음, 어두움)
- 다양한 컨테이너 종류
- 다양한 거리 (가까이, 멀리)

### 학습 모니터링

학습 로그에서 `eval_loss`가 감소하는지 확인하세요:

```
Epoch 1: eval_loss: 2.34
Epoch 2: eval_loss: 1.89
Epoch 3: eval_loss: 1.45
...
```

### 하이퍼파라미터 튜닝

성능이 좋지 않으면:

- learning_rate 조정 (1e-5 ~ 1e-3)
- batch_size 증가 (메모리 허용 시)
- epochs 증가

## ⚠️ 주의사항

1. **메모리 부족**: batch_size를 줄이세요 (1로 설정)
2. **과적합**: 검증 손실이 증가하면 학습 중단
3. **라벨링 품질**: 정확한 라벨이 가장 중요합니다

## 🔧 문제 해결

### "CUDA out of memory"

```bash
python finetune_florence.py --batch_size 1
```

### "MPS backend error"

```bash
python finetune_florence.py --device cpu
```

### 학습이 너무 느림

- CUDA GPU 사용 권장
- batch_size 증가 (메모리 허용 시)
- 데이터셋 크기 조정
