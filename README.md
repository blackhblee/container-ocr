# container-ocr

Automated container number recognition using HunyuanOCR.

## Features

- **Lightweight**: HunyuanOCR with only 1B parameters
- **Fast**: Optimized for real-time OCR tasks
- **Accurate**: State-of-the-art performance on multilingual documents
- **Hardware Support**: CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU

## Installation

### Step 1: Install transformers from specific commit

```bash
pip install git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4
```

### Step 2: Install other dependencies

```bash
pip install -r requirements.txt
```

### Optional: CUDA/GPU support

For NVIDIA GPU:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

For CPU or Apple Silicon (MPS):

```bash
pip install torch torchvision torchaudio
```

## Usage

### Batch Processing (Recommended)

```bash
python batch_process.py ./image --output results.txt
```

- Groups images by truck ID (first 6 characters of filename)
- Validates check digits (ISO 6346)
- Max 2 containers per truck
- Removes duplicates

**Output format:**

```
234855, MSMU4125810
235045, MSMU1234567, ABCU7654321
```

### Single Image

```bash
python container_ocr.py image.jpg
```

## Model Information

- **Model**: [tencent/HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR)
- **Size**: 1B parameters
- **Features**: OCR expert VLM with multilingual support
- **Applications**: Text spotting, information extraction, document parsing
