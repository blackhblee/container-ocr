# container-ocr

Automated container number recognition using Qwen3-VL.

## Installation

```bash
pip install -r requirements.txt
```

For CUDA/GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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
