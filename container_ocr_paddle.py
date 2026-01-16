"""
컨테이너 일련번호 인식 시스템 (PaddleOCR-VL 버전)
PaddleOCR-VL을 사용하여 컨테이너 이미지에서 일련번호를 추출합니다.
"""

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
import re
from typing import Union, List, Dict, Optional
from pathlib import Path
from datetime import datetime


class ContainerOCRPaddle:
    """PaddleOCR-VL을 사용한 컨테이너 일련번호 인식 클래스"""
    
    def __init__(self, model_path: str = "PaddlePaddle/PaddleOCR-VL", lora_path: Optional[str] = None):
        """
        PaddleOCR-VL 모델 초기화
        
        Args:
            model_path: PaddleOCR-VL 기본 모델 경로
            lora_path: LoRA 가중치 경로 (선택사항)
        """
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PaddleOCR-VL 모델 초기화 중: {model_path}")
        
        if lora_path:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LoRA 가중치: {lora_path}")
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ CUDA 감지됨: {gpu_name}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ MPS(Apple Silicon) 감지됨")
        else:
            self.device = "cpu"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 사용 디바이스: {self.device}")
        
        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
        )
        
        # LoRA 가중치 로드 (있는 경우)
        if lora_path:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] LoRA 가중치 로딩 중...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ LoRA 가중치 로드 완료")
        
        self.model = self.model.to(self.device).eval()
        
        # Processor 로드
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # OCR 태스크 프롬프트
        self.ocr_prompt = "OCR:"
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PaddleOCR-VL 모델 초기화 완료!")
    
    def extract_container_number(self, image_path: Union[str, Path]) -> Dict[str, any]:
        """
        이미지에서 컨테이너 일련번호를 추출
        
        Args:
            image_path: 컨테이너 이미지 경로
            
        Returns:
            추출된 컨테이너 정보 딕셔너리
        """
        # Path 객체를 문자열로 변환
        image_path_str = str(image_path)
        
        # 이미지 로드
        image = Image.open(image_path_str).convert('RGB')
        
        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.ocr_prompt},
                ]
            }
        ]
        
        # 입력 준비
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=1024)
        
        # 디코딩
        response = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 모델 출력: '{response}'")
        
        # 컨테이너 번호 파싱
        container_info = self._parse_container_number(response)
        container_info['raw_output'] = response
        container_info['image_path'] = image_path_str
        
        return container_info
    
    def extract_container_number_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, any]]:
        """
        여러 이미지를 배치로 처리
        
        Args:
            image_paths: 컨테이너 이미지 경로 리스트
            
        Returns:
            추출된 컨테이너 정보 딕셔너리 리스트
        """
        if not image_paths:
            return []
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 배치 처리 시작: {len(image_paths)}개 이미지")
        
        results = []
        
        # PaddleOCR-VL은 배치 처리가 복잡하므로 개별 처리
        for idx, image_path in enumerate(image_paths):
            try:
                filename = Path(image_path).name
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] [{idx+1}/{len(image_paths)}] 처리 중: {filename}")
                
                result = self.extract_container_number(image_path)
                results.append(result)
                
            except Exception as e:
                filename = Path(image_path).name
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ 이미지 처리 오류: {filename} - {str(e)}")
                results.append({
                    'image_path': str(image_path),
                    'container_number': None,
                    'owner_code': None,
                    'serial_number': None,
                    'check_digit': None,
                    'check_digit_valid': None,
                    'calculated_check_digit': None,
                    'found': False,
                    'raw_output': f'Error: {str(e)}'
                })
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 배치 처리 완료")
        return results
    
    def _calculate_check_digit(self, owner_code: str, serial_number: str) -> int:
        """
        ISO 6346 규격에 따라 체크 디지트를 계산
        
        Args:
            owner_code: 4글자 소유자 코드
            serial_number: 6-7자리 일련번호
            
        Returns:
            계산된 체크 디지트 (0-9)
        """
        # ISO 6346 문자-숫자 매핑
        char_values = {
            'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17,
            'H': 18, 'I': 19, 'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25,
            'O': 26, 'P': 27, 'Q': 28, 'R': 29, 'S': 30, 'T': 31, 'U': 32,
            'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
        }
        
        # 소유자 코드 + 일련번호를 결합
        container_id = owner_code + serial_number
        
        # 각 문자/숫자에 대해 값 계산
        total = 0
        for i, char in enumerate(container_id):
            if char.isalpha():
                value = char_values[char]
            else:
                value = int(char)
            
            # 위치에 따른 가중치 (2^position)
            weight = 2 ** i
            total += value * weight
        
        # 11로 나눈 나머지
        remainder = total % 11
        
        # 나머지가 10이면 0으로 처리
        check_digit = 0 if remainder == 10 else remainder
        
        return check_digit
    
    def _parse_container_number(self, text: str) -> Dict[str, str]:
        """
        모델 출력에서 ISO 6346 규격 컨테이너 번호를 파싱
        
        Args:
            text: 모델의 출력 텍스트
            
        Returns:
            파싱된 컨테이너 정보
        """
        # 점이나 특수문자를 공백으로 치환
        cleaned = text.replace('.', ' ').replace(',', ' ').replace('-', ' ').replace('_', ' ')
        
        # ISO 6346 규격 패턴 (4글자 + 6~7숫자 + 1체크디지트)
        patterns = [
            r'([A-Z]{4})\s*(\d{6})\s*(\d)',    # 4글자 + 6숫자 + 1체크디지트
            r'([A-Z]{4})\s*(\d{7})\s*(\d)',    # 4글자 + 7숫자 + 1체크디지트
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned)
            if match:
                owner_code = match.group(1)
                serial_number = match.group(2)
                check_digit = match.group(3)
                
                # ISO 6346 체크 디지트 검증
                calculated_check_digit = self._calculate_check_digit(owner_code, serial_number)
                is_valid = (int(check_digit) == calculated_check_digit)
                
                if not is_valid:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [WARNING] 체크 디지트 불일치: 인식된={check_digit}, 계산된={calculated_check_digit}")
                
                return {
                    'container_number': f"{owner_code} {serial_number} {check_digit}",
                    'owner_code': owner_code,
                    'serial_number': serial_number,
                    'check_digit': check_digit,
                    'check_digit_valid': is_valid,
                    'calculated_check_digit': str(calculated_check_digit),
                    'found': True
                }
        
        return {
            'container_number': None,
            'owner_code': None,
            'serial_number': None,
            'check_digit': None,
            'check_digit_valid': None,
            'calculated_check_digit': None,
            'found': False
        }
    
    def process_batch(self, image_paths: List[Union[str, Path]], batch_size: int = None) -> List[Dict[str, any]]:
        """
        여러 이미지를 배치로 처리
        
        Args:
            image_paths: 이미지 경로 리스트
            batch_size: 배치 크기 (None이면 전체를 순차 처리)
            
        Returns:
            추출된 컨테이너 정보 리스트
        """
        return self.extract_container_number_batch(image_paths)


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="컨테이너 이미지에서 일련번호를 인식합니다 (PaddleOCR-VL)"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="컨테이너 이미지 파일 경로"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA 가중치 경로 (예: ./paddle-container-lora)"
    )
    
    args = parser.parse_args()
    
    # OCR 시스템 초기화
    ocr = ContainerOCRPaddle(lora_path=args.lora)
    
    # 컨테이너 번호 추출
    result = ocr.extract_container_number(args.image_path)
    
    # 결과 출력
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \n" + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 컨테이너 번호 인식 결과")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 이미지: {result['image_path']}")
    
    if result['found']:
        valid_icon = "✓" if result.get('check_digit_valid', False) else "✗"
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \n✓ 컨테이너 번호 발견!")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 전체 번호: {result['container_number']}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 소유자 코드: {result['owner_code']}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 일련번호: {result['serial_number']}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 체크 디지트: {result['check_digit']} ({valid_icon} {'유효' if result.get('check_digit_valid') else '무효'})")
        if not result.get('check_digit_valid', False):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 올바른 체크 디지트: {result.get('calculated_check_digit', 'N/A')}")
    else:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \n✗ 컨테이너 번호를 찾을 수 없습니다")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] \n원본 출력:\n{result['raw_output']}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)


if __name__ == "__main__":
    main()
