"""
컨테이너 일련번호 인식 시스템
HunyuanOCR을 사용하여 컨테이너 이미지에서 일련번호를 추출합니다.
"""

from transformers import AutoProcessor, HunYuanVLForConditionalGeneration
from PIL import Image
import torch
import re
from typing import Union, List, Dict
from pathlib import Path
from datetime import datetime


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:] 
        count = 0
        i = n - length
        
        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]  

    return text


class ContainerOCR:
    """컨테이너 일련번호를 인식하는 클래스"""
    
    def __init__(self, model_name: str = "tencent/HunyuanOCR"):
        """
        HunyuanOCR 초기화
        
        Args:
            model_name: 사용할 HunyuanOCR 모델
        """
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HunyuanOCR 초기화 중: {model_name}")
        
        # GPU 사용 가능 여부 확인
        if torch.cuda.is_available():
            self.device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ CUDA 감지됨: {gpu_name}")
            use_dtype = torch.bfloat16
            use_device_map = "auto"
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✓ MPS(Apple Silicon) 감지됨")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  MPS는 bfloat16을 완전히 지원하지 않아 float32 사용")
            use_dtype = torch.float32
            use_device_map = None
        else:
            self.device = "cpu"
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠ GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            use_dtype = torch.float32
            use_device_map = None
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 사용 디바이스: {self.device}, dtype: {use_dtype}")
        
        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        
        # 모델 로드
        self.model = HunYuanVLForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=use_dtype if self.device == "cuda" else None,  # MPS/CPU는 None
            device_map=use_device_map
        )
        
        # MPS/CPU의 경우 명시적으로 float32로 변환 후 디바이스로 이동
        if self.device in ["mps", "cpu"]:
            self.model = self.model.float()  # 명시적으로 float32로 변환
            self.model = self.model.to(self.device)
        
        # Inference mode로 설정 (속도 향상)
        self.model.eval()
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] HunyuanOCR 초기화 완료!")
    
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
        image_inputs = Image.open(image_path_str)
        
        # 프롬프트 구성 - 영어로 컨테이너 번호 추출 요청
        messages = [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path_str},
                    {"type": "text", "text": """Read all text visible on this shipping container.

Look for the container identification number painted on the container:
- First part: 4 capital letters
- Second part: 6 or 7 digits
- Third part: 1 digit

IMPORTANT: Only return the actual text you see in THIS image. Do not make up or guess any numbers. If you cannot clearly read a complete container number in this image, say NONE.

Response format: Just write the letters and numbers with spaces, like: ABCD 1234567 8"""}
                ]
            }
        ]
        
        # 메시지 템플릿 적용
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 입력 준비
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # 추론
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        
        # 입력 제거하고 디코딩
        if "input_ids" in inputs:
            input_ids = inputs.input_ids
        else:
            input_ids = inputs.inputs
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        response = clean_repeated_substrings(
            self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]
        )
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 모델 출력: '{response}'")
        
        # 컨테이너 번호 파싱
        container_info = self._parse_container_number(response, [response])
        container_info['raw_output'] = response
        container_info['all_detected_text'] = [response]
        container_info['image_path'] = image_path_str
        
        return container_info
    
    def extract_container_number_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, any]]:
        """
        여러 이미지를 하나의 프롬프트에 넣어서 한 번에 추론
        
        Args:
            image_paths: 컨테이너 이미지 경로 리스트
            
        Returns:
            추출된 컨테이너 정보 딕셔너리 리스트
        """
        if not image_paths:
            return []
        
        # Path 객체를 문자열로 변환
        image_path_strs = [str(p) for p in image_paths]
        
        # 이미지 로드
        images = [Image.open(path) for path in image_path_strs]
        
        # 하나의 메시지에 모든 이미지 포함
        content = []
        for idx, image_path_str in enumerate(image_path_strs, 1):
            content.append({"type": "image", "image": image_path_str})
        
        # 텍스트 프롬프트 추가
        content.append({
            "type": "text", 
            "text": f"""You are analyzing {len(images)} shipping container images.

For EACH image, find the container identification number painted on the container:
- First part: 4 capital letters
- Second part: 6 or 7 digits
- Third part: 1 digit

IMPORTANT RULES:
1. Analyze each image separately
2. Return results for ALL {len(images)} images in order
3. If an image has no clear container number, write "NONE" for that image
4. Do not make up or guess numbers

Response format (one line per image):
Image 1: XXXX YYYYYYY Z
Image 2: XXXX YYYYYYY Z
Image 3: NONE
...

Analyze all {len(images)} images now:"""
        })
        
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": content}
        ]
        
        # 메시지 템플릿 적용
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 입력 준비
        inputs = self.processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )
        
        # 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)
        
        # 추론
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 배치 추론 시작: {len(image_paths)}개 이미지를 하나의 프롬프트로")
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 배치 추론 완료")
        
        # 입력 제거하고 디코딩
        if "input_ids" in inputs:
            input_ids = inputs.input_ids
        else:
            input_ids = inputs.inputs
        
        generated_trimmed = generated_ids[0][len(input_ids[0]):]
        response = clean_repeated_substrings(
            self.processor.decode(
                generated_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
        )
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] 전체 응답:\n{response}")
        
        # 응답 파싱 - "Image N:" 형식으로 분리
        results = []
        lines = response.split('\n')
        
        for idx, image_path_str in enumerate(image_path_strs, 1):
            # 해당 이미지 번호의 응답 찾기
            container_text = None
            for line in lines:
                if f"Image {idx}:" in line or f"image {idx}:" in line:
                    container_text = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                    break
            
            if not container_text:
                # 라인별로 순차적으로 파싱 시도
                if idx - 1 < len(lines):
                    container_text = lines[idx - 1].strip()
            
            filename = Path(image_path_str).name
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG] [{idx}/{len(image_paths)}] {filename}: '{container_text}'")
            
            # 컨테이너 번호 파싱
            if container_text:
                container_info = self._parse_container_number(container_text, [container_text])
                container_info['raw_output'] = container_text
                container_info['all_detected_text'] = [container_text]
            else:
                container_info = {
                    'container_number': None,
                    'owner_code': None,
                    'serial_number': None,
                    'check_digit': None,
                    'check_digit_valid': None,
                    'calculated_check_digit': None,
                    'found': False,
                    'raw_output': 'No response',
                    'all_detected_text': []
                }
            
            container_info['image_path'] = image_path_str
            results.append(container_info)
        
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
    
    def _parse_container_number(self, text: str, text_list: List[str] = None) -> Dict[str, str]:
        """
        모델 출력에서 ISO 6346 규격 컨테이너 번호를 파싱
        
        Args:
            text: 모델의 출력 텍스트 (결합된 전체 텍스트)
            text_list: 개별 인식된 텍스트 리스트
            
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
            batch_size: 배치 크기 (None이면 전체를 한 번에 처리)
            
        Returns:
            추출된 컨테이너 정보 리스트
        """
        if not image_paths:
            return []
        
        # batch_size가 None이면 전체를 한 번에 처리
        if batch_size is None:
            batch_size = len(image_paths)
        
        results = []
        
        # 배치 단위로 처리
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # 배치 추론 사용
                batch_results = self.extract_container_number_batch(batch_paths)
                results.extend(batch_results)
            except Exception as e:
                # 배치 처리 실패 시 개별 처리로 폴백
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  배치 처리 실패, 개별 처리로 전환: {str(e)}")
                for image_path in batch_paths:
                    try:
                        result = self.extract_container_number(image_path)
                        results.append(result)
                    except Exception as e2:
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ 이미지 처리 오류: {Path(image_path).name} - {str(e2)}")
                        results.append({
                            'image_path': str(image_path),
                            'found': False,
                            'error': str(e2)
                        })
        
        return results


def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="컨테이너 이미지에서 일련번호를 인식합니다 (HunyuanOCR)"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="컨테이너 이미지 파일 경로"
    )
    
    args = parser.parse_args()
    
    # OCR 시스템 초기화
    ocr = ContainerOCR()
    
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
