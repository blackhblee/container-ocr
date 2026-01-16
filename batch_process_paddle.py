"""
배치 처리 - PaddleOCR-VL 버전
여러 컨테이너 이미지를 배치로 처리합니다
"""

from container_ocr_paddle import ContainerOCRPaddle
from pathlib import Path
from datetime import datetime


def process_folder(folder_path: str, output_file: str = "results.txt", batch_size: int = 4, lora_path: str = None):
    """
    폴더 내의 모든 이미지를 배치로 처리
    
    Args:
        folder_path: 이미지가 있는 폴더 경로
        output_file: 결과를 저장할 TXT 파일 (기본: results.txt)
        batch_size: 배치 크기 (기본: 4)
        lora_path: LoRA 가중치 경로 (선택사항)
    """
    # OCR 시스템 초기화
    ocr = ContainerOCRPaddle(lora_path=lora_path)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 처리 모드: PaddleOCR-VL (배치 크기: {batch_size})\n")
    
    # 이미지 파일 찾기
    folder = Path(folder_path)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️  {folder_path}에서 이미지를 찾을 수 없습니다")
        return
    # 파일명으로 정렬
    image_files = sorted(image_files)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📁 {len(image_files)}개의 이미지를 찾았습니다\n")
    
    # 결과 저장
    results_data = []
    total_success = 0
    total_valid = 0
    
    # 배치별로 처리
    for i in range(0, len(image_files), batch_size):
        batch_images = image_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(image_files) + batch_size - 1) // batch_size
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 배치 {batch_num}/{total_batches} 처리 중 ({len(batch_images)}개 이미지)...")
        
        try:
            # 배치 처리
            results = ocr.process_batch(batch_images)
            
            # 결과 처리
            for img_path, result in zip(batch_images, results):
                filename = img_path.name
                
                if result.get('found', False):
                    total_success += 1
                    is_valid = result.get('check_digit_valid', False)
                    container_num = result['container_number'].replace(' ', '')  # 공백 제거
                    
                    if is_valid:
                        total_valid += 1
                        results_data.append((filename, container_num, 'VALID'))
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   ✓ {filename}: {result['container_number']}")
                    else:
                        calculated = result.get('calculated_check_digit', 'N/A')
                        results_data.append((filename, container_num, f'CHECK_ERROR({calculated})'))
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   ✗ {filename}: {result['container_number']} (체크디지트 오류, 계산값: {calculated})")
                else:
                    results_data.append((filename, 'NOT_FOUND', 'ERROR'))
                    raw_output = result.get('raw_output', 'N/A')
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - {filename}: 인식 실패 (raw: {raw_output[:50]}...)")
                    
        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ✗ 배치 {batch_num} 처리 실패: {str(e)}")
            # 실패한 배치의 이미지들은 ERROR로 기록
            for img_path in batch_images:
                results_data.append((img_path.name, 'ERROR', str(e)))
        
        print()  # 배치 구분용 빈 줄
    
    # 결과 요약
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 처리 결과 요약")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 총 이미지: {len(image_files)}개")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 인식 성공: {total_success}개")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 유효한 컨테이너: {total_valid}개")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 실패: {len(image_files) - total_success}개")
    
    # TXT 파일로 저장
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for filename, container, status in results_data:
            f.write(f"{filename}\t{container}\t{status}\n")
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📄 결과 저장: {output_path.absolute()}")
    
    # 콘솔에도 출력
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 최종 결과:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "-" * 60)
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="폴더 내의 모든 컨테이너 이미지를 배치로 처리합니다 (PaddleOCR-VL)"
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="이미지가 있는 폴더 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_paddle.txt",
        help="결과를 저장할 TXT 파일 (기본: results_paddle.txt)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 크기 (기본: 4)"
    )
    parser.add_argument(
        "--lora",
        type=str,
        default=None,
        help="LoRA 가중치 경로 (예: ./paddle-container-lora)"
    )
    
    args = parser.parse_args()
    process_folder(args.folder_path, args.output, args.batch_size, args.lora)
