"""
배치 처리 - 트럭별 컨테이너 번호 인식
여러 컨테이너 이미지를 한 번에 처리하고 트럭별로 그룹화합니다
"""

from container_ocr import ContainerOCR
from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime


def process_folder(folder_path: str, output_file: str = "results.txt"):
    """
    폴더 내의 모든 이미지를 처리하고 트럭별로 그룹화
    
    Args:
        folder_path: 이미지가 있는 폴더 경로
        output_file: 결과를 저장할 TXT 파일 (기본: results.txt)
    """
    # OCR 시스템 초기화
    ocr = ContainerOCR()
    
    # 디바이스에 따라 배치 크기 설정
    # GPU: 배치 처리로 성능 향상
    # CPU: 순차 처리로 안정성 확보
    batch_size = 4 if ocr.device == "cuda" else 1
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 처리 모드: {ocr.device.upper()} (배치 크기: {batch_size})\n")
    
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
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📁 {len(image_files)}개의 이미지를 찾았습니다\n")
    
    # 트럭별로 이미지 그룹화 (파일명 앞 6자리 기준)
    truck_images = defaultdict(list)
    for img_path in image_files:
        filename = img_path.name
        # 파일명 앞 6자리 추출
        if len(filename) >= 6:
            truck_id = filename[:6]
            truck_images[truck_id].append(img_path)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚛 {len(truck_images)}대의 트럭을 찾았습니다")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 트럭 ID: {', '.join(sorted(truck_images.keys()))}\n")
    
    # 트럭별로 컨테이너 번호 수집
    truck_containers = {}
    
    for truck_id in sorted(truck_images.keys()):
        images = truck_images[truck_id]
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 처리 중: 트럭 {truck_id} ({len(images)}개 이미지)")
        
        valid_containers = set()  # 유효한 컨테이너 번호 (중복 제거)
        
        # 배치 단위로 이미지 처리
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   배치 처리: {len(batch)}개 이미지 ({i+1}~{min(i+batch_size, len(images))})")
            
            # 배치로 한 번에 처리
            results = ocr.process_batch(batch)
            
            # 결과 처리
            for img_path, result in zip(batch, results):
                if result.get('found', False):
                    # container_ocr에서 이미 체크디지트 검증을 수행함
                    is_valid = result.get('check_digit_valid', False)
                    container_num = result['container_number'].replace(' ', '')  # 공백 제거
                    
                    if is_valid:
                        valid_containers.add(container_num)
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     ✓ {img_path.name}: {result['container_number']} (유효)")
                    else:
                        calculated = result.get('calculated_check_digit', 'N/A')
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     ✗ {img_path.name}: {result['container_number']} (체크디지트 오류, 계산값: {calculated})")
                else:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]     - {img_path.name}: 인식 실패 (raw: {result.get('raw_output', 'N/A')})")
        
        # 최대 2개까지만 저장
        if valid_containers:
            truck_containers[truck_id] = sorted(list(valid_containers))[:2]
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📦 트럭 {truck_id}: {len(truck_containers[truck_id])}개 컨테이너 확정")
    
    # 결과 요약
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 처리 결과 요약")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "="*60)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 총 트럭: {len(truck_images)}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 컨테이너 인식 성공: {len(truck_containers)}대 트럭")
    
    # TXT 파일로 저장
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for truck_id in sorted(truck_containers.keys()):
            containers = truck_containers[truck_id]
            line = f"{truck_id}, {', '.join(containers)}\n"
            f.write(line)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 💾 결과가 {output_path}에 저장되었습니다")
    
    # 콘솔에도 출력
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 최종 결과:")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] " + "-" * 60)
    with open(output_path, 'r', encoding='utf-8') as f:
        print(f.read())


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="폴더 내의 모든 컨테이너 이미지를 처리하고 트럭별로 그룹화합니다"
    )
    parser.add_argument(
        "folder_path",
        type=str,
        help="이미지가 있는 폴더 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.txt",
        help="결과를 저장할 TXT 파일 (기본: results.txt)"
    )
    
    args = parser.parse_args()
    process_folder(args.folder_path, args.output)
