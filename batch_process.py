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
        
        # 트럭 전체 이미지 분석
        analysis = ocr.analyze_truck_containers(images)
        
        # 분석 결과 출력
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📊 분석 결과:")
        
        type_description = {
            'none': '컨테이너 없음',
            '40ft_single': '40피트 1개',
            '20ft_single_front': '20피트 1개 (앞쪽)',
            '20ft_single_rear': '20피트 1개 (뒤쪽)',
            '20ft_double': '20피트 2개',
            'single': '1개 (40피트 또는 20피트)',
            'double': '20피트 2개',
            'unknown': '알 수 없음'
        }.get(analysis['container_type'], analysis['container_type'])
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 컨테이너 타입: {type_description}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   - 감지된 컨테이너 수: {analysis['container_count']}")
        
        if analysis['container_count'] > 0:
            for i, container in enumerate(analysis['containers'], 1):
                position_str = {
                    'full': '40피트 (전체)',
                    'front': '앞쪽',
                    'rear': '뒤쪽',
                    'unknown': '위치 미상'
                }.get(container.get('position', 'unknown'), container.get('position', 'unknown'))
                
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]   [{i}] {container['number']}")
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]       위치: {position_str}")
                
                # detection_count가 있으면 출력 (폴백 방식)
                if 'detection_count' in container:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]       감지 횟수: {container['detection_count']}회")
                if 'images' in container:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]       감지 이미지: {', '.join(container['images'][:3])}{'...' if len(container['images']) > 3 else ''}")
            
            # 컨테이너 번호만 저장 (위치 순서대로: front → full → rear)
            containers_sorted = sorted(
                analysis['containers'],
                key=lambda x: (0 if x.get('position') == 'front' else 1 if x.get('position') == 'full' else 2)
            )
            truck_containers[truck_id] = [c['number'] for c in containers_sorted]
            
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📦 트럭 {truck_id}: {len(truck_containers[truck_id])}개 컨테이너 확정")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ⚠️ 트럭 {truck_id}: 유효한 컨테이너를 찾지 못했습니다")
    
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
