"""
CSV 파일을 Florence-2 파인튜닝용 JSON 데이터셋으로 변환
"""

import csv
import json
from pathlib import Path
import random
from typing import List, Dict


def csv_to_florence_dataset(
    csv_file: str,
    image_folder: str,
    output_dir: str,
    train_ratio: float = 0.8,
    skip_empty: bool = True
):
    """
    CSV 파일을 Florence-2 파인튜닝용 데이터셋으로 변환
    
    Args:
        csv_file: CSV 파일 경로 (filename, label 형식)
        image_folder: 이미지가 있는 폴더 경로
        output_dir: 출력 디렉토리
        train_ratio: 학습 데이터 비율 (기본: 0.8)
        skip_empty: 빈 라벨을 가진 항목 제외 여부 (기본: True)
    """
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    image_path = Path(image_folder)
    
    # CSV 읽기
    dataset = []
    skipped_count = 0
    missing_image_count = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename'].strip()
            label = row['label'].strip()
            
            # 빈 라벨 처리
            if not label:
                if skip_empty:
                    skipped_count += 1
                    continue
                else:
                    label = "NONE"
            
            # 이미지 파일 존재 확인
            img_file = image_path / filename
            if not img_file.exists():
                print(f"⚠️  이미지 파일 없음: {filename}")
                missing_image_count += 1
                continue
            
            # 컨테이너 번호 파싱
            parts = label.split()
            if len(parts) >= 3 and parts[0] != "NONE":
                owner_code = parts[0]
                serial_number = parts[1]
                check_digit = parts[2]
            else:
                owner_code = None
                serial_number = None
                check_digit = None
            
            dataset.append({
                "image_path": str(image_path / filename),
                "container_number": label,
                "owner_code": owner_code,
                "serial_number": serial_number,
                "check_digit": check_digit
            })
    
    print(f"\n📊 데이터셋 통계:")
    print(f"   총 항목: {len(dataset)}")
    print(f"   제외된 빈 라벨: {skipped_count}")
    print(f"   누락된 이미지: {missing_image_count}")
    
    if len(dataset) == 0:
        print("❌ 사용 가능한 데이터가 없습니다!")
        return
    
    # 데이터 셔플
    random.shuffle(dataset)
    
    # Train/Val 분할
    split_idx = int(len(dataset) * train_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Florence-2 형식으로 변환
    train_florence = []
    for item in train_data:
        train_florence.append({
            "image": item["image_path"],
            "prefix": "<OCR>",
            "suffix": item["container_number"]
        })
    
    val_florence = []
    for item in val_data:
        val_florence.append({
            "image": item["image_path"],
            "prefix": "<OCR>",
            "suffix": item["container_number"]
        })
    
    # JSON 파일로 저장
    train_file = output_path / "train.json"
    val_file = output_path / "val.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_florence, f, ensure_ascii=False, indent=2)
    
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_florence, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 데이터셋 생성 완료!")
    print(f"   학습 데이터: {len(train_florence)}개 → {train_file}")
    print(f"   검증 데이터: {len(val_florence)}개 → {val_file}")
    
    # 샘플 출력
    print(f"\n📝 샘플 데이터:")
    for i, sample in enumerate(train_florence[:3]):
        print(f"   [{i+1}] {Path(sample['image']).name}: {sample['suffix']}")
    
    return train_file, val_file


def analyze_csv(csv_file: str):
    """CSV 파일 분석"""
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    total = len(rows)
    with_label = sum(1 for row in rows if row['label'].strip())
    without_label = total - with_label
    
    # 유니크 컨테이너 번호
    unique_containers = set()
    for row in rows:
        label = row['label'].strip()
        if label:
            unique_containers.add(label)
    
    print(f"\n📊 CSV 파일 분석:")
    print(f"   총 항목: {total}개")
    print(f"   라벨 있음: {with_label}개 ({with_label/total*100:.1f}%)")
    print(f"   라벨 없음: {without_label}개 ({without_label/total*100:.1f}%)")
    print(f"   유니크 컨테이너 번호: {len(unique_containers)}개")
    
    # 상위 5개 컨테이너 번호
    if unique_containers:
        from collections import Counter
        labels = [row['label'].strip() for row in rows if row['label'].strip()]
        counter = Counter(labels)
        print(f"\n🔝 가장 많은 컨테이너 번호:")
        for label, count in counter.most_common(5):
            print(f"   {label}: {count}개")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV를 Florence-2 데이터셋으로 변환")
    parser.add_argument("--csv", type=str, required=True, help="입력 CSV 파일")
    parser.add_argument("--image_folder", type=str, default="train_image", help="이미지 폴더")
    parser.add_argument("--output_dir", type=str, default="dataset", help="출력 디렉토리")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="학습 데이터 비율")
    parser.add_argument("--include_empty", action="store_true", help="빈 라벨도 포함")
    parser.add_argument("--analyze_only", action="store_true", help="분석만 수행")
    
    args = parser.parse_args()
    
    if args.analyze_only:
        analyze_csv(args.csv)
    else:
        csv_to_florence_dataset(
            csv_file=args.csv,
            image_folder=args.image_folder,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            skip_empty=not args.include_empty
        )
