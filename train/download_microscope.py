"""
현미경 이미지 데이터셋 다운로드 스크립트
Roboflow에서 BGA Solder Ball 데이터셋을 다운로드합니다.
"""

import os
from roboflow import Roboflow

# Roboflow API 키
API_KEY = "inKCCaiy8WoPDlbg0o8P"

# 데이터 저장 경로
DATA_DIR = "../data/microscope"
os.makedirs(DATA_DIR, exist_ok=True)

def download_bga_solder_ball():
    """BGA Solder Ball 데이터셋 다운로드"""
    print("=" * 60)
    print("BGA Solder Ball 데이터셋 다운로드 시작...")
    print("=" * 60)
    
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("paulo-correa").project("bga_solder_ball")
    version = project.version(2)
    
    dataset = version.download(
        "yolov8",
        location=os.path.join(DATA_DIR, "bga_solder_ball")
    )
    
    print(f"\n✓ BGA Solder Ball 데이터셋 다운로드 완료: {dataset.location}")
    return dataset

def download_solder_ball():
    """Solder Ball 데이터셋 다운로드"""
    print("\n" + "=" * 60)
    print("Solder Ball 데이터셋 다운로드 시작...")
    print("=" * 60)
    
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace("project-1mwgl").project("solder-ball-wnejr")
    version = project.version(61)
    
    dataset = version.download(
        "yolov8",
        location=os.path.join(DATA_DIR, "solder_ball")
    )
    
    print(f"\n✓ Solder Ball 데이터셋 다운로드 완료: {dataset.location}")
    return dataset

def merge_datasets():
    """두 데이터셋을 병합하여 하나의 학습 데이터셋 생성"""
    print("\n" + "=" * 60)
    print("데이터셋 병합 중...")
    print("=" * 60)
    
    import shutil
    import yaml
    
    merged_dir = os.path.join(DATA_DIR, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    os.makedirs(os.path.join(merged_dir, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(merged_dir, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(merged_dir, "valid/images"), exist_ok=True)
    os.makedirs(os.path.join(merged_dir, "valid/labels"), exist_ok=True)
    
    # BGA Solder Ball 복사
    bga_dir = os.path.join(DATA_DIR, "bga_solder_ball")
    if os.path.exists(os.path.join(bga_dir, "train")):
        for split in ["train", "valid"]:
            src_img = os.path.join(bga_dir, split, "images")
            src_lbl = os.path.join(bga_dir, split, "labels")
            dst_img = os.path.join(merged_dir, split, "images")
            dst_lbl = os.path.join(merged_dir, split, "labels")
            
            if os.path.exists(src_img):
                for f in os.listdir(src_img):
                    shutil.copy(os.path.join(src_img, f), os.path.join(dst_img, f"bga_{f}"))
            if os.path.exists(src_lbl):
                for f in os.listdir(src_lbl):
                    shutil.copy(os.path.join(src_lbl, f), os.path.join(dst_lbl, f"bga_{f}"))
    
    # Solder Ball 복사
    sb_dir = os.path.join(DATA_DIR, "solder_ball")
    if os.path.exists(os.path.join(sb_dir, "train")):
        for split in ["train", "valid"]:
            src_img = os.path.join(sb_dir, split, "images")
            src_lbl = os.path.join(sb_dir, split, "labels")
            dst_img = os.path.join(merged_dir, split, "images")
            dst_lbl = os.path.join(merged_dir, split, "labels")
            
            if os.path.exists(src_img):
                for f in os.listdir(src_img):
                    shutil.copy(os.path.join(src_img, f), os.path.join(dst_img, f"sb_{f}"))
            if os.path.exists(src_lbl):
                for f in os.listdir(src_lbl):
                    shutil.copy(os.path.join(src_lbl, f), os.path.join(dst_lbl, f"sb_{f}"))
    
    # data.yaml 생성
    data_yaml = {
        "path": merged_dir,
        "train": "train/images",
        "val": "valid/images",
        "names": {
            0: "Type1",
            1: "Type2",
            2: "Type3",
            3: "Type4"
        },
        "nc": 4
    }
    
    with open(os.path.join(merged_dir, "data.yaml"), "w") as f:
        yaml.dump(data_yaml, f)
    
    print(f"\n✓ 데이터셋 병합 완료: {merged_dir}")
    print(f"  - Train 이미지: {len(os.listdir(os.path.join(merged_dir, 'train/images')))}장")
    print(f"  - Valid 이미지: {len(os.listdir(os.path.join(merged_dir, 'valid/images')))}장")

def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("현미경 이미지 데이터셋 다운로드 시작")
    print("=" * 60 + "\n")
    
    # 데이터셋 다운로드
    download_bga_solder_ball()
    download_solder_ball()
    
    # 데이터셋 병합
    merge_datasets()
    
    print("\n" + "=" * 60)
    print("✓ 모든 작업 완료!")
    print("=" * 60)
    print(f"\n학습 데이터 경로: {os.path.join(DATA_DIR, 'merged')}")
    print("다음 단계: python train_microscope.py")

if __name__ == "__main__":
    main()
