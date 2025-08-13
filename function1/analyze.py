import sys
import os
import json
from collections import defaultdict
import pandas as pd
import numpy as np # 숫자 계산을 위해 numpy 추가
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

# --- 1. 라이브러리 경로 설정 ---
# nuScenes SDK가 설치된 경로를 지정하세요.
sdk_path = '/workspace/nuscenes-devkit/python-sdk'
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from nuscenes.nuscenes import NuScenes

# --- 2. 설정값 ---
# metadata.py에서 지정한 경로/버전과 정확히 일치시켜야 합니다.
DATAROOT = '/workspace/nuscenes_from_real_data'
VERSION = 'v1.0-real-data'

def analyze_real_data():
    """실제 변환된 데이터셋을 분석하고 심층 리포트를 생성하는 메인 함수"""
    
    # --- 데이터셋 로딩 ---
    print(f"'{DATAROOT}'에서 '{VERSION}' 버전의 데이터셋을 로딩합니다...")
    try:
        nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)
        print("✅ 데이터셋 로딩 완료!")
    except Exception as e:
        print(f"❌ 데이터셋 로딩 실패: {e}")
        return

    # object_ann.json 수동 로딩
    try:
        object_ann_path = os.path.join(nusc.dataroot, nusc.version, 'object_ann.json')
        with open(object_ann_path) as f:
            nusc.object_ann = json.load(f)
        print(f"✅ 'object_ann' 테이블에 {len(nusc.object_ann)}개의 주석을 로드했습니다.")
    except FileNotFoundError:
        print("❌ 'object_ann.json' 파일을 찾을 수 없습니다.")
        return

    # --- 통계 집계를 위한 데이터 구조 초기화 ---
    category_counts = defaultdict(int)
    bbox_areas = defaultdict(list)
    sample_token_to_obj_count = defaultdict(int)

    # --- 메인 분석 루프 ---
    print("\n객체 주석(object_ann)을 순회하며 통계 집계 중...")
    if not nusc.object_ann:
        print("경고: 분석할 주석 데이터가 없습니다.")
    else:
        for ann in tqdm(nusc.object_ann, desc="어노테이션 통계 분석"):
            # 1. 카테고리별 개수 집계
            category_name = nusc.get('category', ann['category_token'])['name']
            category_counts[category_name] += 1
            
            # 2. 카테고리별 BBox 넓이 계산 및 저장
            bbox = ann['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if area > 0:
                bbox_areas[category_name].append(area)

            # 3. 프레임(Sample)당 객체 수 집계
            sample_data = nusc.get('sample_data', ann['sample_data_token'])
            sample_token_to_obj_count[sample_data['sample_token']] += 1

    print("✅ 통계 집계 완료!")
    
    # --- 결과 리포트 출력 ---
    print("\n\n" + "="*50)
    print("      커스텀 데이터셋 심층 분석 리포트")
    print("="*50 + "\n")

    # 분석 1: 전체 객체 카테고리 분포
    print("--- 1. 전체 객체 카테고리 분포 ---")
    if not category_counts:
        print("분석할 객체 데이터가 없습니다.")
    else:
        cat_df = pd.DataFrame.from_dict(category_counts, orient='index', columns=['Count'])
        print(cat_df.sort_values(by='Count', ascending=False))
    
    # 분석 2: 카테고리별 평균 BBox 크기 분석
    print("\n--- 2. 카테고리별 평균 BBox 크기(Area) 분석 ---")
    if not bbox_areas:
        print("분석할 Bounding Box 데이터가 없습니다.")
    else:
        avg_areas = {cat: int(np.mean(areas)) for cat, areas in bbox_areas.items()}
        area_df = pd.DataFrame.from_dict(avg_areas, orient='index', columns=['Average Area (pixels)'])
        print(area_df.sort_values(by='Average Area (pixels)', ascending=False))
        # 그래프 생성 (선택사항)
        try:
            # ... (폰트 설정 및 그래프 생성 코드는 이전과 동일) ...
            pass
        except Exception:
            pass

    # 분석 3: 프레임당 객체 수 분석
    print("\n--- 3. 프레임(Sample)당 객체 수 분석 ---")
    if not sample_token_to_obj_count:
        print("분석할 프레임 데이터가 없습니다.")
    else:
        counts = list(sample_token_to_obj_count.values())
        print(f"  - 전체 프레임(Sample) 수: {len(nusc.sample)}")
        print(f"  - 객체가 검출된 프레임 수: {len(counts)}")
        print(f"  - 프레임당 평균 객체 수: {np.mean(counts):.2f} 개")
        print(f"  - 프레임당 최대 객체 수: {np.max(counts)} 개")
        print(f"  - 프레임당 최소 객체 수: {np.min(counts)} 개")
    
    print("\n" + "="*50)

# --- 스크립트 실행 ---
if __name__ == "__main__":
    analyze_real_data()