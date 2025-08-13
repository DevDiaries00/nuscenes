import json
import os
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

# --- 1. 사용자 설정 영역 ---

# [개선] 기본 데이터 경로 (수정할 필요 없음)
base_data_path = Path('C:/Users/KATRI/Desktop/nuscenes-devkit/data/normal')
# [개선] 변환할 실제 데이터 폴더 이름 (이 부분만 바꿔서 사용하세요!)
# 예: 'data1/some_other_folder', 'data2/another_folder'
target_run_folder = 'data3/230615_001' 
# [개선] 변환된 파일이 저장될 최상위 폴더
output_root = Path('C:/Users/KATRI/Desktop/nuscenes-devkit/nuscenes_convert_data')

# --- 이하 코드는 자동으로 경로를 생성하므로 수정할 필요 없음 ---
data_root = base_data_path / target_run_folder
# GNSS/INS 파일 이름은 폴더 이름에서 가져오도록 자동화 (예: N_DFG_230615_001_GI.txt)
gnss_file_name = f"N_DFG_{data_root.name}_GI.txt"
gnss_ins_file = data_root / 'Meta' / 'GNSS_INS' / gnss_file_name


#  한국교통안전공단 데이터 --> nuScenes 데이터
SENSOR_CHANNEL_MAPPING = {
    'Camera_Front_Center': 'CAM_FRONT',
}
#  한국교통안전공단 데이터 --> nuScenes 데이터
CUSTOM_TO_NUSCENES_CAT = {
    'car': 'vehicle.car',
    'truck': 'vehicle.truck',
    'bus': 'vehicle.bus',
}

# --- 2. 함수 정의 ---

def generate_token():
    """고유 토큰을 생성합니다."""
    return uuid.uuid4().hex

def get_image_dimensions(image_path):
    """이미지 파일 경로를 받아 가로, 세로 크기를 반환합니다."""
    with Image.open(image_path) as img:
        return img.width, img.height

def get_calibrated_sensor_data(channel_name):
    """지정된 카메라 채널에 대한 더미 캘리브레이션 데이터를 반환합니다."""
    placeholder_calib = {
        'CAM_FRONT': {'translation': [1.5, 0.0, 1.4], 'rotation': [0.5, -0.5, 0.5, -0.5], 'camera_intrinsic': [[1200, 0, 800], [0, 1200, 450], [0, 0, 1]]}
    }
    return placeholder_calib.get(channel_name, {})

def load_gnss_data(file_path):
    """GNSS/INS 텍스트 파일을 읽어 pandas DataFrame으로 반환합니다."""
    if not file_path.exists():
        print(f"경고: GNSS/INS 파일이 존재하지 않습니다: {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, delim_whitespace=True, header=0)
        return df
    except Exception as e:
        print(f"오류: GNSS/INS 파일 로딩 중 오류 발생 {e}")
        return None

def get_annotations_from_2dbb_file(img_path, category_token_map):
    """2DBB JSON 파일에서 어노테이션 정보를 읽어 nuScenes 형식으로 변환합니다."""
    annotations = []
    try:
        # [개선] 어노테이션 파일 경로 구조를 보다 일반적으로 수정
        ann_path = img_path.parents[1] / '2DBB' / (img_path.stem + '.json')
        if not ann_path.exists():
            return []
    except Exception as e:
        print(f"경고: 어노테이션 경로 생성 중 오류 발생 {e}")
        return []

    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for ann in data.get("Annotation", []):
        custom_class_name, bbox_data = ann.get("class_name"), ann.get("data")
        if not all([custom_class_name, bbox_data, len(bbox_data) == 4]):
            continue

        nusc_cat_name = CUSTOM_TO_NUSCENES_CAT.get(custom_class_name)
        if not nusc_cat_name or nusc_cat_name not in category_token_map:
            continue
            
        annotations.append({
            "token": generate_token(), "sample_data_token": "",
            "category_token": category_token_map[nusc_cat_name], "attribute_tokens": [],
            "bbox": [round(c) for c in bbox_data], "mask": None
        })
    return annotations

def convert_to_nuscenes():
    """데이터를 nuScenes 포맷으로 변환하는 메인 함수입니다."""
    version = 'v1.0-real-data'
    json_folder = output_root / data_root.name / version
    samples_folder = output_root / data_root.name / 'samples'
    maps_folder = output_root / data_root.name / 'maps'
    
    for folder in [json_folder, samples_folder, maps_folder]:
        folder.mkdir(parents=True, exist_ok=True)
    
    dummy_map_path = maps_folder / 'dummy_map.png'
    if not dummy_map_path.exists():
        Image.new('RGB', (1, 1), color='black').save(dummy_map_path)

    print(f"'{gnss_ins_file}' 파일에서 ego_pose 데이터를 로드합니다.")
    gnss_df = load_gnss_data(gnss_ins_file)
    if gnss_df is None:
        print("ego_pose 데이터를 로드할 수 없어 변환을 중단합니다.")
        return
    gnss_records = gnss_df.to_dict('records')

    nuscenes_tables = {k: [] for k in ['log', 'scene', 'sample', 'sample_data', 'calibrated_sensor', 'sensor', 'ego_pose', 'map', 'category', 'attribute', 'object_ann', 'surface_ann', 'visibility', 'instance', 'sample_annotation']}
    
    print("정적 테이블 정보 생성 중...")

    category_token_map = {name: generate_token() for name in CUSTOM_TO_NUSCENES_CAT.values()}
    for name, token in category_token_map.items():
        nuscenes_tables['category'].append({"token": token, "name": name, "description": ""})
    
    sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    for (ch_folder, nu_ch), token in zip(SENSOR_CHANNEL_MAPPING.items(), sensor_tokens.values()):
        (samples_folder / nu_ch).mkdir(exist_ok=True)
        nuscenes_tables['sensor'].append({"token": token, "channel": nu_ch, "modality": "camera"})

    log_name, log_token = data_root.name, generate_token()
    nuscenes_tables['log'].append({"token": log_token, "logfile": log_name, "vehicle": "custom_vehicle", "date_captured": "2023-06-15", "location": "Pangyo, South Korea"})
    scene_token = generate_token()
    map_relative_path = Path(data_root.name) / 'maps' / 'dummy_map.png'
    nuscenes_tables['map'].append({"token": generate_token(), "log_tokens": [log_token], "category": "semantic_prior", "filename": str(map_relative_path)})

    calibrated_sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    for nu_ch, cs_token in calibrated_sensor_tokens.items():
        calib_data = get_calibrated_sensor_data(nu_ch)
        nuscenes_tables['calibrated_sensor'].append({'token': cs_token, 'sensor_token': sensor_tokens[nu_ch], **calib_data})

    print("이미지 파일 탐색 및 그룹핑 중...")
    frame_to_images = defaultdict(list)
    for channel_folder, nu_channel in SENSOR_CHANNEL_MAPPING.items():
        image_dir = data_root / 'Camera' / channel_folder
        if not image_dir.exists(): 
            print(f"경고: 이미지 디렉터리를 찾을 수 없습니다: {image_dir}")
            continue
        for img_path in sorted(image_dir.glob('*.jpg')):
            try: frame_to_images[int(img_path.stem.split('_')[-1])].append((nu_channel, img_path))
            except (ValueError, IndexError): continue
    
    if not frame_to_images:
        print("처리할 이미지를 찾지 못했습니다.")
        return

    scene_samples_tokens, prev_sample_token = [], ""
    prev_sd_tokens = {ch: "" for ch in SENSOR_CHANNEL_MAPPING.values()}
    
    sorted_frame_keys = sorted(frame_to_images.keys())
    for i, frame_idx in enumerate(tqdm(sorted_frame_keys, desc=f"'{log_name}' 씬 처리")):
        if i >= len(gnss_records):
            print(f"경고: 프레임 {frame_idx}에 해당하는 GNSS 데이터가 없습니다. 마지막 GNSS 데이터를 사용합니다.")
            gnss_record = gnss_records[-1]
        else:
            gnss_record = gnss_records[i]

        timestamp = 1686787200000000 + frame_idx * 100000
        sample_token, ego_pose_token = generate_token(), generate_token()

        translation = [gnss_record['Easting'], gnss_record['Northing'], gnss_record['Up']]
        roll_rad, pitch_rad, heading_rad = np.deg2rad(gnss_record['Roll']), np.deg2rad(gnss_record['Pitch']), np.deg2rad(gnss_record['Heading'])
        
        rotation = R.from_euler('xyz', [roll_rad, pitch_rad, heading_rad], degrees=False)
        quaternion = rotation.as_quat()
        rotation_nuscenes = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        
        nuscenes_tables['ego_pose'].append({'token': ego_pose_token, 'timestamp': timestamp, 'rotation': rotation_nuscenes, 'translation': translation})
        
        sample_data_tokens = {}
        for nu_channel, img_path in frame_to_images[frame_idx]:
            width, height = get_image_dimensions(img_path)
            sd_token = generate_token()
            output_img_path = samples_folder / nu_channel / img_path.name
            sample_data_tokens[nu_channel] = sd_token
            shutil.copy(img_path, output_img_path)
            
            # [개선] 저장되는 파일 경로를 output_root 기준으로 상대 경로화
            file_relative_path = Path(data_root.name) / 'samples' / nu_channel / img_path.name
            sd_entry = {'token': sd_token, 'sample_token': sample_token, 'ego_pose_token': ego_pose_token, 'calibrated_sensor_token': calibrated_sensor_tokens[nu_channel], 'timestamp': timestamp, 'fileformat': 'jpg', 'is_key_frame': True, 'height': height, 'width': width, 'filename': str(file_relative_path), 'prev': prev_sd_tokens[nu_channel], 'next': ""}
            nuscenes_tables['sample_data'].append(sd_entry)
            
            if prev_sd_tokens[nu_channel]:
                next(sd for sd in reversed(nuscenes_tables['sample_data']) if sd['token'] == prev_sd_tokens[nu_channel])['next'] = sd_token
            prev_sd_tokens[nu_channel] = sd_token

            real_annotations = get_annotations_from_2dbb_file(img_path, category_token_map)
            for ann in real_annotations:
                ann['sample_data_token'] = sd_token
            nuscenes_tables['object_ann'].extend(real_annotations)

        nuscenes_tables['sample'].append({'token': sample_token, 'timestamp': timestamp, 'scene_token': scene_token, 'prev': prev_sample_token, 'next': "", 'data': sample_data_tokens})
        if prev_sample_token:
            next(s for s in reversed(nuscenes_tables['sample']) if s['token'] == prev_sample_token)['next'] = sample_token
        prev_sample_token = sample_token
        scene_samples_tokens.append(sample_token)

    if scene_samples_tokens:
        nuscenes_tables['scene'].append({"token": scene_token, "log_token": log_token, "nbr_samples": len(scene_samples_tokens), "first_sample_token": scene_samples_tokens[0], "last_sample_token": scene_samples_tokens[-1], "name": f"scene-from-{log_name}", "description": f"Driving scene from {log_name}"})

    print(f"\n'{json_folder}'에 JSON 파일들을 저장합니다.")
    for table_name, data in nuscenes_tables.items():
        with open(json_folder / f'{table_name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    print(f"\n'{data_root.name}' 데이터의 nuScenes 변환이 완료되었습니다!")

# --- 4. 스크립트 실행 ---
if __name__ == '__main__':
    convert_to_nuscenes()