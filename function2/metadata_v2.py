# [코드_추적기능_완성본_v3_metadata.py]: object_ann 기반 추적 방식으로 변경하여 안정성 확보

import json
import os
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm


# --- 1. 사용자 설정 영역 ---

data_root = Path('/workspace/data/230615_001')
output_root = Path('/workspace/nuscenes_with_tracking')
SENSOR_CHANNEL_MAPPING = {'Camera_Front_Center': 'CAM_FRONT'}
CUSTOM_TO_NUSCENES_CAT = {'car': 'vehicle.car', 'truck': 'vehicle.truck', 'bus': 'vehicle.bus'}


# --- 2. 헬퍼 함수 ---

def generate_token():
    return uuid.uuid4().hex

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        return img.width, img.height

def get_calibrated_sensor_data(channel_name):
    placeholder_calib = {
        'CAM_FRONT': {
            'translation': [1.5, 0.0, 1.4],
            'rotation': [0.5, -0.5, 0.5, -0.5],
            'camera_intrinsic': [[1200, 0, 800], [0, 1200, 450], [0, 0, 1]]
        }
    }
    return placeholder_calib.get(channel_name, {})

def get_ego_pose_data(timestamp):
    time_since_start = (timestamp - 1686787200000000) / 1e6
    return {
        'translation': [time_since_start * 5.0, 0.0, 0.0],
        'rotation': [1.0, 0.0, 0.0, 0.0],
        'speed': 5.0,
        'acceleration': [0.0, 0.0, 0.0],
        'rotation_rate': [0.0, 0.0, 0.0]
    }

def get_annotations_from_2dbb_file(img_path, category_token_map):
    annotations, ann_token_to_instance_id = [], {}
    ann_path = img_path.parent / '2DBB' / (img_path.stem + '.json')
    if not ann_path.exists():
        return [], {}
    
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for ann in data.get("Annotation", []):
        custom_class_name = ann.get("class_name")
        bbox_data = ann.get("data")
        inst_id = ann.get("instance_id")
        
        if not all([custom_class_name, bbox_data, isinstance(bbox_data, list), len(bbox_data) == 4, inst_id is not None]):
            continue
        
        nusc_cat_name = CUSTOM_TO_NUSCENES_CAT.get(custom_class_name)
        if not nusc_cat_name:
            continue
            
        ann_token = generate_token()
        # [수정] object_ann 형식에 맞추고, 추적 필드 추가
        nusc_ann = {
            "token": ann_token,
            "sample_data_token": "",
            "category_token": category_token_map[nusc_cat_name],
            "attribute_tokens": [],
            "bbox": [round(c) for c in bbox_data],
            "mask": None,
            "instance_token": "",
            "prev": "",
            "next": ""  # 추적을 위한 커스텀 필드
        }
        annotations.append(nusc_ann)
        ann_token_to_instance_id[ann_token] = inst_id
        
    return annotations, ann_token_to_instance_id


# --- 3. 메인 변환 함수 ---

def convert_to_nuscenes():
    version = 'v1.0-tracking'
    json_folder = output_root / version
    samples_folder = output_root / 'samples'
    maps_folder = output_root / 'maps'
    
    for folder in [json_folder, samples_folder, maps_folder]:
        folder.mkdir(parents=True, exist_ok=True)

    dummy_map_path = maps_folder / 'dummy_map.png'
    if not dummy_map_path.exists():
        Image.new('RGB', (1, 1), color='black').save(dummy_map_path)

    nuscenes_tables = {k: [] for k in [
        'log', 'scene', 'sample', 'sample_data', 'calibrated_sensor', 'sensor',
        'ego_pose', 'map', 'category', 'attribute', 'visibility', 'instance',
        'object_ann', 'surface_ann', 'sample_annotation'
    ]}

    print("정적 테이블 정보 생성 중...")
    category_token_map = {name: generate_token() for name in CUSTOM_TO_NUSCENES_CAT.values()}
    for name, token in category_token_map.items():
        nuscenes_tables['category'].append({"token": token, "name": name, "description": ""})
    
    # (나머지 정적 테이블 생성은 이전과 동일)
    sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    for (ch_folder, nu_ch), token in zip(SENSOR_CHANNEL_MAPPING.items(), sensor_tokens.values()):
        (samples_folder / nu_ch).mkdir(exist_ok=True)
        nuscenes_tables['sensor'].append({"token": token, "channel": nu_ch, "modality": "camera"})
    
    for attr_name in ['cycle.with_rider', 'cycle.without_rider', 'vehicle.is_emergency']:
        nuscenes_tables['attribute'].append({"token": generate_token(), "name": attr_name, "description": ""})
    
    for level in ['1', '2', '3', '4']:
        nuscenes_tables['visibility'].append({"token": level, "level": f'v{int(level)*20-20}-{int(level)*20}', "description": ""})

    instance_id_to_token = {}
    instance_token_records = defaultdict(lambda: {'anns': [], 'cat_token': ''})
    
    log_name = data_root.name
    log_token = generate_token()
    nuscenes_tables['log'].append({
        "token": log_token,
        "logfile": log_name,
        "vehicle": "custom_vehicle",
        "date_captured": "2023-06-15",
        "location": "Pangyo, South Korea"
    })
    
    scene_token = generate_token()
    nuscenes_tables['map'].append({
        "token": generate_token(),
        "log_tokens": [log_token],
        "category": "semantic_prior",
        "filename": str(dummy_map_path.relative_to(output_root))
    })
    
    calibrated_sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    for nu_ch, cs_token in calibrated_sensor_tokens.items():
        nuscenes_tables['calibrated_sensor'].append({
            'token': cs_token,
            'sensor_token': sensor_tokens[nu_ch],
            **get_calibrated_sensor_data(nu_ch)
        })
    
    print("이미지 파일 탐색 및 그룹핑 중...")
    frame_to_images = defaultdict(list)
    for channel_folder, nu_channel in SENSOR_CHANNEL_MAPPING.items():
        image_dir = data_root / 'Camera' / channel_folder
        if not image_dir.exists():
            continue
        for img_path in sorted(image_dir.glob('*.jpg')):
            try:
                frame_to_images[int(img_path.stem.split('_')[-1])].append((nu_channel, img_path))
            except (ValueError, IndexError):
                continue
                
    if not frame_to_images:
        print("처리할 이미지를 찾지 못했습니다.")
        return

    scene_samples_tokens, prev_sample_token = [], ""
    prev_sd_tokens = {ch: "" for ch in SENSOR_CHANNEL_MAPPING.values()}
    
    for frame_idx in tqdm(sorted(frame_to_images.keys()), desc=f"'{log_name}' 씬 처리"):
        timestamp = 1686787200000000 + frame_idx * 100000
        sample_token, ego_pose_token = generate_token(), generate_token()
        nuscenes_tables['ego_pose'].append({
            'token': ego_pose_token,
            'timestamp': timestamp,
            **get_ego_pose_data(timestamp)
        })
        
        sample_data_tokens = {}
        for nu_channel, img_path in frame_to_images[frame_idx]:
            width, height = get_image_dimensions(img_path)
            sd_token = generate_token()
            output_img_path = samples_folder / nu_channel / img_path.name
            shutil.copy(img_path, output_img_path)
            
            sd_entry = {
                'token': sd_token,
                'sample_token': sample_token,
                'ego_pose_token': ego_pose_token,
                'calibrated_sensor_token': calibrated_sensor_tokens[nu_channel],
                'timestamp': timestamp,
                'fileformat': 'jpg',
                'is_key_frame': True,
                'height': height,
                'width': width,
                'filename': str(output_img_path.relative_to(output_root)),
                'prev': prev_sd_tokens[nu_channel],
                'next': ""
            }
            nuscenes_tables['sample_data'].append(sd_entry)
            
            if prev_sd_tokens[nu_channel]:
                next(sd for sd in reversed(nuscenes_tables['sample_data']) if sd['token'] == prev_sd_tokens[nu_channel])['next'] = sd_token
            
            prev_sd_tokens[nu_channel] = sd_token
            sample_data_tokens[nu_channel] = sd_token

            real_annotations, ann_token_to_id = get_annotations_from_2dbb_file(img_path, category_token_map)
            
            for ann in real_annotations:
                ann['sample_data_token'] = sd_token
                original_inst_id = ann_token_to_id[ann['token']]

                if original_inst_id not in instance_id_to_token:
                    instance_id_to_token[original_inst_id] = generate_token()
                
                inst_token = instance_id_to_token[original_inst_id]
                ann['instance_token'] = inst_token
                
                records = instance_token_records[inst_token]
                if records['anns']:
                    last_ann_token = records['anns'][-1]
                    ann['prev'] = last_ann_token
                    next(a for a in reversed(nuscenes_tables['object_ann']) if a['token'] == last_ann_token)['next'] = ann['token']

                records['anns'].append(ann['token'])
                records['cat_token'] = ann['category_token']
                
            nuscenes_tables['object_ann'].extend(real_annotations)  # sample_annotation 대신 object_ann 사용

        nuscenes_tables['sample'].append({
            'token': sample_token,
            'timestamp': timestamp,
            'scene_token': scene_token,
            'prev': prev_sample_token,
            'next': "",
            'data': sample_data_tokens
        })
        
        if prev_sample_token:
            next(s for s in reversed(nuscenes_tables['sample']) if s['token'] == prev_sample_token)['next'] = sample_token
        
        prev_sample_token = sample_token
        scene_samples_tokens.append(sample_token)

    if scene_samples_tokens:
        # instance 테이블 생성
        for inst_token, data in instance_token_records.items():
            nuscenes_tables['instance'].append({
                "token": inst_token,
                "category_token": data['cat_token'],
                "nbr_annotations": len(data['anns']),
                "first_annotation_token": data['anns'][0],
                "last_annotation_token": data['anns'][-1]
            })
        
        # scene 테이블 생성
        nuscenes_tables['scene'].append({
            "token": scene_token,
            "log_token": log_token,
            "nbr_samples": len(scene_samples_tokens),
            "first_sample_token": scene_samples_tokens[0],
            "last_sample_token": scene_samples_tokens[-1],
            "name": f"scene-from-{log_name}",
            "description": f"Driving scene from {log_name}"
        })

    print(f"\n'{json_folder}'에 JSON 파일들을 저장합니다.")
    for table_name, data in nuscenes_tables.items():
        with open(json_folder / f'{table_name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    print("\n객체 추적 정보가 포함된 데이터셋 변환이 완료되었습니다!")

if __name__ == '__main__':
    convert_to_nuscenes()