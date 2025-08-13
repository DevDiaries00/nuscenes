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
output_root = Path('/workspace/nuscenes_from_real_data')
#  한국교통안전공단 데이터 --> nuScenes 데이터
SENSOR_CHANNEL_MAPPING = {
    'Camera_Front_Center': 'CAM_FRONT',
}
#  한국교통안전공단 데이터 --> nuScenes 데이터
CUSTOM_TO_NUSCENES_CAT = {
    'car': 'vehicle.car',
    'truck': 'vehicle.truck',
    'bus': 'vehicle.bus',  # 'bus'에 대한 번역 규칙 추가
}

# 고유 토큰 할당
def generate_token(): return uuid.uuid4().hex

# 파일 경로 할당, 이미지 크기 가로, 세로 높이 측정
def get_image_dimensions(image_path):
    with Image.open(image_path) as img: return img.width, img.height

# 켈리브레이션 정보(더미 데이터)
def get_calibrated_sensor_data(channel_name):
    placeholder_calib = {
        'CAM_FRONT': {'translation': [1.5, 0.0, 1.4], 'rotation': [0.5, -0.5, 0.5, -0.5], 'camera_intrinsic': [[1200, 0, 800], [0, 1200, 450], [0, 0, 1]]}
    }
    #key에 해당하는 값을 딕셔너리에서 찾아서 반환, channel_name이 CAM_FRONT라면 그에 맞는 보정 정보를 반환
    # {} 오류 방지 --> 프로그램 멈추는 거 방지
    return placeholder_calib.get(channel_name, {})


# 차량 정보(더미 데이터)
def get_ego_pose_data(timestamp):
    time_since_start = (timestamp - 1686787200000000) / 1e6
    return {
        'translation': [time_since_start * 5.0, 0.0, 0.0], 'rotation': [1.0, 0.0, 0.0, 0.0],
        'speed': 5.0, 'acceleration': [0.0, 0.0, 0.0], 'rotation_rate': [0.0, 0.0, 0.0]
    }

def get_annotations_from_2dbb_file(img_path, category_token_map):
    annotations = []
    try:
        ann_path = img_path.parent / '2DBB' / (img_path.stem + '.json')
        if not ann_path.exists(): return []
    except Exception as e:
        print(f"경고: 어노테이션 경로 생성 중 오류 발생 {e}")
        return []

# json은 데이터를 교환하는 데 널리 쓰이는 데이터 형식
# json.load()  파일 객체 f로부터 JSON 형식의 텍스트를 읽어 들여, 파이썬이 이해할 수 있는 딕셔너리나 리스트로 변환(파싱)  
    with open(ann_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

# data.get은 data 딕셔너리에서 "Annotation"이라는 Key를 찾습니다. 만약 Key가 없으면, 오류 대신 빈 리스트 []를 반환하여 반복문이 문제없이 실행
    for ann in data.get("Annotation", []):
        custom_class_name, bbox_data = ann.get("class_name"), ann.get("data")
        #all([...]): 리스트 안의 모든 요소가 참(True)으로 평가될 때만 True를 반환
        if not all([custom_class_name, bbox_data, len(bbox_data) == 4]): continue

        # CUSTOM_TO_NUSCENES_CAT라는 (미리 정의된) 딕셔너리에서 custom_class_name에 해당하는 값을 찾아 nusc_cat_name 변수에 저장
        nusc_cat_name = CUSTOM_TO_NUSCENES_CAT.get(custom_class_name)
        if not nusc_cat_name or nusc_cat_name not in category_token_map: continue
            
        annotations.append({
            "token": generate_token(), "sample_data_token": "",
            "category_token": category_token_map[nusc_cat_name], "attribute_tokens": [],
            "bbox": [round(c) for c in bbox_data], "mask": None
        })
    return annotations

def convert_to_nuscenes():
    version = 'v1.0-real-data'
    json_folder, samples_folder, maps_folder = output_root/version, output_root/'samples', output_root/'maps'
    
    for folder in [json_folder, samples_folder, maps_folder]:
        #parents=True: 만약 상위 폴더(예: output_root)가 없으면 그것까지 함께 만듬
        #exist_ok=True: 폴더가 이미 존재하더라도 오류를 발생시키지 않고 넘어감
        folder.mkdir(parents=True, exist_ok=True)
    
    # nuScenes는 지도 파일이 필요하지만, 여기서는 실제 지도 데이터가 없으므로 가짜(dummy) 지도 파일을 만듬
    dummy_map_path = maps_folder / 'dummy_map.png'
    if not dummy_map_path.exists():
        print(f"'{dummy_map_path}'에 더미 맵 파일을 생성합니다.")
        Image.new('RGB', (1, 1), color='black').save(dummy_map_path)

    # 딕셔너리 컴프리헨션  k를 키로 새로운 빈 상자 한 쌍을 만든다 --> 딕셔너리 생성
    # 결과: {'log': [], 'scene': [], 'sample': [], 'sample_data': [], ...}
    nuscenes_tables = {k: [] for k in ['log', 'scene', 'sample', 'sample_data', 'calibrated_sensor', 'sensor', 'ego_pose', 'map', 'category', 'attribute', 'object_ann', 'surface_ann', 'visibility', 'instance', 'sample_annotation']}
    
    print("정적 테이블 정보 생성 중...")

    # catergory.json 파일
    category_token_map = {name: generate_token() for name in CUSTOM_TO_NUSCENES_CAT.values()}
    for name, token in category_token_map.items():
        nuscenes_tables['category'].append({"token": token, "name": name, "description": ""})
    
    # sensor.json 파일 
    sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    
    #.items() 메서드는 딕셔너리의 키(key)와 값(value)을 하나의 쌍(튜플)을 가져옴
    #.values() 메서드는 딕셔너리에서 값(value)들만 가져옴
    #zip() 함수는 두 개의 리스트(또는 이와 유사한 목록)를 지퍼처럼 하나로 합쳐줌
    #구조 분해 할당
    for (ch_folder, nu_ch), token in zip(SENSOR_CHANNEL_MAPPING.items(), sensor_tokens.values()):
        (samples_folder / nu_ch).mkdir(exist_ok=True)
        nuscenes_tables['sensor'].append({"token": token, "channel": nu_ch, "modality": "camera"})

    attribute_token_map = {}
    attr_names = ['cycle.with_rider', 'cycle.without_rider', 'vehicle.is_emergency']
    for attr_name in attr_names:
        attr_token = generate_token()
        attribute_token_map[attr_name] = attr_token
        nuscenes_tables['attribute'].append({"token": attr_token, "name": attr_name, "description": ""})

    for level in ['1', '2', '3', '4']:
        nuscenes_tables['visibility'].append({"token": level, "level": f'v{int(level)*20-20}-{int(level)*20}', "description": ""})

    log_name, log_token = data_root.name, generate_token()
    nuscenes_tables['log'].append({"token": log_token, "logfile": log_name, "vehicle": "custom_vehicle", "date_captured": "2023-06-15", "location": "Pangyo, South Korea"})
    
    scene_token = generate_token()
    nuscenes_tables['map'].append({"token": generate_token(), "log_tokens": [log_token], "category": "semantic_prior", "filename": str(dummy_map_path.relative_to(output_root))})

    calibrated_sensor_tokens = {nu_ch: generate_token() for nu_ch in SENSOR_CHANNEL_MAPPING.values()}
    for nu_ch, cs_token in calibrated_sensor_tokens.items():
        calib_data = get_calibrated_sensor_data(nu_ch)
        nuscenes_tables['calibrated_sensor'].append({'token': cs_token, 'sensor_token': sensor_tokens[nu_ch], **calib_data})

    print("이미지 파일 탐색 및 그룹핑 중...")
    frame_to_images = defaultdict(list)
    for channel_folder, nu_channel in SENSOR_CHANNEL_MAPPING.items():
        image_dir = data_root / 'Camera' / channel_folder
        if not image_dir.exists(): continue
        for img_path in sorted(image_dir.glob('*.jpg')):
            try: frame_to_images[int(img_path.stem.split('_')[-1])].append((nu_channel, img_path))
            except (ValueError, IndexError): continue
    
    if not frame_to_images:
        print("처리할 이미지를 찾지 못했습니다.")
        return

    scene_samples_tokens, prev_sample_token = [], ""
    prev_sd_tokens = {ch: "" for ch in SENSOR_CHANNEL_MAPPING.values()}
    for frame_idx in tqdm(sorted(frame_to_images.keys()), desc=f"'{log_name}' 씬 처리"):
        timestamp = 1686787200000000 + frame_idx * 100000
        sample_token, ego_pose_token = generate_token(), generate_token()
        nuscenes_tables['ego_pose'].append({'token': ego_pose_token, 'timestamp': timestamp, **get_ego_pose_data(timestamp)})
        
        sample_data_tokens = {}
        for nu_channel, img_path in frame_to_images[frame_idx]:
            width, height = get_image_dimensions(img_path)
            sd_token, output_img_path = generate_token(), samples_folder/nu_channel/img_path.name
            sample_data_tokens[nu_channel] = sd_token
            shutil.copy(img_path, output_img_path)
            
            sd_entry = {'token': sd_token, 'sample_token': sample_token, 'ego_pose_token': ego_pose_token, 'calibrated_sensor_token': calibrated_sensor_tokens[nu_channel], 'timestamp': timestamp, 'fileformat': 'jpg', 'is_key_frame': True, 'height': height, 'width': width, 'filename': str(output_img_path.relative_to(output_root)), 'prev': prev_sd_tokens[nu_channel], 'next': ""}
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
        prev_sample_token, scene_samples_tokens = sample_token, scene_samples_tokens + [sample_token]

    if scene_samples_tokens:
        nuscenes_tables['scene'].append({"token": scene_token, "log_token": log_token, "nbr_samples": len(scene_samples_tokens), "first_sample_token": scene_samples_tokens[0], "last_sample_token": scene_samples_tokens[-1], "name": f"scene-from-{log_name}", "description": f"Driving scene from {log_name}"})

    print(f"\n'{json_folder}'에 JSON 파일들을 저장합니다.")
    for table_name, data in nuscenes_tables.items():
        with open(json_folder / f'{table_name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
    print("\n실제 커스텀 데이터의 nuScenes 변환이 완료되었습니다!")


# --- 4. 스크립트 실행 ---
if __name__ == '__main__':
    convert_to_nuscenes()