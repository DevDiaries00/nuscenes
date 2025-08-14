import json
import uuid
import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
from datetime import datetime, timezone

# --- 1. 헬퍼 함수 정의 (변경 없음) ---
def generate_token():
    return uuid.uuid4().hex

def get_image_dimensions(image_path):
    try:
        with Image.open(image_path) as img: return img.width, img.height
    except FileNotFoundError: return 0, 0

def load_ego_poses(gnss_file_path, scene_date_str):
    if not gnss_file_path.exists(): return None
    df = pd.read_csv(gnss_file_path, sep=r'\s+', engine='python')
    scene_date = datetime.strptime(scene_date_str, '%y%m%d')
    ego_poses = []
    for _, row in df.iterrows():
        time_of_day = row['TIME']
        full_dt = datetime.fromtimestamp(scene_date.timestamp() + time_of_day, tz=timezone.utc)
        timestamp = int(full_dt.timestamp() * 1_000_000)
        translation = [row['Easting'], row['Northing'], row['Up']]
        rotation = R.from_euler('zyx', [row['Heading'], row['Pitch'], row['Roll']], degrees=True).as_quat()
        rotation_wxyz = [rotation[3], rotation[0], rotation[1], rotation[2]]
        ego_poses.append({"token": generate_token(), "timestamp": timestamp, "rotation": rotation_wxyz, "translation": translation})
    return sorted(ego_poses, key=lambda p: p['timestamp'])

def load_sensors_and_calibrations(calib_file_path, channel_mapping):
    if not calib_file_path.exists(): return None, None
    try:
        with open(calib_file_path, 'r', encoding='utf-8') as f: custom_data = json.load(f)
    except json.JSONDecodeError: return None, None
    sensor_records, calibrated_sensor_records = [], []
    for _, sensor_data in custom_data.items():
        if not isinstance(sensor_data, dict): continue
        sensor_name = sensor_data.get("0_name")
        if not sensor_name or sensor_name not in channel_mapping: continue
        nusc_channel = channel_mapping[sensor_name]
        modality = 'camera' if 'CAM' in nusc_channel else 'lidar'
        extrinsic = sensor_data.get("4_extrinsic") or sensor_data.get("3_extrinsic", {})
        sensor_token, cs_token = generate_token(), generate_token()
        sensor_records.append({"token": sensor_token, "channel": nusc_channel, "modality": modality})
        calibrated_sensor = {"token": cs_token, "sensor_token": sensor_token, "translation": [extrinsic.get('tx', 0.0), extrinsic.get('ty', 0.0), extrinsic.get('tz', 0.0)], "rotation": [extrinsic.get('w', 1.0), extrinsic.get('x', 0.0), extrinsic.get('y', 0.0), extrinsic.get('z', 0.0)], "camera_intrinsic": []}
        if '3_intrinsic' in sensor_data:
            intrinsic = sensor_data['3_intrinsic']
            calibrated_sensor["camera_intrinsic"] = [[intrinsic.get('fx', 0.0), 0, intrinsic.get('cx', 0.0)], [0, intrinsic.get('fy', 0.0), intrinsic.get('cy', 0.0)], [0, 0, 1]]
        calibrated_sensor_records.append(calibrated_sensor)
    return sensor_records, calibrated_sensor_records

def get_box_params_from_points(points):
    p0, p1, p2, p3 = np.array(points[0]), np.array(points[1]), np.array(points[2]), np.array(points[3])
    v_width, v_length, v_height = p1 - p0, p2 - p0, p3 - p0
    size = [np.linalg.norm(v_width), np.linalg.norm(v_length), np.linalg.norm(v_height)]
    center = p0 + 0.5 * (v_width + v_length + v_height)
    x_axis, y_axis = v_length / size[1], v_width / size[0]
    z_axis = np.cross(x_axis, y_axis)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    quat_xyzw = R.from_matrix(rotation_matrix).as_quat()
    rotation_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return center.tolist(), size, rotation_wxyz

def load_annotations(annotation_dir):
    if not annotation_dir.is_dir(): return defaultdict(list)
    annotations_by_frame = defaultdict(list)
    for ann_file in sorted(annotation_dir.glob("*.json")):
        try:
            frame_idx = int(ann_file.stem.split('_')[-1])
            with open(ann_file, 'r', encoding='utf-8') as f: data = json.load(f)
            for ann in data.get("Annotation", []):
                if not all(k in ann for k in ["instance_id", "class_name", "data"]): continue
                center, size, rotation = get_box_params_from_points(ann["data"])
                annotations_by_frame[frame_idx].append({"instance_id": ann.get("instance_id"), "category": ann.get("class_name"), "translation": center, "size": size, "rotation": rotation})
        except Exception: continue
    return annotations_by_frame

# --- 2. 메인 변환 함수 ---

def convert_scene_to_nuscenes(run_name, base_path, output_path, sensor_config, cat_mapping):
    """단일 씬(run)에 대한 모든 필수 nuScenes JSON 파일을 생성합니다."""
    version = f'v1.0-converted-{run_name}'
    json_folder = output_path / run_name / version
    samples_folder = output_path / run_name / 'samples'
    json_folder.mkdir(parents=True, exist_ok=True)
    print(f"--- '{run_name}' 데이터셋 변환 시작 ---")

    channel_mapping = {s['json_name']: s['nusc_name'] for s in sensor_config}

    # 1. 필수 데이터 로드
    print("\n[단계 1/6] 모든 필수 데이터를 로드합니다...")
    scene_date_str = run_name.split('_')[0]
    gnss_path = base_path / 'data3' / run_name / 'Meta' / 'GNSS_INS' / f"N_DFG_{run_name}_GI.txt"
    calib_path = base_path / 'data1' / run_name / 'Meta' / 'Calib' / f"N_DFG_{run_name}_CA.json"
    ann_path = base_path / 'data2' / run_name / 'Lidar' / 'Lidar_Roof' / '3DBB'

    ego_pose_records = load_ego_poses(gnss_path, scene_date_str)
    sensor_records, cs_records = load_sensors_and_calibrations(calib_path, channel_mapping)
    annotations_by_frame = load_annotations(ann_path)
    
    if not ego_pose_records or not sensor_records: return
    print(f"  -> 데이터 로드 완료. (어노테이션 프레임: {len(annotations_by_frame)}개)")

    cs_token_map = {rec['channel']: cs['token'] for rec, cs in zip(sensor_records, cs_records)}
    lidar_calib = next((cs for sr, cs in zip(sensor_records, cs_records) if sr['channel'] == 'LIDAR_TOP'), None)

    # 2. 정적 테이블 생성
    print("[단계 2/6] 정적 테이블(log, map, category 등)을 생성합니다...")
    nuscenes_tables = defaultdict(list)
    log_token = generate_token()
    nuscenes_tables['log'].append({"token": log_token, "logfile": run_name, "vehicle": "custom-vehicle", "date_captured": scene_date_str, "location": "Hwaseong-si"})
    
    # [추가됨] map.json 더미 데이터 생성
    nuscenes_tables['map'].append({"token": generate_token(), "log_tokens": [log_token], "category": "semantic_prior", "filename": ""})

    category_token_map = {name: generate_token() for name in cat_mapping.values()}
    for name, token in category_token_map.items():
        nuscenes_tables['category'].append({"token": token, "name": name, "description": ""})

    # [추가됨] attribute.json 더미 데이터 생성
    nuscenes_tables['attribute'].append({"token": generate_token(), "name": "vehicle.moving", "description": "Vehicle is moving."})
    nuscenes_tables['attribute'].append({"token": generate_token(), "name": "vehicle.parked", "description": "Vehicle is parked."})

    # 3. 센서 파일 탐색 및 그룹핑
    print("[단계 3/6] 센서 파일(.jpg, .pcd)을 탐색하고 그룹화합니다...")
    keyframe_data = defaultdict(list)
    for sensor in sensor_config:
        nusc_ch, folder_name = sensor['nusc_name'], sensor['folder_name']
        modality = 'Camera' if 'CAM' in nusc_ch else 'Lidar'
        sensor_dir = base_path / 'data1' / run_name / modality / folder_name
        if sensor_dir.is_dir():
            files = sorted(sensor_dir.glob('*.jpg' if modality == 'Camera' else '*.pcd'))
            for f in files:
                try:
                    frame_idx = int(f.stem.split('_')[-1])
                    keyframe_data[frame_idx].append({'channel': nusc_ch, 'path': f})
                except (ValueError, IndexError): continue
    print(f"  -> {len(keyframe_data)}개의 키프레임을 발견했습니다.")
    if len(keyframe_data) == 0:
        print("오류: 센서 파일을 찾지 못했습니다. SENSOR_CONFIG의 'folder_name'이 실제 폴더명과 일치하는지 확인하세요.")
        return

    # 4. 동적 테이블 생성
    print("[단계 4/6] 동적 테이블을 생성합니다...")
    scene_token, prev_sample_token = generate_token(), ""
    prev_sd_tokens = {s['nusc_name']: "" for s in sensor_config}
    instance_token_map, prev_ann_map = {}, {}
    
    num_frames = min(len(keyframe_data), len(ego_pose_records))
    sorted_frame_indices = sorted(keyframe_data.keys())

    for i in tqdm(range(num_frames), desc="  - 키프레임 및 어노테이션 처리 중"):
        frame_idx, ego_pose, sample_token = sorted_frame_indices[i], ego_pose_records[i], generate_token()
        
        for sensor_file in keyframe_data[frame_idx]:
            nusc_ch, file_path = sensor_file['channel'], sensor_file['path']
            if nusc_ch not in cs_token_map: continue
            (samples_folder / nusc_ch).mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, samples_folder / nusc_ch / file_path.name)
            relative_path = str(Path('samples') / nusc_ch / file_path.name).replace('\\', '/')
            width, height = get_image_dimensions(file_path) if 'CAM' in nusc_ch else (0, 0)
            sd_token = generate_token()
            nuscenes_tables['sample_data'].append({'token': sd_token, 'sample_token': sample_token, 'ego_pose_token': ego_pose['token'], 'calibrated_sensor_token': cs_token_map[nusc_ch], 'timestamp': ego_pose['timestamp'], 'fileformat': file_path.suffix[1:], 'is_key_frame': True, 'height': height, 'width': width, 'filename': relative_path, 'prev': prev_sd_tokens.get(nusc_ch, ""), 'next': ""})
            if prev_sd_tokens.get(nusc_ch):
                next(sd for sd in reversed(nuscenes_tables['sample_data']) if sd['token'] == prev_sd_tokens[nusc_ch])['next'] = sd_token
            prev_sd_tokens[nusc_ch] = sd_token

        nuscenes_tables['sample'].append({'token': sample_token, 'timestamp': ego_pose['timestamp'], 'scene_token': scene_token, 'prev': prev_sample_token, 'next': ""})
        if prev_sample_token:
            next(s for s in reversed(nuscenes_tables['sample']) if s['token'] == prev_sample_token)['next'] = sample_token
        prev_sample_token = sample_token
        
        if frame_idx in annotations_by_frame and lidar_calib:
            for ann in annotations_by_frame[frame_idx]:
                instance_id, custom_cat = ann.get("instance_id"), ann.get("category")
                if not all([instance_id is not None, custom_cat, ann.get("translation")]): continue
                if instance_id not in instance_token_map:
                    instance_token = generate_token()
                    instance_token_map[instance_id] = instance_token
                    nuscenes_tables['instance'].append({"token": instance_token, "category_token": category_token_map.get(cat_mapping.get(custom_cat)), "nbr_annotations": 0, "first_annotation_token": "", "last_annotation_token": ""})
                instance_token = instance_token_map[instance_id]
                
                box_in_ego_vehicle_rot = R.from_quat(lidar_calib['rotation'][1:] + [lidar_calib['rotation'][0]]) * R.from_quat(ann['rotation'][1:] + [ann['rotation'][0]])
                global_rot = R.from_quat(ego_pose['rotation'][1:] + [ego_pose['rotation'][0]]) * box_in_ego_vehicle_rot
                trans_in_ego = R.from_quat(lidar_calib['rotation'][1:] + [lidar_calib['rotation'][0]]).apply(ann['translation']) + lidar_calib['translation']
                global_trans = R.from_quat(ego_pose['rotation'][1:] + [ego_pose['rotation'][0]]).apply(trans_in_ego) + ego_pose['translation']
                
                ann_token = generate_token()
                # [수정됨] attribute_tokens 필드 추가
                nuscenes_tables['sample_annotation'].append({"token": ann_token, "sample_token": sample_token, "instance_token": instance_token, "category_token": category_token_map.get(cat_mapping.get(custom_cat)), "translation": global_trans.tolist(), "size": ann.get("size"), "rotation": [global_rot.as_quat()[3]] + global_rot.as_quat()[:3].tolist(), "attribute_tokens": [], "num_lidar_pts": 0, "num_radar_pts": 0, "visibility_token": "4", "prev": prev_ann_map.get(instance_id, ""), "next": ""})
                
                inst_rec = next(inst for inst in nuscenes_tables['instance'] if inst['token'] == instance_token)
                inst_rec['nbr_annotations'] += 1
                if not inst_rec['first_annotation_token']: inst_rec['first_annotation_token'] = ann_token
                if prev_ann_map.get(instance_id):
                    next(sa for sa in reversed(nuscenes_tables['sample_annotation']) if sa['token'] == prev_ann_map[instance_id])['next'] = ann_token
                inst_rec['last_annotation_token'] = ann_token
                prev_ann_map[instance_id] = ann_token
    
    print("[단계 5/6] Scene 정보를 업데이트합니다...")
    if nuscenes_tables['sample']:
        nuscenes_tables['scene'].append({"token": scene_token, "log_token": log_token, "nbr_samples": len(nuscenes_tables['sample']), "first_sample_token": nuscenes_tables['sample'][0]['token'], "last_sample_token": nuscenes_tables['sample'][-1]['token'], "name": f"scene-{run_name}", "description": f"Converted from {run_name}"})
    
    print("[단계 6/6] 모든 JSON 파일을 저장합니다...")
    nuscenes_tables['sensor'] = sensor_records
    nuscenes_tables['calibrated_sensor'] = cs_records
    nuscenes_tables['ego_pose'] = ego_pose_records
    nuscenes_tables['visibility'] = [{"token": "4", "level": "unknown", "description": ""}]
    
    for table_name, data in nuscenes_tables.items():
        if data:
            filepath = json_folder / f'{table_name}.json'
            with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=2)
            print(f"  -> '{filepath}' 저장 완료.")
    print(f"\n🎉 모든 변환 작업이 성공적으로 완료되었습니다!")

# --- 3. 스크립트 실행 ---
if __name__ == '__main__':
    BASE_PATH = Path('C:/Users/KATRI/Desktop/nuscenes-devkit/data/normal')
    OUTPUT_PATH = Path('C:/Users/KATRI/Desktop/nuscenes-devkit/nuscenes_format_final')
    TARGET_RUN = '230615_001'
    
    SENSOR_CONFIG = [
        {'json_name': '01_camera', 'folder_name': 'Camera_Front_Center', 'nusc_name': 'CAM_FRONT'},
        {'json_name': '05_camera', 'folder_name': 'Camera_Front_Left', 'nusc_name': 'CAM_FRONT_LEFT'},
        {'json_name': '02_camera', 'folder_name': 'Camera_Front_Right', 'nusc_name': 'CAM_FRONT_RIGHT'},
        {'json_name': '04_camera', 'folder_name': 'Camera_Rear', 'nusc_name': 'CAM_REAR'},
        {'json_name': '03_camera', 'folder_name': 'Camera_Rear_Right', 'nusc_name': 'CAM_REAR_RIGHT'},
        {'json_name': '01_lidar', 'folder_name': 'Lidar_Roof', 'nusc_name': 'LIDAR_TOP'},
    ]
    
    CUSTOM_TO_NUSCENES_CAT = {
        'car': 'vehicle.car', 'truck': 'vehicle.truck', 'bus': 'vehicle.bus',
        'pedestrian': 'human.pedestrian.adult', 'bicycle': 'vehicle.bicycle',
        'motorbike': 'vehicle.motorcycle',
    }
    
    convert_scene_to_nuscenes(TARGET_RUN, BASE_PATH, OUTPUT_PATH, SENSOR_CONFIG, CUSTOM_TO_NUSCENES_CAT)