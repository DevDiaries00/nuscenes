# [visualize_tracks_v3_fixed.py]: import tqdm 누락 문제를 수정한 최종 버전

import sys
import os
import cv2
import random
import numpy as np
import json
from collections import defaultdict
from PIL import Image, ImageDraw
from tqdm import tqdm  # <-- [핵심 수정] 누락되었던 tqdm 라이브러리를 import합니다.

# --- SDK 경로 설정 ---
sdk_path = '/workspace/nuscenes-devkit/python-sdk'
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)
from nuscenes.nuscenes import NuScenes

# --- 설정값 ---
DATAROOT = '/workspace/nuscenes_with_tracking'
VERSION = 'v1.0-tracking'
OUTPUT_DIR = os.path.join(DATAROOT, 'rendered_videos')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_scene_tracks(scene_idx=0):
    """지정된 장면의 객체 궤적을 비디오로 생성합니다."""
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=False)
    
    object_ann_path = os.path.join(nusc.dataroot, nusc.version, 'object_ann.json')
    with open(object_ann_path) as f:
        nusc.object_ann = json.load(f)
    
    sd_token_to_anns = defaultdict(list)
    for ann in nusc.object_ann:
        sd_token_to_anns[ann['sample_data_token']].append(ann)

    my_scene = nusc.scene[scene_idx]
    print(f"--- Scene '{my_scene['name']}'의 궤적 시각화 시작 ---")
    
    instance_tokens_in_scene = set()
    for sample in nusc.get_sample_content(my_scene['token'])[1]:
        for sd_token in sample['data'].values():
            for ann in sd_token_to_anns.get(sd_token, []):
                if 'instance_token' in ann:
                    instance_tokens_in_scene.add(ann['instance_token'])
                    
    instance_colors = {token: (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)) for token in instance_tokens_in_scene}
    trajectories = defaultdict(list)
    video_writer = None
    
    sample_token, samples = my_scene['first_sample_token'], nusc.get_sample_content(my_scene['token'])[1]
    # 이제 tqdm을 사용할 수 있습니다.
    for frame_count, sample in enumerate(tqdm(samples, desc="비디오 프레임 생성")):
        cam_token = sample['data']['CAM_FRONT']
        cam_data = nusc.get('sample_data', cam_token)
        img_path = os.path.join(nusc.dataroot, cam_data['filename'])
        
        pil_img = Image.open(img_path)
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        if video_writer is None:
            height, width, _ = cv_img.shape
            video_path = os.path.join(OUTPUT_DIR, f"scene_{my_scene['token'][:6]}_tracks.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 5, (width, height))

        for ann in sd_token_to_anns.get(cam_token, []):
            inst_token = ann.get('instance_token')
            if not inst_token:
                continue

            bbox = ann['bbox']
            color = instance_colors.get(inst_token, (0, 0, 255))
            cv2.rectangle(cv_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            center_x, center_y = int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)
            trajectories[inst_token].append((center_x, center_y))
            
            if len(trajectories[inst_token]) > 1:
                points = np.array(trajectories[inst_token], np.int32)
                cv2.polylines(cv_img, [points], isClosed=False, color=color, thickness=2)

        cv2.putText(cv_img, f"Frame: {frame_count + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        video_writer.write(cv_img)

    video_writer.release()
    print(f"\n✅ 비디오 저장 완료: {video_path}")


if __name__ == "__main__":
    # nusc.get_sample_content 와 같은 SDK 헬퍼 함수를 NuScenes 클래스에 추가
    def get_sample_content(self, scene_token: str):
        scene = self.get('scene', scene_token)
        sample_tok, samples = scene['first_sample_token'], []
        while sample_tok:
            sample = self.get('sample', sample_tok)
            samples.append(sample)
            sample_tok = sample['next']
        return scene, samples
    
    NuScenes.get_sample_content = get_sample_content
    
    visualize_scene_tracks(scene_idx=0)