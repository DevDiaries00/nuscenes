# visualize_converted_data.py (Explorer 미사용 버전)

import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

sdk_path = 'C:/Users/KATRI/Desktop/nuscenes-devkit/python-sdk'
sys.path.append(sdk_path)

# NuScenes 기본 클래스만 임포트합니다.
from nuscenes.nuscenes import NuScenes

# --- 1. 설정 ---
CONFIG = {
    "dataroot": 'C:/Users/KATRI/Desktop/nuscenes-devkit/nuscenes_format_final',
    "run_name": '230615_001',
}


# --- [새로 추가된 함수] BEV 렌더링 직접 구현 ---
def render_bev_manual(nusc: NuScenes, sample_token: str, bev_range: tuple = (-50, 50, -50, 50)):
    """
    NuScenesExplorer 없이 Bird's-Eye-View를 직접 렌더링합니다.
    :param nusc: NuScenes 객체
    :param sample_token: 시각화할 샘플의 토큰
    :param bev_range: (x_min, x_max, y_min, y_max) 렌더링 범위
    """
    # 1. 데이터 가져오기
    my_sample = nusc.get('sample', sample_token)
    
    # ego_pose 정보 가져오기 (LIDAR_TOP 기준)
    sd_token = my_sample['data']['LIDAR_TOP']
    sd_record = nusc.get('sample_data', sd_token)
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])
    
    # 2. Matplotlib figure 준비
    fig, ax = plt.subplots(figsize=(8, 8))

    # 3. 주석(Annotation) 처리
    for ann_token in my_sample['anns']:
        ann_record = nusc.get('sample_annotation', ann_token)
        
        # 전역 좌표계에서 ego_pose를 기준으로 로컬 좌표계로 변환
        box = nusc.get_box(ann_token) # Box 객체 사용
        box.translate(-np.array(pose_record['translation']))
        box.rotate(R.from_quat(pose_record['rotation']).inv())
        
        # 2D 평면(x, y)에 박스의 코너를 그립니다.
        corners = box.corners()
        bev_corners = corners[[0, 1, 3, 2, 0], [0, 1, 1, 0, 0]] # BEV의 4개 코너 선택
        ax.plot(bev_corners[:, 0], bev_corners[:, 1], color='pink', linewidth=2)

    # 4. 자기 차량(Ego Vehicle) 그리기
    # 자기 차량은 항상 로컬 좌표계의 원점(0,0)에 위치합니다.
    ego_width, ego_length = 1.8, 4.5 # 차량 크기 (미터)
    half_w, half_l = ego_width / 2, ego_length / 2
    ego_box = np.array([
        [half_l, -half_w], [half_l, half_w], [-half_l, half_w],
        [-half_l, -half_w], [half_l, -half_w]
    ])
    ax.plot(ego_box[:, 0], ego_box[:, 1], 'r-', linewidth=2, label='Ego Vehicle')

    # 5. 플롯 설정
    ax.set_xlim(bev_range[0], bev_range[1])
    ax.set_ylim(bev_range[2], bev_range[3])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Top-down Bird's-Eye-View (Manual Render)\nSample: {sample_token[:6]}")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True)
    ax.legend()


# --- 메인 시각화 함수 ---
def main():
    version = f'v1.0-converted-{CONFIG["run_name"]}'
    dataroot_path = Path(CONFIG["dataroot"]) / CONFIG["run_name"]

    print(f"🚀 nuScenes 데이터를 로드합니다...")
    nusc = NuScenes(version=version, dataroot=str(dataroot_path), verbose=False)
    
    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    
    # --- 시각화 실행 ---
    # [1] 단일 카메라 렌더링
    print("\n--- [1] 전방 카메라(CAM_FRONT) 샘플 데이터 렌더링 ---")
    nusc.render_sample_data(nusc.get('sample', first_sample_token)['data']['CAM_FRONT'], with_anns=True)
    plt.suptitle("CAM_FRONT with 3D Annotations")
    plt.show()

    # [2] 전체 카메라 렌더링
    print("\n--- [2] 전체 샘플(모든 카메라) 렌더링 ---")
    nusc.render_sample(first_sample_token)
    plt.show()

    # [3] 직접 구현한 BEV 렌더링 함수 호출
    print("\n--- [3] 직접 구현한 탑다운 뷰(BEV) 렌더링 ---")
    render_bev_manual(nusc, first_sample_token)
    plt.show()
    
    print("\n🎉 모든 시각화가 완료되었습니다.")


if __name__ == '__main__':
    main()