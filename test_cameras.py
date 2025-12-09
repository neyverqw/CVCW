#!/usr/bin/env python3
"""
快速测试相机是否可用
用于确定相机的ID
"""

import cv2
import sys

def test_camera(camera_id):
    """测试相机是否可用"""
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        return False, None
    
    # 尝试读取一帧
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        return True, frame.shape
    else:
        return False, None

if __name__ == "__main__":
    print("="*60)
    print("相机检测工具")
    print("="*60)
    print("\n正在检测可用的相机...\n")
    
    available_cameras = []
    
    # 检测前10个相机ID
    for i in range(10):
        print(f"检测相机 ID {i}...", end=" ")
        success, shape = test_camera(i)
        
        if success:
            print(f"✓ 可用 (分辨率: {shape[1]}x{shape[0]})")
            available_cameras.append(i)
        else:
            print("✗ 不可用")
    
    print("\n" + "="*60)
    if available_cameras:
        print(f"找到 {len(available_cameras)} 个可用相机:")
        for cam_id in available_cameras:
            print(f"  - 相机 ID: {cam_id}")
        print("\n可以使用以下命令进行标定:")
        if len(available_cameras) >= 2:
            print(f"  python camera_calibration.py --camera1 {available_cameras[0]} --camera2 {available_cameras[1]}")
        else:
            print(f"  python camera_calibration.py --camera1 {available_cameras[0]}")
    else:
        print("未找到可用相机")
        print("\n请检查:")
        print("  1. 相机是否正确连接")
        print("  2. 相机驱动是否已安装")
        print("  3. 相机是否被其他程序占用")
    print("="*60)

