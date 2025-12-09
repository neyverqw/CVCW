#!/usr/bin/env python3
"""
示例：如何加载和使用标定结果
"""

import numpy as np
import json
import cv2

def load_calibration_numpy(calibration_file):
    """从NumPy格式文件加载标定数据（推荐）"""
    data = np.load(calibration_file)
    return {
        'camera_matrix': data['camera_matrix'],
        'dist_coeffs': data['dist_coeffs'],
        'focal_length': data['focal_length']
    }

def load_calibration_json(calibration_file):
    """从JSON格式文件加载标定数据"""
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)
    return {
        'camera_matrix': np.array(calib_data['camera_matrix']),
        'dist_coeffs': np.array(calib_data['distortion_coefficients']),
        'focal_length': calib_data['focal_length']
    }

def undistort_image(image, camera_matrix, dist_coeffs):
    """使用标定结果校正图像畸变"""
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)
    return undistorted, roi

# 示例使用
if __name__ == "__main__":
    # 加载第一个相机的标定数据
    try:
        calib1 = load_calibration_numpy('calibration_results/Camera1_calibration.npz')
        print("相机1标定数据:")
        print(f"  焦距: {calib1['focal_length']:.2f} 像素")
        print(f"  相机矩阵:\n{calib1['camera_matrix']}")
        print(f"  畸变系数: {calib1['dist_coeffs'].flatten()}")
    except FileNotFoundError:
        print("未找到相机1的标定文件")
    
    # 加载第二个相机的标定数据
    try:
        calib2 = load_calibration_numpy('calibration_results/Camera2_calibration.npz')
        print("\n相机2标定数据:")
        print(f"  焦距: {calib2['focal_length']:.2f} 像素")
        print(f"  相机矩阵:\n{calib2['camera_matrix']}")
        print(f"  畸变系数: {calib2['dist_coeffs'].flatten()}")
        
        # 比较焦距
        focal_diff = abs(calib1['focal_length'] - calib2['focal_length'])
        focal_diff_percent = (focal_diff / min(calib1['focal_length'], calib2['focal_length'])) * 100
        print(f"\n焦距差异: {focal_diff:.2f} 像素 ({focal_diff_percent:.2f}%)")
        
        if focal_diff_percent < 1.0:
            print("✓ 焦距非常接近，适合立体视觉")
        elif focal_diff_percent < 5.0:
            print("⚠ 焦距有轻微差异，可以使用")
        else:
            print("✗ 焦距差异较大，建议调整")
            
    except FileNotFoundError:
        print("未找到相机2的标定文件")
    
    # 示例：使用标定结果校正图像
    print("\n" + "="*60)
    print("示例：校正图像畸变")
    print("="*60)
    print("""
    # 打开相机
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    # 加载标定数据
    calib = load_calibration_numpy('calibration_results/Camera1_calibration.npz')
    
    # 校正畸变
    undistorted, roi = undistort_image(
        frame,
        calib['camera_matrix'],
        calib['dist_coeffs']
    )
    
    # 显示结果
    cv2.imshow('Original', frame)
    cv2.imshow('Undistorted', undistorted)
    cv2.waitKey(0)
    """)

