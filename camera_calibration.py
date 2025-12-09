#!/usr/bin/env python3
"""
双相机棋盘格标定工具
实时显示相机画面和焦距，确保两个相机焦距一致
"""

import cv2
import numpy as np
import os
import json
from datetime import datetime
import argparse

class CameraCalibrator:
    def __init__(self, camera_id, camera_name, chessboard_size=(9, 6), square_size=25.0):
        """
        初始化相机标定器
        
        参数:
            camera_id: 相机ID (0, 1, 2, ...)
            camera_name: 相机名称 (用于保存文件)
            chessboard_size: 棋盘格内角点数量 (width, height)
            square_size: 棋盘格方格大小 (毫米)
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        
        # 标定数据
        self.obj_points = []  # 3D点
        self.img_points = []  # 2D点
        self.calibration_images = []
        
        # 标定结果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.focal_length = None
        self.calibrated = False
        
        # 准备3D对象点
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
    def detect_chessboard(self, img):
        """检测棋盘格角点"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        return ret, corners, gray
    
    def calibrate(self):
        """执行标定"""
        if len(self.img_points) < 10:
            print(f"警告: {self.camera_name} 标定图像数量不足 (需要至少10张，当前{len(self.img_points)}张)")
            return False
        
        print(f"\n开始标定 {self.camera_name}...")
        print(f"使用 {len(self.img_points)} 张标定图像")
        
        # 执行标定
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            (self.calibration_images[0].shape[1], self.calibration_images[0].shape[0]),
            None,
            None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.focal_length = (camera_matrix[0, 0] + camera_matrix[1, 1]) / 2.0
            self.calibrated = True
            
            # 计算重投影误差
            total_error = 0
            for i in range(len(self.obj_points)):
                imgpoints2, _ = cv2.projectPoints(
                    self.obj_points[i], rvecs[i], tvecs[i],
                    camera_matrix, dist_coeffs
                )
                error = cv2.norm(self.img_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error
            
            mean_error = total_error / len(self.obj_points)
            
            print(f"✓ {self.camera_name} 标定完成!")
            print(f"  焦距: {self.focal_length:.2f} 像素")
            print(f"  重投影误差: {mean_error:.4f} 像素")
            print(f"  相机矩阵:\n{camera_matrix}")
            print(f"  畸变系数: {dist_coeffs.flatten()}")
            
            return True
        else:
            print(f"✗ {self.camera_name} 标定失败")
            return False
    
    def save_calibration(self, output_dir="calibration_results"):
        """保存标定结果"""
        if not self.calibrated:
            print(f"错误: {self.camera_name} 尚未标定，无法保存")
            return False
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON
        calibration_data = {
            "camera_name": self.camera_name,
            "camera_id": self.camera_id,
            "focal_length": float(self.focal_length),
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist(),
            "image_size": [
                self.calibration_images[0].shape[1],
                self.calibration_images[0].shape[0]
            ],
            "chessboard_size": self.chessboard_size,
            "square_size": self.square_size,
            "num_images": len(self.img_points),
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        json_path = os.path.join(output_dir, f"{self.camera_name}_calibration.json")
        with open(json_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # 保存为NumPy格式（方便后续使用）
        np_path = os.path.join(output_dir, f"{self.camera_name}_calibration.npz")
        np.savez(
            np_path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            focal_length=self.focal_length
        )
        
        print(f"✓ {self.camera_name} 标定结果已保存到 {output_dir}/")
        return True


def calibrate_single_camera(camera_id, camera_name, chessboard_size=(9, 6), square_size=25.0):
    """
    标定单个相机
    
    参数:
        camera_id: 相机ID
        camera_name: 相机名称
        chessboard_size: 棋盘格内角点数量
        square_size: 棋盘格方格大小（毫米）
    """
    calibrator = CameraCalibrator(camera_id, camera_name, chessboard_size, square_size)
    
    # 打开相机
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开相机 {camera_id} ({camera_name})")
        return None
    
    # 设置相机分辨率（可选）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print(f"\n{'='*60}")
    print(f"开始标定相机: {camera_name} (ID: {camera_id})")
    print(f"棋盘格大小: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
    print(f"方格大小: {square_size}mm")
    print(f"{'='*60}")
    print("\n操作说明:")
    print("  - 按 's' 键: 保存当前帧用于标定")
    print("  - 按 'c' 键: 执行标定")
    print("  - 按 'r' 键: 重置标定数据")
    print("  - 按 'q' 键: 退出")
    print(f"\n已收集: {len(calibrator.img_points)} 张图像")
    print(f"当前焦距: {'未标定' if not calibrator.calibrated else f'{calibrator.focal_length:.2f} 像素'}")
    
    window_name = f"Camera Calibration - {camera_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"错误: 无法从相机 {camera_id} 读取帧")
            break
        
        # 检测棋盘格
        found, corners, gray = calibrator.detect_chessboard(frame)
        
        # 绘制检测结果
        display_frame = frame.copy()
        if found:
            cv2.drawChessboardCorners(display_frame, calibrator.chessboard_size, corners, found)
            color = (0, 255, 0)  # 绿色
            status_text = "棋盘格已检测到 - 按 's' 保存"
        else:
            color = (0, 0, 255)  # 红色
            status_text = "未检测到棋盘格 - 调整位置"
        
        # 显示信息
        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # 显示已收集的图像数量
        cv2.putText(display_frame, f"已收集: {len(calibrator.img_points)} 张图像",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示当前焦距
        if calibrator.calibrated:
            focal_text = f"焦距: {calibrator.focal_length:.2f} 像素"
            cv2.putText(display_frame, focal_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(display_frame, "焦距: 未标定", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # 显示操作提示
        cv2.putText(display_frame, "按 's' 保存 | 'c' 标定 | 'r' 重置 | 'q' 退出",
                   (10, display_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            if found:
                # 优化角点位置
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                
                calibrator.obj_points.append(calibrator.objp)
                calibrator.img_points.append(corners2)
                calibrator.calibration_images.append(frame.copy())
                
                print(f"✓ 已保存第 {len(calibrator.img_points)} 张标定图像")
            else:
                print("✗ 未检测到棋盘格，无法保存")
        
        elif key == ord('c'):
            if len(calibrator.img_points) >= 10:
                calibrator.calibrate()
            else:
                print(f"✗ 标定图像不足 (需要至少10张，当前{len(calibrator.img_points)}张)")
        
        elif key == ord('r'):
            calibrator.obj_points = []
            calibrator.img_points = []
            calibrator.calibration_images = []
            calibrator.calibrated = False
            calibrator.focal_length = None
            print("✓ 已重置标定数据")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyWindow(window_name)
    
    return calibrator


def compare_focal_lengths(calibrator1, calibrator2):
    """比较两个相机的焦距"""
    if not calibrator1.calibrated or not calibrator2.calibrated:
        print("\n警告: 两个相机尚未完全标定，无法比较焦距")
        return
    
    focal1 = calibrator1.focal_length
    focal2 = calibrator2.focal_length
    diff = abs(focal1 - focal2)
    diff_percent = (diff / min(focal1, focal2)) * 100
    
    print(f"\n{'='*60}")
    print("焦距比较结果:")
    print(f"  {calibrator1.camera_name}: {focal1:.2f} 像素")
    print(f"  {calibrator2.camera_name}: {focal2:.2f} 像素")
    print(f"  差异: {diff:.2f} 像素 ({diff_percent:.2f}%)")
    
    if diff_percent < 1.0:
        print("  ✓ 焦距非常接近，适合立体视觉")
    elif diff_percent < 5.0:
        print("  ⚠ 焦距有轻微差异，可以使用但建议调整")
    else:
        print("  ✗ 焦距差异较大，建议调整相机设置使焦距一致")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='双相机棋盘格标定工具')
    parser.add_argument('--camera1', type=int, default=0, help='第一个相机ID (默认: 0)')
    parser.add_argument('--camera2', type=int, default=1, help='第二个相机ID (默认: 1)')
    parser.add_argument('--name1', type=str, default='Camera1', help='第一个相机名称')
    parser.add_argument('--name2', type=str, default='Camera2', help='第二个相机名称')
    parser.add_argument('--chessboard', type=str, default='9x6', 
                       help='棋盘格内角点数量 (格式: width x height, 默认: 9x6)')
    parser.add_argument('--square-size', type=float, default=25.0,
                       help='棋盘格方格大小，单位：毫米 (默认: 25.0)')
    
    args = parser.parse_args()
    
    # 解析棋盘格大小
    chessboard_size = tuple(map(int, args.chessboard.split('x')))
    
    print("="*60)
    print("双相机棋盘格标定工具")
    print("="*60)
    print(f"\n配置:")
    print(f"  相机1: ID={args.camera1}, 名称={args.name1}")
    print(f"  相机2: ID={args.camera2}, 名称={args.name2}")
    print(f"  棋盘格: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
    print(f"  方格大小: {args.square_size}mm")
    print()
    
    # 标定第一个相机
    print("\n" + "="*60)
    print("步骤 1/2: 标定第一个相机")
    print("="*60)
    calibrator1 = calibrate_single_camera(
        args.camera1, args.name1, chessboard_size, args.square_size
    )
    
    if calibrator1 is None:
        print("错误: 第一个相机标定失败")
        return
    
    # 如果已标定，保存结果
    if calibrator1.calibrated:
        calibrator1.save_calibration()
    
    # 标定第二个相机
    print("\n" + "="*60)
    print("步骤 2/2: 标定第二个相机")
    print("="*60)
    calibrator2 = calibrate_single_camera(
        args.camera2, args.name2, chessboard_size, args.square_size
    )
    
    if calibrator2 is None:
        print("错误: 第二个相机标定失败")
        return
    
    # 如果已标定，保存结果
    if calibrator2.calibrated:
        calibrator2.save_calibration()
    
    # 比较焦距
    if calibrator1.calibrated and calibrator2.calibrated:
        compare_focal_lengths(calibrator1, calibrator2)
        
        # 如果焦距差异较大，给出建议
        focal1 = calibrator1.focal_length
        focal2 = calibrator2.focal_length
        diff_percent = (abs(focal1 - focal2) / min(focal1, focal2)) * 100
        
        if diff_percent > 5.0:
            print("\n建议:")
            print("  如果焦距差异较大，可以尝试:")
            print("  1. 调整相机的焦距设置（如果支持）")
            print("  2. 调整相机到棋盘格的距离，使焦距接近")
            print("  3. 使用立体标定方法，OpenCV可以处理焦距差异")
    
    print("\n标定完成！")


if __name__ == "__main__":
    main()

