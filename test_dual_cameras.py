#!/usr/bin/env python3
"""
快速测试双相机是否可用
"""

import cv2
import sys

def test_dual_cameras(camera1_id, camera2_id):
    """测试两个相机是否可用"""
    print(f"\n测试相机 {camera1_id} 和 {camera2_id}...")
    
    # 打开相机1
    cap1 = cv2.VideoCapture(camera1_id)
    if not cap1.isOpened():
        print(f"✗ 相机 {camera1_id} 无法打开")
        return False
    
    # 打开相机2
    cap2 = cv2.VideoCapture(camera2_id)
    if not cap2.isOpened():
        print(f"✗ 相机 {camera2_id} 无法打开")
        cap1.release()
        return False
    
    # 尝试读取帧
    print("正在读取测试帧...")
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    cap1.release()
    cap2.release()
    
    if ret1 and ret2 and frame1 is not None and frame2 is not None:
        print(f"✓ 相机 {camera1_id}: 可用 (分辨率: {frame1.shape[1]}x{frame1.shape[0]})")
        print(f"✓ 相机 {camera2_id}: 可用 (分辨率: {frame2.shape[1]}x{frame2.shape[0]})")
        return True
    else:
        print(f"✗ 无法从相机读取帧")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='测试双相机')
    parser.add_argument('--camera1', type=int, default=2, help='第一个相机ID (默认: 2)')
    parser.add_argument('--camera2', type=int, default=4, help='第二个相机ID (默认: 4)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("双相机测试工具")
    print("="*60)
    
    if test_dual_cameras(args.camera1, args.camera2):
        print("\n✓ 两个相机都可用！")
        print(f"\n可以使用以下命令启动双相机控制:")
        print(f"  python dual_camera_control.py --camera1 {args.camera1} --camera2 {args.camera2}")
    else:
        print("\n✗ 相机测试失败")
        print("\n请检查:")
        print("  1. 相机是否正确连接")
        print("  2. 相机ID是否正确")
        print("  3. 相机是否被其他程序占用")
        print("  4. 相机驱动是否已安装")
        print("\n提示: 使用 'python test_cameras.py' 检测所有可用相机")
    
    print("="*60)

