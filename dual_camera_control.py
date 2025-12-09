#!/usr/bin/env python3
"""
双相机同步控制工具
支持同时拍照、录制视频，视频内嵌时间码
适用于任务3f的双视角同步拍摄
"""

import cv2
import numpy as np
import threading
import time
import os
from datetime import datetime
import argparse
from queue import Queue
import json

class DualCameraController:
    def __init__(self, camera1_id, camera2_id, camera1_name="Camera1", camera2_name="Camera2"):
        """
        初始化双相机控制器
        
        参数:
            camera1_id: 第一个相机ID
            camera2_id: 第二个相机ID
            camera1_name: 第一个相机名称
            camera2_name: 第二个相机名称
        """
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.camera1_name = camera1_name
        self.camera2_name = camera2_name
        
        # 相机对象
        self.cap1 = None
        self.cap2 = None
        
        # 录制状态
        self.recording = False
        self.video_writer1 = None
        self.video_writer2 = None
        
        # 同步时间戳
        self.start_time = None
        self.frame_timestamps = []
        
        # 输出目录
        self.output_dir = "dual_camera_output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 帧队列（用于多线程）
        self.frame_queue1 = Queue(maxsize=10)
        self.frame_queue2 = Queue(maxsize=10)
        
    def initialize_cameras(self, width=1280, height=720, fps=30):
        """初始化两个相机"""
        print(f"正在初始化相机...")
        print(f"  {self.camera1_name}: ID={self.camera1_id}")
        print(f"  {self.camera2_name}: ID={self.camera2_id}")
        
        # 打开相机1
        self.cap1 = cv2.VideoCapture(self.camera1_id)
        if not self.cap1.isOpened():
            raise Exception(f"无法打开相机 {self.camera1_id} ({self.camera1_name})")
        
        # 打开相机2
        self.cap2 = cv2.VideoCapture(self.camera2_id)
        if not self.cap2.isOpened():
            self.cap1.release()
            raise Exception(f"无法打开相机 {self.camera2_id} ({self.camera2_name})")
        
        # 设置相机参数
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap1.set(cv2.CAP_PROP_FPS, fps)
        
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap2.set(cv2.CAP_PROP_FPS, fps)
        
        # 获取实际设置
        actual_width1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
        
        actual_width2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
        
        print(f"\n相机设置:")
        print(f"  {self.camera1_name}: {actual_width1}x{actual_height1} @ {actual_fps1:.1f} fps")
        print(f"  {self.camera2_name}: {actual_width2}x{actual_height2} @ {actual_fps2:.1f} fps")
        
        return (actual_width1, actual_height1), (actual_width2, actual_height2)
    
    def add_timestamp(self, frame, timestamp, elapsed_time=None):
        """
        在帧上添加时间码
        
        参数:
            frame: 图像帧
            timestamp: 时间戳字符串
            elapsed_time: 从开始录制经过的时间（秒）
        """
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # 背景矩形（半透明）
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
        
        # 显示时间戳
        cv2.putText(display_frame, f"Time: {timestamp}",
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示经过时间
        if elapsed_time is not None:
            elapsed_str = f"Elapsed: {elapsed_time:.3f}s"
            cv2.putText(display_frame, elapsed_str,
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return display_frame
    
    def capture_frame(self, cap, queue):
        """从相机捕获帧（用于多线程）"""
        while True:
            ret, frame = cap.read()
            if ret:
                if not queue.full():
                    queue.put((ret, frame, time.time()))
                else:
                    # 队列满时，丢弃最旧的帧
                    try:
                        queue.get_nowait()
                        queue.put((ret, frame, time.time()))
                    except:
                        pass
            else:
                queue.put((False, None, time.time()))
                break
    
    def capture_sync_photos(self):
        """同时拍摄两张照片"""
        print("\n准备同时拍照...")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("拍照!")
        
        # 同时读取两帧
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            print("错误: 拍照失败")
            return False
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        filename1 = os.path.join(self.output_dir, f"{self.camera1_name}_{timestamp}.jpg")
        filename2 = os.path.join(self.output_dir, f"{self.camera2_name}_{timestamp}.jpg")
        
        # 保存照片
        cv2.imwrite(filename1, frame1)
        cv2.imwrite(filename2, frame2)
        
        print(f"✓ 照片已保存:")
        print(f"  {filename1}")
        print(f"  {filename2}")
        
        # 保存同步信息
        sync_info = {
            "timestamp": timestamp,
            "camera1_file": filename1,
            "camera2_file": filename2,
            "datetime": datetime.now().isoformat()
        }
        sync_file = os.path.join(self.output_dir, f"sync_{timestamp}.json")
        with open(sync_file, 'w') as f:
            json.dump(sync_info, f, indent=2)
        
        return True
    
    def start_recording(self, fps=30, codec='XVID'):
        """开始同时录制视频"""
        if self.recording:
            print("警告: 已经在录制中")
            return False
        
        print("\n准备开始录制...")
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("开始录制!")
        
        # 获取第一帧以确定尺寸
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            print("错误: 无法读取初始帧")
            return False
        
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename1 = os.path.join(self.output_dir, f"{self.camera1_name}_{timestamp}.avi")
        filename2 = os.path.join(self.output_dir, f"{self.camera2_name}_{timestamp}.avi")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.video_writer1 = cv2.VideoWriter(filename1, fourcc, fps, (w1, h1))
        self.video_writer2 = cv2.VideoWriter(filename2, fourcc, fps, (w2, h2))
        
        if not self.video_writer1.isOpened() or not self.video_writer2.isOpened():
            print("错误: 无法创建视频文件")
            return False
        
        self.recording = True
        self.start_time = time.time()
        self.frame_timestamps = []
        
        print(f"✓ 开始录制:")
        print(f"  {filename1}")
        print(f"  {filename2}")
        print(f"  按 's' 键停止录制")
        
        # 保存录制信息
        record_info = {
            "start_time": datetime.now().isoformat(),
            "camera1_file": filename1,
            "camera2_file": filename2,
            "fps": fps,
            "resolution1": [w1, h1],
            "resolution2": [w2, h2]
        }
        record_file = os.path.join(self.output_dir, f"record_{timestamp}.json")
        with open(record_file, 'w') as f:
            json.dump(record_info, f, indent=2)
        
        return True
    
    def stop_recording(self):
        """停止录制"""
        if not self.recording:
            print("警告: 当前没有在录制")
            return False
        
        self.recording = False
        
        # 释放视频写入器
        if self.video_writer1:
            self.video_writer1.release()
        if self.video_writer2:
            self.video_writer2.release()
        
        self.video_writer1 = None
        self.video_writer2 = None
        
        elapsed = time.time() - self.start_time
        print(f"\n✓ 录制已停止")
        print(f"  录制时长: {elapsed:.2f} 秒")
        print(f"  总帧数: {len(self.frame_timestamps)} 帧")
        
        # 保存时间戳信息
        if self.frame_timestamps:
            timestamp_file = os.path.join(self.output_dir, 
                                        f"timestamps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(timestamp_file, 'w') as f:
                json.dump({
                    "start_time": self.start_time,
                    "frame_timestamps": self.frame_timestamps,
                    "total_frames": len(self.frame_timestamps),
                    "duration": elapsed
                }, f, indent=2)
            print(f"  时间戳已保存: {timestamp_file}")
        
        self.start_time = None
        self.frame_timestamps = []
        
        return True
    
    def record_frame(self, frame1, frame2, timestamp1, timestamp2):
        """录制一帧（带时间码）"""
        if not self.recording:
            return
        
        # 计算经过时间
        elapsed = time.time() - self.start_time
        
        # 生成时间戳字符串
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 添加时间码到帧
        frame1_with_tc = self.add_timestamp(frame1, timestamp_str, elapsed)
        frame2_with_tc = self.add_timestamp(frame2, timestamp_str, elapsed)
        
        # 写入视频
        self.video_writer1.write(frame1_with_tc)
        self.video_writer2.write(frame2_with_tc)
        
        # 记录时间戳
        self.frame_timestamps.append({
            "frame_num": len(self.frame_timestamps),
            "timestamp1": timestamp1,
            "timestamp2": timestamp2,
            "elapsed": elapsed,
            "datetime": timestamp_str
        })
    
    def run_interactive(self):
        """运行交互式控制界面"""
        print("\n" + "="*60)
        print("双相机同步控制")
        print("="*60)
        print("\n操作说明:")
        print("  'p' - 同时拍照")
        print("  'r' - 开始/停止录制")
        print("  'q' - 退出")
        print("  'i' - 显示相机信息")
        print("\n提示: 录制时视频会内嵌时间码")
        print("="*60 + "\n")
        
        window1 = f"{self.camera1_name} - Dual Camera Control"
        window2 = f"{self.camera2_name} - Dual Camera Control"
        
        cv2.namedWindow(window1, cv2.WINDOW_NORMAL)
        cv2.namedWindow(window2, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # 读取帧
                ret1, frame1 = self.cap1.read()
                ret2, frame2 = self.cap2.read()
                
                if not ret1 or not ret2:
                    print("错误: 无法读取帧")
                    break
                
                # 如果正在录制，添加时间码并录制
                if self.recording:
                    timestamp1 = time.time()
                    timestamp2 = time.time()
                    self.record_frame(frame1, frame2, timestamp1, timestamp2)
                    
                    # 显示录制状态
                    elapsed = time.time() - self.start_time
                    status_text = f"REC {elapsed:.1f}s"
                    cv2.putText(frame1, status_text, (frame1.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame2, status_text, (frame2.shape[1] - 150, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # 显示帧
                cv2.imshow(window1, frame1)
                cv2.imshow(window2, frame2)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    self.capture_sync_photos()
                elif key == ord('r'):
                    if self.recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('i'):
                    self.show_camera_info()
        
        finally:
            # 清理
            if self.recording:
                self.stop_recording()
            self.cleanup()
            cv2.destroyAllWindows()
    
    def show_camera_info(self):
        """显示相机信息"""
        print("\n" + "="*60)
        print("相机信息")
        print("="*60)
        
        if self.cap1:
            print(f"\n{self.camera1_name} (ID: {self.camera1_id}):")
            print(f"  宽度: {int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))}")
            print(f"  高度: {int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {self.cap1.get(cv2.CAP_PROP_FPS):.1f}")
            print(f"  亮度: {self.cap1.get(cv2.CAP_PROP_BRIGHTNESS)}")
            print(f"  对比度: {self.cap1.get(cv2.CAP_PROP_CONTRAST)}")
        
        if self.cap2:
            print(f"\n{self.camera2_name} (ID: {self.camera2_id}):")
            print(f"  宽度: {int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))}")
            print(f"  高度: {int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
            print(f"  FPS: {self.cap2.get(cv2.CAP_PROP_FPS):.1f}")
            print(f"  亮度: {self.cap2.get(cv2.CAP_PROP_BRIGHTNESS)}")
            print(f"  对比度: {self.cap2.get(cv2.CAP_PROP_CONTRAST)}")
        
        print(f"\n录制状态: {'正在录制' if self.recording else '未录制'}")
        print(f"输出目录: {self.output_dir}")
        print("="*60 + "\n")
    
    def cleanup(self):
        """清理资源"""
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
        if self.video_writer1:
            self.video_writer1.release()
        if self.video_writer2:
            self.video_writer2.release()
        print("资源已释放")


def main():
    parser = argparse.ArgumentParser(
        description='双相机同步控制工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
功能说明:
  - 同时拍照: 按 'p' 键，两个相机会同时拍摄照片
  - 同时录制: 按 'r' 键开始/停止录制，视频会自动内嵌时间码
  - 时间码格式: 显示日期时间和从开始录制的经过时间

输出文件:
  - 照片: {camera_name}_{timestamp}.jpg
  - 视频: {camera_name}_{timestamp}.avi
  - 同步信息: sync_{timestamp}.json
  - 时间戳数据: timestamps_{timestamp}.json

所有文件保存在: dual_camera_output/ 目录
        """
    )
    parser.add_argument('--camera1', type=int, default=2, 
                       help='第一个相机ID (默认: 2)')
    parser.add_argument('--camera2', type=int, default=4, 
                       help='第二个相机ID (默认: 4)')
    parser.add_argument('--name1', type=str, default='Camera2', 
                       help='第一个相机名称 (默认: Camera2)')
    parser.add_argument('--name2', type=str, default='Camera4', 
                       help='第二个相机名称 (默认: Camera4)')
    parser.add_argument('--width', type=int, default=1280, 
                       help='视频宽度 (默认: 1280)')
    parser.add_argument('--height', type=int, default=720, 
                       help='视频高度 (默认: 720)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='帧率 (默认: 30)')
    
    args = parser.parse_args()
    
    # 创建控制器
    controller = DualCameraController(
        args.camera1, args.camera2, 
        args.name1, args.name2
    )
    
    try:
        # 初始化相机
        controller.initialize_cameras(
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        
        # 运行交互式界面
        controller.run_interactive()
    
    except Exception as e:
        print(f"错误: {e}")
        controller.cleanup()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

