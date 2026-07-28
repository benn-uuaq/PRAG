from PyQt5.QtCore import QThread, pyqtSignal, QMutex
import numpy as np
import pyrealsense2 as rs
import sys
from pathlib import Path

from VISION.robot_yolo_seg_qt import YOLOToRobotQt


class VisionWorker(QThread):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.visualize_enabled = False

        self.pipeline = None
        self.align = None
        self.depth_scale = None
        
        # 스레드 간 데이터 공유 시 충돌 방지를 위한 자물쇠(Mutex)
        self.mutex = QMutex()
        self.latest_color = None
        self.latest_depth = None
        
        # YOLO 래퍼 초기화
        self.yolo_qt = YOLOToRobotQt()
        
        print("[VISION WORKER] Initialized with YOLO")
        
    def update_confidence(self):
        """YOLOQt 래퍼의 신뢰도 갱신 메서드 호출"""
        if hasattr(self, 'yolo_qt') and self.yolo_qt:
            self.yolo_qt.update_confidence()

    def run(self):
        """카메라 스트림 시작 (무한 루프)"""
        self.running = True
        
        # RealSense 초기화
        try:
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            profile = self.pipeline.start(cfg)
            self.align = rs.align(rs.stream.color)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
            print("[VISION WORKER] Camera stream started")
        except Exception as e:
            print(f"[VISION WORKER ERROR] Camera init failed: {e}")
            self.running = False
            return

        while self.running:
            try:
                # 1. 프레임 받기
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                aligned = self.align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                color = np.asanyarray(color_frame.get_data())
                depth = np.asanyarray(depth_frame.get_data())

                # 2. 🔥 [핵심] 최신 프레임을 변수에 백업 (Lock 사용)
                self.mutex.lock()
                self.latest_color = color.copy()
                self.latest_depth = depth.copy()
                self.mutex.unlock()

                # 3. UI 전송 (Visualizing)
                if self.visualize_enabled:
                    # 실시간 객체 탐지 + 시각화
                    vis_img = self.yolo_qt.visualize(color, depth, self.depth_scale)
                    self.frame_signal.emit(vis_img)
                else:
                    # 일반 카메라 프레임만 전송
                    self.frame_signal.emit(color)

            except Exception as e:
                print("[VISION WORKER ERROR]", e)
                break

        # 정리
        if self.pipeline:
            self.pipeline.stop()
        print("[VISION WORKER] Camera stream stopped")

    def stop(self):
        """카메라 스트림 중지"""
        self.running = False
        self.wait()

    def detect_pick_once(self):
        """
        1회 검출 (로봇 작업용)
        - 카메라에 새로 요청하지 않고, run()이 저장해둔 최신 프레임을 훔쳐서 씁니다.
        Returns: (success, base_xyz, result, img)
        """
        try:
            if not self.running:
                print("[VISION WORKER] Camera is NOT running. Cannot detect.")
                return False, None, None, None

            # 1. 🔥 [핵심] 카메라 호출 없이, 메모리에 저장된 최신 프레임 가져오기
            self.mutex.lock()
            try:
                if self.latest_color is None or self.latest_depth is None:
                    print("[VISION WORKER] No frame captured yet.")
                    return False, None, None, None
                color = self.latest_color.copy()
                depth = self.latest_depth.copy()
            finally:
                self.mutex.unlock()

            # 2. 가져온 프레임으로 좌표 계산 (이미지 처리는 빠르므로 여기서 수행해도 됨)
            success, base_xyz, result, img = self.yolo_qt.detect_once(
                color, depth, self.depth_scale
            )

            return success, base_xyz, result, img

        except Exception as e:
            print(f"[VISION WORKER] detect_pick_once failed: {e}")
            return False, None, None, None
        
    def get_latest_frames(self):
        """
        VisionMain / 각 QT Tool에서 공통으로 쓰는 프레임 공급 함수
        Returns:
            (color(np.ndarray), depth(np.ndarray), depth_scale(float)) or None
        """
        if not self.running:
            return None

        self.mutex.lock()
        try:
            if self.latest_color is None or self.latest_depth is None:
                return None
            color = self.latest_color.copy()
            depth = self.latest_depth.copy()
            depth_scale = float(self.depth_scale) if self.depth_scale is not None else None
        finally:
            self.mutex.unlock()

        if depth_scale is None:
            return None

        return (color, depth, depth_scale)
    
    def get_intrinsics(self):
        """
        RealSense intrinsics 반환
        """
        if self.pipeline is None:
            return None

        profile = self.pipeline.get_active_profile()
        stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()

        return {
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "dist": list(intr.coeffs[:5]),
            "width": intr.width,
            "height": intr.height,
        }
        
    def reload_vision_config(self):
        """비전 설정(ROI, Calib 등) 핫 리로드"""
        self.mutex.lock()
        try:
            if self.yolo_qt:
                self.yolo_qt.reload_config()
                print("[VISION WORKER] YOLO wrapper reloaded.")
        finally:
            self.mutex.unlock()
            
    def change_model(self, model_path: str):
        """
        UI에서 선택한 제품의 단독 모델 가중치로 안전하게 교체
        """
        self.mutex.lock()
        try:
            # ⭐ [수정] YOLOToRobotQt에 실제로 존재하는 change_target_model()을 호출.
            # 예전엔 없는 load_model()을 hasattr로 찾다 실패하고, 존재하지도 않는
            # weight_path 속성에 값을 넣은 뒤 reload_config()를 호출해서
            # (reload_config는 current_weight_path를 사용하므로) 실제로는 모델이
            # 전혀 바뀌지 않는 버그가 있었음.
            ok = self.yolo_qt.change_target_model(str(model_path))

            if ok:
                print(f"[VISION WORKER] 모델 교체 완료: {model_path}")
            else:
                print(f"[VISION WORKER ERROR] 모델 교체 실패(로드 거부됨): {model_path}")
        except Exception as e:
            print(f"[VISION WORKER ERROR] 모델 교체 실패: {e}")
        finally:
            self.mutex.unlock()

