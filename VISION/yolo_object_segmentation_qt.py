import sys
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from pathlib import Path
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QDialog

# ------------------------------------------------------------
#  프로젝트 경로
# ------------------------------------------------------------
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    # ⭐ 특정 단일 파일이 아닌 runs_seg 폴더를 바라보도록 수정
    WEIGHT_DIR      = PROJECT_ROOT / "VISION" / "runs_seg"
    HANDEYE_YAML    = PROJECT_ROOT / "VISION" / "config" / "aruco_rigid_result.yaml"
    ROI_CONFIG      = PROJECT_ROOT / "VISION" / "config" / "roi_config.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "VISION" / "config" / "calibration_intrinsics.yaml"
    BASE_MODEL      = PROJECT_ROOT / "VISION" / "config" / "yolo11n-seg.pt"
else:
    WEIGHT_DIR      = PROJECT_ROOT / "PRAG" / "VISION" / "runs_seg"
    HANDEYE_YAML    = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "aruco_rigid_result.yaml"
    ROI_CONFIG      = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "roi_config.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "calibration_intrinsics.yaml"
    BASE_MODEL      = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "yolo11n-seg.pt"

# ------------------------------------------------------------
# Confidence (DB) 함수화
# ------------------------------------------------------------
def get_confidence():
    try:
        from DB.robot_db import RobotDB
        row = RobotDB().fetch_vision("vision_reliability")
        return float(row[2]) / 100.0 if row else 0.2
    except Exception:
        return 0.2


# ------------------------------------------------------------
# ROI Load
# ------------------------------------------------------------
def load_roi_from_yaml():
    if not ROI_CONFIG.exists():
        return None
    with open(ROI_CONFIG, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return tuple(data.get("roi", []))


class YOLOSegDetector:
    def __init__(
        self,
        weight_path=None,
        intrinsics_yaml=INTRINSICS_YAML,
        roi=None
    ):
        self.model = None
        self.roi = roi
        
        self.confidence = get_confidence()

        # 1. 카메라 매트릭스 로드
        with open(intrinsics_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.K = np.array(data["camera_matrix"], dtype=float)

        # 2. 지정된 가중치 로드 또는 디폴트 모델 자동 탐색
        if weight_path and Path(weight_path).exists():
            self.set_model(weight_path)
        else:
            # runs_seg 폴더 내의 첫 번째 best_*.pt 검색 후 로드
            first_model = next(WEIGHT_DIR.glob("best_*.pt"), None) if WEIGHT_DIR.exists() else None
            if first_model and first_model.exists():
                self.set_model(first_model)
            elif BASE_MODEL.exists():
                self.set_model(BASE_MODEL)
            else:
                print("[WARN] 사용할 수 있는 YOLO 모델 파일(.pt)이 없습니다.")

    def set_model(self, weight_path):
        """⭐ 메인 UI에서 제품 변경 시 동적으로 YOLO 모델을 교체하는 핵심 메서드"""
        try:
            p = Path(weight_path)
            if not p.exists():
                print(f"[ERROR] 모델 파일이 존재하지 않습니다: {p}")
                return False

            self.model = YOLO(str(p))
            print(f"[INFO] 🎯 비전 모델 동적 로드 완료: {p.name}")
            return True
        except Exception as e:
            print(f"[ERROR] 모델 로드 중 예외 발생: {e}")
            return False

    def update_confidence(self):
        """DB 신뢰도가 변경되었을 때 외부에서 호출해주는 메서드"""
        self.confidence = get_confidence()
        
    # --------------------------------------------------
    # 최소 외접 사각형 중심 기반 검출
    # -------------------------------------------------- 
    def detect_from_frames(self, color, depth, depth_scale):
        if color is None or depth is None or self.model is None:
            return [], None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx_k, cy_k = self.K[0, 2], self.K[1, 2]
        h, w = color.shape[:2]

        # 실시간 DB 신뢰도 적용
        conf_val = self.confidence

        # YOLO 추론
        results = self.model(
            color,
            conf=conf_val,
            imgsz=640,
            verbose=False,
            task="segment"
        )[0]

        if results.boxes is None or len(results.boxes) == 0 or results.masks is None:
            return [], color

        boxes = results.boxes
        raw_masks = results.masks.data.cpu().numpy()
        detections = []

        if self.roi:
            rx1, ry1, rx2, ry2 = map(int, self.roi)
        else:
            rx1, ry1, rx2, ry2 = 0, 0, w, h

        for i in range(len(boxes)):
            bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)

            mask_resized = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
            mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255

            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)

            rect = cv2.minAreaRect(largest_contour)
            (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect

            cx = int(rect_cx)
            cy = int(rect_cy)

            if cx <= 0 or cy <= 0:
                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)

            mask_area = cv2.contourArea(largest_contour)

            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

            win = 3
            roi_depth = depth[
                max(0, cy - win):cy + win + 1,
                max(0, cx - win):cx + win + 1
            ]

            valid = roi_depth[roi_depth > 0]
            if valid.size < 5:
                continue

            z = float(np.median(valid) * depth_scale)

            X = (cx - cx_k) * z / fx
            Y = (cy - cy_k) * z / fy

            detections.append({
                "label": self.model.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "center_xyz": (X, Y, z),
                "center_pixel": (cx, cy),
                "mask_area": mask_area
            })

            cv2.rectangle(color, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.circle(color, (cx, cy), 4, (0, 0, 255), -1)

        return detections, color


class YoloSegDialog(QDialog):
    def __init__(self, get_frame_func, parent=None):
        super().__init__(parent)

        self.get_frame_func = get_frame_func
        self.detector = YOLOSegDetector(roi=load_roi_from_yaml())

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.run_once)
        self.timer.start(50)

    def run_once(self):
        result = self.get_frame_func()
        if result is None:
            return

        color, depth, depth_scale = result
        detections, vis = self.detector.detect_from_frames(
            color.copy(),
            depth,
            depth_scale
        )