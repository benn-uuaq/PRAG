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
    WEIGHT_PATH = PROJECT_ROOT / "VISION" / "runs_seg" / "segment" / "train" / "weights" / "best.pt"
    HANDEYE_YAML = PROJECT_ROOT / "VISION" / "config" / "aruco_rigid_result.yaml"
    ROI_CONFIG = PROJECT_ROOT / "VISION" / "config" / "roi_config.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "VISION" / "config" / "calibration_intrinsics.yaml"
else:
    WEIGHT_PATH = PROJECT_ROOT / "PRAG" / "VISION" / "runs_seg" / "segment" / "train" / "weights" / "best.pt"
    HANDEYE_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "aruco_rigid_result.yaml"
    ROI_CONFIG = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "roi_config.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "calibration_intrinsics.yaml"

# ------------------------------------------------------------
# Confidence (DB)
# ------------------------------------------------------------
from DB.robot_db import RobotDB
row = RobotDB().fetch_vision("vision_reliability")
try:
    CONFIDENCE = float(row[2]) / 100.0 if row else 0.2
except Exception:
    CONFIDENCE = 0.2


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
        weight_path=WEIGHT_PATH,
        intrinsics_yaml=INTRINSICS_YAML,
        roi=None
    ):
        self.model = YOLO(weight_path)
        self.roi = roi

        with open(intrinsics_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.K = np.array(data["camera_matrix"], dtype=float)

    # # --------------------------------------------------
    # # 바운딩 박스 중심점
    # # --------------------------------------------------
    # def detect_from_frames(self, color, depth, depth_scale):
    #     if color is None or depth is None:
    #         return [], None

    #     fx, fy = self.K[0, 0], self.K[1, 1]
    #     cx_k, cy_k = self.K[0, 2], self.K[1, 2]

    #     results = self.model(
    #         color,
    #         conf=CONFIDENCE,
    #         imgsz=640,
    #         verbose=False,
    #         task="segment"
    #     )[0]

    #     if results.boxes is None or len(results.boxes.xyxy) == 0:
    #         return [], color

    #     boxes = results.boxes
    #     detections = []

    #     if self.roi:
    #         rx1, ry1, rx2, ry2 = map(int, self.roi)
    #     else:
    #         rx1, ry1, rx2, ry2 = 0, 0, color.shape[1], color.shape[0]

    #     for i in range(len(boxes.xyxy)):
    #         bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)

    #         cx = int((bx1 + bx2) / 2)
    #         cy = int((by1 + by2) / 2)

    #         if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
    #             continue

    #         win = 3
    #         roi_depth = depth[
    #             max(0, cy - win):cy + win + 1,
    #             max(0, cx - win):cx + win + 1
    #         ]

    #         valid = roi_depth[roi_depth > 0]
    #         if valid.size < 5:
    #             continue

    #         z = float(np.median(valid) * depth_scale)

    #         X = (cx - cx_k) * z / fx
    #         Y = (cy - cy_k) * z / fy

    #         detections.append({
    #             "label": self.model.names[int(boxes.cls[i])],
    #             "confidence": float(boxes.conf[i]),
    #             "center_xyz": (X, Y, z),
    #             "center_pixel": (cx, cy)
    #         })

    #         # 시각화
    #         cv2.rectangle(color, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    #         cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)

    #     return detections, color

    # # --------------------------------------------------
    # # 마스크 내접원 중심점
    # # --------------------------------------------------
    # def detect_from_frames(self, color, depth, depth_scale):
    #     if color is None or depth is None:
    #         return [], None

    #     fx, fy = self.K[0, 0], self.K[1, 1]
    #     cx_k, cy_k = self.K[0, 2], self.K[1, 2]
    #     h, w = color.shape[:2]

    #     # YOLO 모델 추론 (Segmentation 태스크)
    #     results = self.model(
    #         color,
    #         conf=0.5, # 예시 CONFIDENCE, 실제 값으로 대체 필요
    #         imgsz=640,
    #         verbose=False,
    #         task="segment"
    #     )[0]

    #     # 박스나 마스크가 없으면 조기 리턴
    #     if results.boxes is None or len(results.boxes) == 0 or results.masks is None:
    #         return [], color

    #     boxes = results.boxes
    #     # 마스크 데이터 가져오기 (GPU 텐서를 numpy 배열로 변환)
    #     raw_masks = results.masks.data.cpu().numpy()

    #     detections = []

    #     if self.roi:
    #         rx1, ry1, rx2, ry2 = map(int, self.roi)
    #     else:
    #         rx1, ry1, rx2, ry2 = 0, 0, w, h

    #     for i in range(len(boxes)):
    #         # --- [기존 코드 제거] 바운딩 박스 중심 계산 ---
    #         bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)
    #         # cx = int((bx1 + bx2) / 2)
    #         # cy = int((by1 + by2) / 2)
    #         # ----------------------------------------

    #         # --- [신규 코드 추가] 마스크 기반 내접원 중심 계산 ---
    #         # 1. 현재 객체의 마스크 추출 및 원본 이미지 크기로 리사이즈
    #         # YOLO 마스크는 보통 이미지보다 작으므로 맞춰줘야 정확한 좌표가 나옵니다.
    #         mask_resized = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
            
    #         # 2. 이진화 (Binary Mask 생성)
    #         mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255

    #         # 3. 거리 변환 (Distance Transform) 수행
    #         # 마스크 내부의 각 픽셀에서 가장 가까운 0(배경)까지의 거리를 계산합니다.
    #         dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)

    #         # 4. 최대값(내접원 반지름)과 그 위치(내접원 중심) 찾기
    #         # minMaxLoc은 (minVal, maxVal, minLoc, maxLoc)을 반환합니다.
    #         radius_float, _, _, max_loc = cv2.minMaxLoc(dist_transform)
            
    #         cx, cy = max_loc # 내접원의 중심 좌표
    #         radius = int(radius_float) # 내접원의 반지름
    #         # ----------------------------------------

    #         # ROI 필터링 (새로 계산된 중심점 기준)
    #         if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
    #             continue

    #         # 깊이 추출 (기존 로직 유지, 중심점만 변경됨)
    #         win = 3
    #         roi_depth = depth[
    #             max(0, cy - win):cy + win + 1,
    #             max(0, cx - win):cx + win + 1
    #         ]

    #         valid = roi_depth[roi_depth > 0]
    #         if valid.size < 5:
    #             # 마스크가 너무 얇거나 깊이 정보가 없으면 스킵
    #             continue

    #         z = float(np.median(valid) * depth_scale)

    #         # 3D 좌표 변환
    #         X = (cx - cx_k) * z / fx
    #         Y = (cy - cy_k) * z / fy

    #         detections.append({
    #             "label": self.model.names[int(boxes.cls[i])],
    #             "confidence": float(boxes.conf[i]),
    #             "center_xyz": (X, Y, z),
    #             "center_pixel": (cx, cy),
    #             "radius": radius # 결과에 반지름 정보도 포함 (선택사항)
    #         })

    #         # --- [시각화 수정] 내접원 및 중심점 그리기 ---
    #         # 1. 바운딩 박스 (참고용으로 유지)
    #         cv2.rectangle(color, (bx1, by1), (bx2, by2), (0, 255, 0), 1)
            
    #         # 2. 내접원 그리기 (노란색 실선)
    #         # radius가 너무 작으면 안 보일 수 있으므로 최소값 보정
    #         safe_radius = max(radius, 2) 
    #         cv2.circle(color, (cx, cy), safe_radius, (0, 255, 255), 2)

    #         # 3. 중심점 그리기 (빨간색 채워진 원)
    #         cv2.circle(color, (cx, cy), 2, (0, 0, 255), -1)
    #         # ----------------------------------------

    #     return detections, color
    

    # # --------------------------------------------------
    # # 마스크 무게 중심점
    # # --------------------------------------------------   
    # def detect_from_frames(self, color, depth, depth_scale):
    #     if color is None or depth is None:
    #         return [], None

    #     fx, fy = self.K[0, 0], self.K[1, 1]
    #     cx_k, cy_k = self.K[0, 2], self.K[1, 2]
    #     h, w = color.shape[:2]

    #     # YOLO 추론
    #     results = self.model(
    #         color,
    #         conf=0.5,
    #         imgsz=640,
    #         verbose=False,
    #         task="segment"
    #     )[0]

    #     if results.boxes is None or len(results.boxes) == 0 or results.masks is None:
    #         return [], color

    #     boxes = results.boxes
    #     raw_masks = results.masks.data.cpu().numpy()
    #     detections = []

    #     if self.roi:
    #         rx1, ry1, rx2, ry2 = map(int, self.roi)
    #     else:
    #         rx1, ry1, rx2, ry2 = 0, 0, w, h

    #     for i in range(len(boxes)):
    #         # 1. 바운딩 박스 좌표 추출 (시각화용)
    #         bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)

    #         # 2. 마스크 처리 (중심점 계산용 - 무게 중심)
    #         mask_resized = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
    #         mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255

    #         # 구멍 메우기 (Hole Filling)
    #         contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         if not contours:
    #             continue
            
    #         largest_contour = max(contours, key=cv2.contourArea)
    #         cv2.drawContours(mask_uint8, [largest_contour], -1, 255, -1) # 내부 채우기

    #         # 무게 중심(Moments) 계산
    #         M = cv2.moments(mask_uint8)
    #         if M["m00"] != 0:
    #             cx = int(M["m10"] / M["m00"])
    #             cy = int(M["m01"] / M["m00"])
    #         else:
    #             cx = int((bx1 + bx2) / 2)
    #             cy = int((by1 + by2) / 2)

    #         # ROI 필터링
    #         if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
    #             continue

    #         # 깊이 추출 (중심점 주변)
    #         win = 3
    #         roi_depth = depth[
    #             max(0, cy - win):cy + win + 1,
    #             max(0, cx - win):cx + win + 1
    #         ]

    #         valid = roi_depth[roi_depth > 0]
    #         if valid.size < 5:
    #             continue

    #         z = float(np.median(valid) * depth_scale)

    #         # 3D 좌표 변환
    #         X = (cx - cx_k) * z / fx
    #         Y = (cy - cy_k) * z / fy

    #         detections.append({
    #             "label": self.model.names[int(boxes.cls[i])],
    #             "confidence": float(boxes.conf[i]),
    #             "center_xyz": (X, Y, z),
    #             "center_pixel": (cx, cy),
    #             "mask_area": M["m00"]
    #         })

    #         # ────────────────────────────────────────────────────────
    #         # [시각화 변경]
    #         # 중심점은 마스크 기준(정확함), 외곽선은 바운딩 박스(깔끔함)
    #         # ────────────────────────────────────────────────────────
            
    #         # 1. 바운딩 박스 그리기 (초록색)
    #         cv2.rectangle(color, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
    #         # 2. 중심점 그리기 (빨간색 점)
    #         cv2.circle(color, (cx, cy), 4, (0, 0, 255), -1)
            
    #         # (선택사항) 중심점 좌표 텍스트 표시
    #         # cv2.putText(color, f"{X:.3f}, {Y:.3f}, {z:.3f}", (bx1, by1 - 10), 
    #         #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    #     return detections, color
    
    # --------------------------------------------------
    # 최소 외접 사각형 중심
    # -------------------------------------------------- 
    def detect_from_frames(self, color, depth, depth_scale):
        if color is None or depth is None:
            return [], None

        fx, fy = self.K[0, 0], self.K[1, 1]
        cx_k, cy_k = self.K[0, 2], self.K[1, 2]
        h, w = color.shape[:2]

        # YOLO 추론
        results = self.model(
            color,
            conf=0.5,
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
            # 1. 바운딩 박스 좌표 추출 (시각화용)
            bx1, by1, bx2, by2 = boxes.xyxy[i].cpu().numpy().astype(int)

            # 2. 마스크 처리
            mask_resized = cv2.resize(raw_masks[i], (w, h), interpolation=cv2.INTER_LINEAR)
            mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255

            # 외곽선 추출
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            largest_contour = max(contours, key=cv2.contourArea)

            # ────────────────────────────────────────────────────────
            # [수정된 부분] 최소 외접 사각형(Rotated Rect)의 중심 사용
            # ────────────────────────────────────────────────────────
            # 물체가 회전되어 있어도, 그 형태를 감싸는 최소 크기의 사각형을 구합니다.
            rect = cv2.minAreaRect(largest_contour)
            (rect_cx, rect_cy), (rect_w, rect_h), rect_angle = rect

            cx = int(rect_cx)
            cy = int(rect_cy)

            # 예외 처리: 데이터가 깨져서 좌표가 이상할 경우 바운딩 박스 중심으로 대체
            if cx <= 0 or cy <= 0:
                cx = int((bx1 + bx2) / 2)
                cy = int((by1 + by2) / 2)

            # 면적 계산 (기존 M["m00"] 대체용)
            mask_area = cv2.contourArea(largest_contour)
            # ────────────────────────────────────────────────────────

            # ROI 필터링
            if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                continue

            # 깊이 추출 (중심점 주변)
            win = 3
            roi_depth = depth[
                max(0, cy - win):cy + win + 1,
                max(0, cx - win):cx + win + 1
            ]

            valid = roi_depth[roi_depth > 0]
            if valid.size < 5:
                continue

            z = float(np.median(valid) * depth_scale)

            # 3D 좌표 변환
            X = (cx - cx_k) * z / fx
            Y = (cy - cy_k) * z / fy

            detections.append({
                "label": self.model.names[int(boxes.cls[i])],
                "confidence": float(boxes.conf[i]),
                "center_xyz": (X, Y, z),
                "center_pixel": (cx, cy),
                "mask_area": mask_area # 수정된 면적 값 저장
            })

            # ────────────────────────────────────────────────────────
            # [시각화]
            # ────────────────────────────────────────────────────────
            # 1. 바운딩 박스 그리기 (초록색)
            cv2.rectangle(color, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            
            # (선택사항) 최소 외접 사각형 시각화 (어떻게 잡히는지 보고 싶다면 주석 해제)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            # cv2.drawContours(color, [box], 0, (255, 0, 0), 2) # 파란색 기울어진 사각형
            
            # 2. 새로운 중심점 그리기 (빨간색 점)
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
