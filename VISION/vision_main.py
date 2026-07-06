from pathlib import Path
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal


class VisionMain(QObject):
    frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, project_root: Path, parent=None):
        super().__init__(parent)

        self.root = project_root
        self.vision_dir = self.root / "VISION"

        self.visualize_enabled = False

        from .vision_worker import VisionWorker
        self.worker = VisionWorker()

        # Worker → UI 프레임 전달
        self.worker.frame_signal.connect(self.frame_signal.emit)

    # --------------------------------------------------
    # 카메라 제어
    # --------------------------------------------------
    def start_camera(self):
        if self.worker.isRunning():
            return
        self.worker.visualize_enabled = self.visualize_enabled
        self.worker.start()
        print("[VISION] Camera started")

    def stop_camera(self):
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        print("[VISION] Camera stopped")

    def set_visualize(self, enabled: bool):
        self.visualize_enabled = enabled
        self.worker.visualize_enabled = enabled

    # --------------------------------------------------
    # 로봇 작업용 1회 검출
    # --------------------------------------------------
    def detect_pick_once(self):
        if not self.worker.isRunning():
            self.start_camera()
        return self.worker.detect_pick_once()

    # --------------------------------------------------
    # 🔥 비전 툴 호출 (이제 subprocess ❌)
    # --------------------------------------------------
    def calibrate_camera_intrinsics(self, parent=None):
        """
        카메라 내부 파라미터 캘리브레이션
        """
        if not self.worker.isRunning():
            self.start_camera()
        from VISION.cam_intrinsics_qt import CamIntrinsicsDialog
        dlg = CamIntrinsicsDialog(
            get_intrinsics_func=self.worker.get_intrinsics,
            parent=parent
        )
        if dlg.exec_(): 
            self.worker.reload_vision_config()

    def select_roi(self, parent=None):
        """
        ROI 선택
        """
        from VISION.roi_selector_qt import ROIDialog
        # 🔥 카메라가 꺼져 있으면 켜준다
        if not self.worker.isRunning():
            self.start_camera()

        dlg = ROIDialog(
            get_frame_func=self.worker.get_latest_frames,
            parent=parent
        )
        dlg.exec_()
        self.worker.reload_vision_config()

    def capture_and_generate_templates(self, parent=None):
        """
        템플릿 캡처
        """
        from VISION.shape_capture_and_add_qt import TemplateCaptureDialog
        if not self.worker.isRunning():
            self.start_camera()
        dlg = TemplateCaptureDialog(
            get_frame_func=self.worker.get_latest_frames,
            parent=parent
        )
        dlg.exec_()

    def train_yolo_segmentation(self, parent=None):
        """
        YOLO 학습
        """
        if not self.worker.isRunning():
            self.start_camera()
        from VISION.auto_yolo_seg_qt import AutoYoloSegDialog
        dlg = AutoYoloSegDialog(
            get_frame_func=self.worker.get_latest_frames, 
            close_camera_func=self.stop_camera,
            parent=parent)
        dlg.exec_()
        self.worker.reload_vision_config()

    def calibrate_handeye(self, parent=None):
        """
        Hand–Eye 캘리브레이션
        """
        if not self.worker.isRunning():
            self.start_camera()
        from VISION.aruco_rigid_calibration_qt import ArucoCalibrationDialog
        dlg = ArucoCalibrationDialog(
            get_frame_func=self.worker.get_latest_frames,
            parent=parent
        )
        dlg.exec_()
        self.worker.reload_vision_config()
