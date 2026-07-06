import yaml
import sys
from pathlib import Path

from PyQt5.QtWidgets import QDialog, QMessageBox, QPushButton, QVBoxLayout, QLabel


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent
    

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    YAML_PATH = PROJECT_ROOT / "VISION" / "config" / "calibration_intrinsics.yaml"
else:
    YAML_PATH = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "calibration_intrinsics.yaml"

# ------------------------------------------------------------
# Camera Intrinsics Dialog (VisionWorker 연동형)
# ------------------------------------------------------------
class CamIntrinsicsDialog(QDialog):
    """
    ✔ VisionWorker에서 intrinsics만 받아서 저장
    ✔ pipeline / parent / vision_main 접근 ❌
    """

    def __init__(self, get_intrinsics_func, parent=None):
        super().__init__(parent)

        self.get_intrinsics_func = get_intrinsics_func

        self.setWindowTitle("Camera Intrinsics Calibration")
        self.setFixedSize(420, 180)

        self._init_ui()

    # ---------------- UI ----------------
    def _init_ui(self):
        layout = QVBoxLayout(self)

        info = QLabel("현재 카메라 Intrinsics를 YAML로 저장합니다.")
        info.setWordWrap(True)

        self.btn_run = QPushButton("캘리브레이션 실행")
        self.btn_close = QPushButton("닫기")

        layout.addWidget(info)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_close)

        self.btn_run.clicked.connect(self.run_calibration)
        self.btn_close.clicked.connect(self.close)

    # ---------------- Logic ----------------
    def run_calibration(self):
        try:
            """
            VisionWorker.get_intrinsics() 사용
            """
            intr = self.get_intrinsics_func()
            if intr is None:
                QMessageBox.critical(self, "Error", "카메라 Intrinsics를 가져올 수 없습니다.")
                return

            K = [
                [intr["fx"], 0, intr["ppx"]],
                [0, intr["fy"], intr["ppy"]],
                [0, 0, 1]
            ]

            data = {
                "camera_matrix": K,
                "dist_coeffs": intr["dist"],
                "image_size": {
                    "width": intr["width"],
                    "height": intr["height"]
                }
            }

            YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(YAML_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
        finally:

            QMessageBox.information(
                self,
                "완료",
                f"Intrinsics 저장 완료\n\n{YAML_PATH}"
            )