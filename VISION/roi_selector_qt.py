import sys
import cv2
import yaml
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ------------------------------------------------------------
# 경로 설정
# ------------------------------------------------------------
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    ROI_YAML = PROJECT_ROOT / "VISION" / "config" / "roi_config.yaml"
else:
    ROI_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "roi_config.yaml"

ROI_YAML.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 버튼 스타일
# ------------------------------------------------------------
BTN_STYLE = """
QPushButton {
    color: rgb(255, 255, 255);
    background-color: rgb(0, 119, 196);
    font: bold 16pt "Arial";
    border: 1px solid rgb(0, 0, 0);
}
QPushButton:pressed {
    background-color: rgb(0, 80, 150);
}
"""


# ------------------------------------------------------------
# ROI Selector Dialog (라이브러리형)
# ------------------------------------------------------------
class ROIDialog(QDialog):
    def __init__(self, get_frame_func, parent=None):
        super().__init__(parent)

        self.get_frame_func = get_frame_func

        self.setWindowTitle("ROI Selector")
        self.setFixedSize(900, 520)

        # ROI 상태
        self.start_pt = None
        self.end_pt = None
        self.roi = None
        self.drawing = False

        self.frame = None

        # UI
        self._init_ui()

        # 타이머 (VisionWorker 프레임 사용)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------------- UI ----------------
    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet("background-color: black;")
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release

        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(20)

        self.btn_save = QPushButton("저장하기")
        self.btn_exit = QPushButton("종료하기")

        for btn in (self.btn_save, self.btn_exit):
            btn.setStyleSheet(BTN_STYLE)
            btn.setFixedSize(200, 60)

        self.btn_save.clicked.connect(self.save_roi)
        self.btn_exit.clicked.connect(self.close)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_exit)
        btn_layout.addStretch()

        main_layout.addWidget(self.image_label)
        main_layout.addLayout(btn_layout)

    # ---------------- 영상 업데이트 ----------------
    def update_frame(self):
        result = self.get_frame_func()
        if result is None:
            return

        color, _, _ = result
        self.frame = color.copy()
        display = self.frame.copy()

        if self.start_pt and self.end_pt:
            cv2.rectangle(display, self.start_pt, self.end_pt, (0, 0, 255), 2)

        if self.roi:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        qimg = QImage(
            display.data,
            display.shape[1],
            display.shape[0],
            display.strides[0],
            QImage.Format_BGR888
        )

        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    # ---------------- 마우스 이벤트 ----------------
    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self.roi = None
            self.start_pt = (event.x(), event.y())
            self.end_pt = self.start_pt
            self.drawing = True

    def mouse_move(self, event):
        if self.drawing:
            self.end_pt = (event.x(), event.y())

    def mouse_release(self, event):
        if self.drawing:
            self.end_pt = (event.x(), event.y())
            self.drawing = False

            x1, y1 = self.start_pt
            x2, y2 = self.end_pt
            self.roi = (
                min(x1, x2),
                min(y1, y2),
                max(x1, x2),
                max(y1, y2)
            )

    # ---------------- 저장 ----------------
    def save_roi(self):
        if not self.roi:
            QMessageBox.warning(
                self,
                "저장 실패",
                "ROI가 설정되지 않았습니다.\n영역을 드래그해주세요."
            )
            return

        roi = list(map(int, self.roi))
        data = {"roi": roi}

        with open(ROI_YAML, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f)

        QMessageBox.information(
            self,
            "저장 완료",
            f"ROI = {roi}\n저장 완료"
        )

    # ---------------- 종료 ----------------
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()
