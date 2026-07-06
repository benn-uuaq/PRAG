import sys
import cv2
import yaml
import numpy as np
# import pyrealsense2 as rs
from pathlib import Path

import subprocess
import platform

from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QListWidget,
    QMessageBox, QGridLayout, QLineEdit,
    QComboBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# =========================
# 경로
# =========================
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    TEMPLATE_ROOT = PROJECT_ROOT / "VISION" / "data" / "template"
    DB_DIR = PROJECT_ROOT / "VISION" / "data" / "template"
    ROI_YAML = PROJECT_ROOT / "VISION" / "config" / "roi_config.yaml"
else:
    TEMPLATE_ROOT = PROJECT_ROOT / "PRAG" / "VISION" / "data" / "template"
    DB_DIR = PROJECT_ROOT / "PRAG" / "VISION" / "data" / "template"
    ROI_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "roi_config.yaml"

TEMPLATE_ROOT.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


def extract_object(img, pad=10):
    H, W = img.shape[:2]

    border = 20
    bg_pixels = np.concatenate([
        img[0:border].reshape(-1, 3),
        img[:, 0:border].reshape(-1, 3),
        img[H-border:H].reshape(-1, 3),
        img[:, W-border:W].reshape(-1, 3),
    ], axis=0)

    bg_color = np.mean(bg_pixels, axis=0).astype(np.uint8)

    diff = np.abs(img.astype(np.int16) - bg_color)
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.erode(mask, kernel)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None

    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, W)
    y2 = min(y + h + pad, H)

    return img[y1:y2, x1:x2], mask[y1:y2, x1:x2]

# -----------------------------
# 템플릿 저장
# -----------------------------
def save_template(id_name, img, mask):
    out_dir = DB_DIR / id_name
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = list(out_dir.glob("template_*.png"))
    next_id = len(existing)

    cv2.imwrite(str(out_dir / f"template_{next_id}.png"), img)
    cv2.imwrite(str(out_dir / f"mask_{next_id}.png"), mask)

    with open(out_dir / "info.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"id": id_name}, f, allow_unicode=True)


# =========================
# 터치 키보드 제어
# =========================
def show_touch_keyboard():
    if platform.system() == "Windows":
        subprocess.Popen("osk", shell=True)

def hide_touch_keyboard():
    if platform.system() == "Windows":
        subprocess.Popen("taskkill /IM osk.exe /F", shell=True)


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

# =========================
# ROI 로드
# =========================
def load_roi():
    if ROI_YAML.exists():
        try:
            with open(ROI_YAML, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            roi = data.get("roi", None)
            if roi and len(roi) == 4:
                return list(map(int, roi))
        except Exception:
            return None
    return None

# =========================
# 템플릿 이름 입력 (대형 팝업) - 수정됨
# =========================
class TemplateNameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("템플릿 이름 입력")
        self.setFixedSize(700, 500)  # ✅ 드롭다운 추가로 높이를 조금 늘림

        self.name = None

        layout = QVBoxLayout(self)

        label = QLabel("저장할 템플릿 이름을 입력하세요")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font: bold 22pt 'Arial';")

        # ✅ 1. 드롭다운(ComboBox) 생성 및 기존 이름 로드
        self.combo = QComboBox()
        self.combo.setFixedHeight(60)
        self.combo.setStyleSheet("""
            QComboBox {
                font: 20pt 'Arial';
                padding-left: 10px;
            }
            QComboBox::drop-down {
                width: 50px;
            }
        """)
        
        # 기본 옵션 추가
        self.combo.addItem("=== 기존 이름 선택 ===")
        
        # TEMPLATE_ROOT 폴더에서 디렉토리 이름들 가져오기
        if TEMPLATE_ROOT.exists():
            # 폴더 이름순 정렬
            existing_names = sorted([
                p.name for p in TEMPLATE_ROOT.iterdir() if p.is_dir()
            ])
            self.combo.addItems(existing_names)

        # 콤보박스 선택 시 입력창에 텍스트 반영
        self.combo.currentIndexChanged.connect(self._on_combo_changed)

        # ---------------------------------------------------------

        self.edit = QLineEdit()
        self.edit.setFixedHeight(70)
        self.edit.setAlignment(Qt.AlignCenter)
        self.edit.setStyleSheet("font: 22pt 'Arial';")
        self.edit.setPlaceholderText("새로운 이름 입력") # 힌트 텍스트

        self.edit.setFocusPolicy(Qt.NoFocus)
        self.edit.mousePressEvent = self._on_edit_clicked
        self.edit.returnPressed.connect(self._on_return_pressed)

        btn_save = QPushButton("저장")
        btn_cancel = QPushButton("취소")

        for b in (btn_save, btn_cancel):
            b.setStyleSheet(BTN_STYLE)
            b.setFixedHeight(70)
            b.setDefault(False)
            b.setAutoDefault(False)

        btn_save.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        layout.addStretch()
        layout.addWidget(label)
        layout.addSpacing(20)
        
        # ✅ 레이아웃에 콤보박스 추가
        layout.addWidget(self.combo)
        layout.addSpacing(10)
        
        layout.addWidget(self.edit)
        layout.addSpacing(20)
        layout.addWidget(btn_save)
        layout.addWidget(btn_cancel)
        layout.addStretch()

    # ✅ 콤보박스 변경 시 호출되는 함수
    def _on_combo_changed(self, index):
        # 0번 인덱스는 "=== 기존 이름 선택 ===" 이므로 무시하거나 비움
        if index == 0:
            self.edit.clear()
        else:
            selected_text = self.combo.currentText()
            self.edit.setText(selected_text)

    def _on_edit_clicked(self, event):
        show_touch_keyboard()
        self.edit.setFocus(Qt.MouseFocusReason)
        QLineEdit.mousePressEvent(self.edit, event)

    def _on_return_pressed(self):
        hide_touch_keyboard()
        self.edit.clearFocus()

    def accept(self):
        text = self.edit.text().strip()
        if not text:
            QMessageBox.warning(self, "오류", "이름을 입력하세요.")
            return
        self.name = text
        hide_touch_keyboard()
        super().accept()

    def reject(self):
        hide_touch_keyboard()
        super().reject()

# =========================
# 촬영 결과 팝업 (RGB + MASK)
# =========================
class CaptureResultDialog(QDialog):
    def __init__(self, obj_img, mask_img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("촬영 결과 확인")
        self.setFixedSize(1100, 650)

        self.obj_img = obj_img
        self.mask_img = mask_img
        self.accepted_name = None

        self._init_ui()

    def _cv_to_pixmap(self, img):
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = rgb.shape
        qimg = QImage(rgb.tobytes(), w, h, w * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            520, 520, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def _init_ui(self):
        main = QVBoxLayout(self)

        top = QHBoxLayout()

        self.lbl_obj = QLabel()
        self.lbl_mask = QLabel()

        for lbl in (self.lbl_obj, self.lbl_mask):
            lbl.setFixedSize(520, 520)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background-color:black;")

        self.lbl_obj.setPixmap(self._cv_to_pixmap(self.obj_img))
        self.lbl_mask.setPixmap(self._cv_to_pixmap(self.mask_img))

        top.addWidget(self.lbl_obj)
        top.addWidget(self.lbl_mask)

        bottom = QHBoxLayout()
        btn_delete = QPushButton("삭제하기")
        btn_use = QPushButton("사용하기")

        for b in (btn_delete, btn_use):
            b.setStyleSheet(BTN_STYLE)
            b.setFixedHeight(70)

        btn_delete.clicked.connect(self.reject)
        btn_use.clicked.connect(self._on_use)

        bottom.addStretch()
        bottom.addWidget(btn_delete)
        bottom.addSpacing(20)
        bottom.addWidget(btn_use)
        bottom.addStretch()

        main.addLayout(top)
        main.addSpacing(20)
        main.addLayout(bottom)

    def _on_use(self):
        dlg = TemplateNameDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.accepted_name = dlg.name
            self.accept()

# =========================
# 메인 다이얼로그
# =========================
class TemplateCaptureDialog(QDialog):
    def __init__(self, get_frame_func, parent=None):
        super().__init__(parent)

        self.get_frame_func = get_frame_func

        self.setWindowTitle("Template Capture")
        self.setFixedSize(1600, 850)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self.roi = load_roi()
        self.last_frame = None
        self.camera_on = False

        self.current_dir = TEMPLATE_ROOT

        self._init_ui()
        self.load_list(TEMPLATE_ROOT)

    # ================= UI =================
    def _init_ui(self):
        main = QHBoxLayout(self)

        # -------- Left --------
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { font-size: 18pt; }
            QListWidget::item { height: 48px; }
        """)
        self.list_widget.itemClicked.connect(self.on_item_selected)
        self.list_widget.itemDoubleClicked.connect(self.on_item_activated)

        self.btn_delete = QPushButton("삭제하기")
        self.btn_delete.setStyleSheet(BTN_STYLE)
        self.btn_delete.setFixedHeight(60)
        self.btn_delete.clicked.connect(self.delete_selected)

        left = QVBoxLayout()
        left.addWidget(QLabel("템플릿 / 이미지 목록"))
        left.addWidget(self.list_widget)
        left.addWidget(self.btn_delete)

        # -------- Right --------
        self.viewer = QLabel()
        self.viewer.setFixedSize(800, 600)
        self.viewer.setAlignment(Qt.AlignCenter)
        self.viewer.setStyleSheet("background:black;")

        self.btn_cam_on = QPushButton("비전 켜기")
        self.btn_capture = QPushButton("촬영하기")
        self.btn_cam_off = QPushButton("비전 끄기")
        self.btn_exit = QPushButton("작업 종료")

        for b in (self.btn_cam_on, self.btn_capture, self.btn_cam_off, self.btn_exit):
            b.setStyleSheet(BTN_STYLE)
            b.setFixedHeight(60)

        self.btn_cam_on.clicked.connect(self.start_camera)
        self.btn_cam_off.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_exit.clicked.connect(self.close)

        grid = QGridLayout()
        grid.addWidget(self.btn_cam_on, 0, 0)
        grid.addWidget(self.btn_capture, 0, 1)
        grid.addWidget(self.btn_cam_off, 1, 0)
        grid.addWidget(self.btn_exit, 1, 1)

        right = QVBoxLayout()
        right.addStretch()
        right.addWidget(self.viewer, alignment=Qt.AlignCenter)
        right.addSpacing(20)
        right.addLayout(grid)
        right.addStretch()

        main.addLayout(left, 1)
        main.addLayout(right, 2)

    # ================= Camera (외부 프레임 소비) =================
    def start_camera(self):
        if self.camera_on:
            return
        self.camera_on = True
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        self.camera_on = False
        self.last_frame = None
        self.viewer.clear()

    def update_frame(self):
        if not self.camera_on:
            return

        result = self.get_frame_func()
        if result is None:
            return

        frame, _, _ = result
        if frame is None:
            return

        self.last_frame = frame.copy()
        display = frame.copy()

        if self.roi and len(self.roi) == 4:
            x1, y1, x2, y2 = self.roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(
            rgb.tobytes(),
            rgb.shape[1], rgb.shape[0],
            rgb.shape[1] * 3,
            QImage.Format_RGB888
        )
        self.viewer.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.viewer.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    # ================= Capture =================
    def capture(self):
        if not self.camera_on or self.last_frame is None:
            QMessageBox.warning(self, "오류", "카메라가 켜져 있지 않습니다.")
            return

        if not self.roi or len(self.roi) != 4:
            QMessageBox.warning(self, "오류", "ROI가 설정되어 있지 않습니다.\nroi_config.yaml을 먼저 설정하세요.")
            return

        x1, y1, x2, y2 = self.roi
        if x2 <= x1 or y2 <= y1:
            QMessageBox.warning(self, "오류", "ROI 값이 유효하지 않습니다.")
            return

        h, w = self.last_frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        roi_img = self.last_frame[y1:y2, x1:x2]
        if roi_img.size == 0:
            QMessageBox.warning(self, "오류", "ROI 이미지가 비어있습니다.")
            return

        obj, mask = extract_object(roi_img)
        if obj is None or mask is None:
            QMessageBox.warning(self, "실패", "객체 추출 실패")
            return

        dlg = CaptureResultDialog(obj, mask, self)
        if dlg.exec_() == QDialog.Accepted:
            save_template(dlg.accepted_name, obj, mask)
            self.load_list(self.current_dir)
            QMessageBox.information(self, "저장 완료", f"'{dlg.accepted_name}' 저장 완료")

    # ================= List =================
    def load_list(self, path: Path):
        self.list_widget.clear()
        self.current_dir = path

        if path != TEMPLATE_ROOT:
            self.list_widget.addItem("⬅ 뒤로가기")

        for p in sorted(path.iterdir()):
            if p.is_dir():
                self.list_widget.addItem(f"[DIR] {p.name}")
            else:
                self.list_widget.addItem(p.name)

    def on_item_selected(self, item):
        text = item.text()
        if text.startswith("[DIR]") or text.startswith("⬅"):
            return

        self.stop_camera()

        path = self.current_dir / text
        if not path.exists():
            return

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = QImage(
            img.tobytes(),
            img.shape[1], img.shape[0],
            img.shape[1] * 3,
            QImage.Format_RGB888
        )
        self.viewer.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.viewer.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def on_item_activated(self, item):
        text = item.text()
        if text.startswith("⬅"):
            self.load_list(TEMPLATE_ROOT)
            return
        if text.startswith("[DIR]"):
            folder = text.replace("[DIR] ", "")
            self.load_list(self.current_dir / folder)
            return

    def delete_selected(self):
        item = self.list_widget.currentItem()
        if not item:
            return

        text = item.text()
        if text.startswith("⬅"):
            return

        if text.startswith("[DIR]"):
            path = self.current_dir / text.replace("[DIR] ", "")
        else:
            path = self.current_dir / text

        if not path.exists():
            QMessageBox.warning(self, "오류", "삭제할 대상이 존재하지 않습니다.")
            return

        if QMessageBox.question(
            self,
            "삭제 확인",
            f"'{path.name}' 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        ) != QMessageBox.Yes:
            return

        try:
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
            else:
                path.unlink()

            self.load_list(self.current_dir)
            self.viewer.clear()

        except Exception as e:
            QMessageBox.critical(self, "삭제 실패", str(e))
