import sys
import cv2
import yaml
import numpy as np
import random
import shutil
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QTextEdit, QPushButton,
    QVBoxLayout, QHBoxLayout, QMessageBox, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt

# ==============================
# YOLO import
# ==============================
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# ==============================
# 경로 설정
# ==============================
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    SHAPE_DB_DIR = PROJECT_ROOT / "VISION/data/template"
    BACKGROUND_DIR = PROJECT_ROOT / "VISION/data/backgrounds"
    
    ROI_YAML     = PROJECT_ROOT / "VISION/config/roi_config.yaml"
    DATASET_DIR  = PROJECT_ROOT / "VISION/data/dataset_seg"
    WEIGHT_PATH  = PROJECT_ROOT / "VISION/runs_seg"
    BASE_MODEL   = PROJECT_ROOT / "VISION/config/yolo11n-seg.pt"
else:
    SHAPE_DB_DIR = PROJECT_ROOT / "PRAG/VISION/data/template"
    BACKGROUND_DIR = PROJECT_ROOT / "PRAG/VISION/data/backgrounds"
    
    ROI_YAML     = PROJECT_ROOT / "PRAG/VISION/config/roi_config.yaml"
    DATASET_DIR  = PROJECT_ROOT / "PRAG/VISION/data/dataset_seg"
    WEIGHT_PATH  = PROJECT_ROOT / "PRAG/VISION/runs_seg"
    BASE_MODEL   = PROJECT_ROOT / "PRAG/VISION/config/yolo11n-seg.pt"

IMG_TRAIN = DATASET_DIR / "images/train"
IMG_VAL   = DATASET_DIR / "images/val"
LBL_TRAIN = DATASET_DIR / "labels/train"
LBL_VAL   = DATASET_DIR / "labels/val"
DATA_YAML = DATASET_DIR / "dataset.yaml"

# ✅ 총 100장 생성
IMAGES_PER_CLASS = 100

# ==============================
# StreamRedirect (로그 가로채기)
# ==============================
class StreamRedirect(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

# ==============================
# 유틸 (ROI, Templates 로드)
# ==============================
def load_roi():
    if not ROI_YAML.exists(): return (0,0,640,480)
    with open(ROI_YAML, "r", encoding="utf-8") as f:
        return tuple(map(int, yaml.safe_load(f)["roi"]))

def load_templates():
    templates, label_map = {}, {}
    idx = 0
    if not SHAPE_DB_DIR.exists(): return {}, {}

    for folder in sorted(SHAPE_DB_DIR.iterdir()):
        if not folder.is_dir(): continue
        cls = folder.name
        buf = []
        for p in folder.glob("template_*.png"):
            num = p.stem.split("_")[-1]
            m = folder / f"mask_{num}.png"
            rgb = cv2.imread(str(p))
            mask = cv2.imread(str(m), cv2.IMREAD_GRAYSCALE)
            if rgb is None or mask is None: continue
            
            hard = (mask > 10).astype(np.uint8) * 255
            if hard.ndim == 3: hard = hard[:, :, 0]
            rgba = cv2.cvtColor(rgb, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = hard
            buf.append((rgba, hard))
        if buf:
            templates[cls] = buf
            label_map[cls] = idx
            idx += 1
    return templates, label_map

def mask_to_polygon(mask, W, H):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: return None
    eps = 0.01 * cv2.arcLength(c, True)
    poly = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32)
    poly[:, 0] /= W
    poly[:, 1] /= H
    return poly.flatten().tolist()

# ==============================
# Worker (Dataset 생성 + 학습)
# ==============================
class YoloTrainWorker(QThread):
    log = pyqtSignal(str)
    stream_log = pyqtSignal(str)
    done = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        self.redirector = StreamRedirect()
        self.redirector.text_written.connect(self.handle_stream_output)

        sys.stdout = self.redirector
        sys.stderr = self.redirector
        
        try:
            if YOLO is None:
                self.done.emit(False, "ultralytics 미설치")
                return

            if DATASET_DIR.exists():
                shutil.rmtree(DATASET_DIR)

            IMG_TRAIN.mkdir(parents=True, exist_ok=True)
            IMG_VAL.mkdir(parents=True, exist_ok=True)
            LBL_TRAIN.mkdir(parents=True, exist_ok=True)
            LBL_VAL.mkdir(parents=True, exist_ok=True)

            roi = load_roi()
            templates, label_map = load_templates()
            
            real_backgrounds = []
            if BACKGROUND_DIR.exists():
                for ext in ["*.jpg", "*.png", "*.jpeg"]:
                    real_backgrounds.extend(list(BACKGROUND_DIR.glob(ext)))
            
            if not real_backgrounds:
                self.log.emit("[WARN] 실제 배경 이미지가 없습니다. (화이트 배경으로 대체)")

            if not self._is_running:
                self.done.emit(False, "중지됨")
                return

            W, H = 640, 480
            idx = 0

            for cls, tmpl_list in templates.items():
                cls_id = label_map[cls]
                self.log.emit(f"[DATA] {cls} 데이터 생성 중... (총 {IMAGES_PER_CLASS}장)")

                for i in range(IMAGES_PER_CLASS):
                    
                    # -----------------------------------------------------
                    # 배경 생성 비율 (20:20:30:30)
                    # -----------------------------------------------------
                    if i < 20: # 화이트
                        bg = np.full((H, W, 3), 255, dtype=np.uint8)
                    elif i < 40: # 블랙
                        bg = np.zeros((H, W, 3), dtype=np.uint8)
                    elif i < 70: # 촬영된 실제 배경
                        if real_backgrounds:
                            bg_path = random.choice(real_backgrounds)
                            bg_img = cv2.imread(str(bg_path))
                            if bg_img is not None:
                                bg = cv2.resize(bg_img, (W, H))
                            else:
                                bg = np.full((H, W, 3), 255, dtype=np.uint8)
                        else:
                            bg = np.full((H, W, 3), 255, dtype=np.uint8)
                    else: # 템플릿 배경
                        t_rgba, _ = random.choice(tmpl_list)
                        bg = cv2.resize(t_rgba[:, :, :3], (W, H))

                    # 오브젝트 합성
                    masks = []
                    for _ in range(random.randint(15, 30)):
                        rgba, hard = random.choice(tmpl_list)
                        h_obj, w_obj = rgba.shape[:2]
                        
                        safe_x = max(roi[0], roi[2] - w_obj)
                        safe_y = max(roi[1], roi[3] - h_obj)
                        
                        x = random.randint(roi[0], safe_x)
                        y = random.randint(roi[1], safe_y)

                        alpha = rgba[:, :, 3:4] / 255.0
                        bg_roi = bg[y:y+h_obj, x:x+w_obj]
                        if bg_roi.shape[:2] != (h_obj, w_obj): continue 

                        bg[y:y+h_obj, x:x+w_obj] = (
                            rgba[:, :, :3] * alpha +
                            bg_roi * (1 - alpha)
                        ).astype(np.uint8)

                        mask = np.zeros((H, W), np.uint8)
                        mask[y:y+h_obj, x:x+w_obj] = hard
                        masks.append(mask)

                    # 라벨링
                    labels = []
                    front = np.zeros((H, W), np.uint8)

                    for m in reversed(masks):
                        vis = m.copy()
                        vis[front > 0] = 0 
                        if np.count_nonzero(vis) / max(1, np.count_nonzero(m)) < 0.7: continue

                        poly = mask_to_polygon(vis, W, H)
                        if poly:
                            labels.append(f"{cls_id} " + " ".join(map(str, poly)))
                            front[m > 0] = 1

                    if not labels: continue

                    # 저장
                    if i % 5 == 0: is_val = True
                    else: is_val = False

                    img_dir = IMG_VAL if is_val else IMG_TRAIN
                    lbl_dir = LBL_VAL if is_val else LBL_TRAIN

                    cv2.imwrite(str(img_dir / f"{idx:06d}.jpg"), bg)
                    with open(lbl_dir / f"{idx:06d}.txt", "w") as f:
                        f.write("\n".join(labels))

                    idx += 1

            # 데이터셋 정보 저장
            with open(DATA_YAML, "w", encoding="utf-8") as f:
                yaml.safe_dump({
                    "path": str(DATASET_DIR),
                    "train": "images/train",
                    "val": "images/val",
                    "names": list(label_map.keys())
                }, f)
                
            if not self._is_running:
                self.done.emit(False, "중지됨")
                return

            self.log.emit("[YOLO] 데이터셋 준비 완료. 학습 시작...")

            model = YOLO(BASE_MODEL)
            model.train(
                data=str(DATA_YAML),
                epochs=10,
                imgsz=640,
                batch=32,
                device=0,
                project=str(WEIGHT_PATH),
                name="segment/train",
                exist_ok=True,
                workers=4,
                verbose=True,
                patience=5,
                cache=True         
            )

            self.done.emit(True, "학습 완료")

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.done.emit(False, str(e))
            
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
    def handle_stream_output(self, text):
        self.stream_log.emit(text)

# ==============================
# UI Dialog
# ==============================
class AutoYoloSegDialog(QDialog):
    def __init__(self, get_frame_func=None, close_camera_func=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YOLO Segmentation Training")
        self.setFixedSize(1400, 900)
        
        self.get_frame_func = get_frame_func
        self.close_camera_func = close_camera_func
        self.worker = None

        # 로그창
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap) 
        self.log_view.setStyleSheet("background-color: white; color: black; font-family: Consolas;")

        # 버튼들
        self.btn_start = QPushButton("학습 시작")
        self.btn_stop = QPushButton("학습 중지")
        self.btn_close = QPushButton("닫기")
        
        self.btn_start.setFixedHeight(60)
        self.btn_stop.setFixedHeight(60)
        self.btn_close.setFixedHeight(60)
        
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_close.clicked.connect(self.close)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)   
        btns.addWidget(self.btn_close)

        layout = QVBoxLayout(self)
        layout.addWidget(self.log_view)
        layout.addLayout(btns)

    def start(self):
        """학습 시작: 배경 선택(기존/신규) -> 학습 워커 실행"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "경고", "이미 학습 중입니다.")
            return
        
        self.log_view.clear()
        
        # -----------------------------------------------------
        # 1. 기존 배경 이미지 확인 및 사용자 선택
        # -----------------------------------------------------
        use_existing_bg = False
        existing_files = []
        
        if BACKGROUND_DIR.exists():
            # 이미지 파일(.jpg, .png 등)이 있는지 확인
            existing_files = [
                f for f in BACKGROUND_DIR.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']
            ]
            
            if existing_files:
                reply = QMessageBox.question(
                    self, '배경 이미지 확인', 
                    f'기존 배경 폴더에 이미지가 {len(existing_files)}장 있습니다.\n'
                    '기존 이미지를 그대로 사용하여 학습하시겠습니까?\n'
                    '(No를 선택하면 기존 이미지를 삭제하고 새로 촬영합니다.)',
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                
                if reply == QMessageBox.Yes:
                    use_existing_bg = True

        # -----------------------------------------------------
        # 2. 배경 촬영 로직 (기존 배경 안 쓸 때만 실행)
        # -----------------------------------------------------
        if use_existing_bg:
            self.append_log(f"[배경] 기존 배경 이미지 {len(existing_files)}장을 사용하여 학습을 진행합니다.")
            self.append_log("[배경] 촬영 단계를 건너뜁니다.")
        else:
            if self.get_frame_func is None:
                 self.append_log("[오류] 카메라 함수가 없습니다. 배경 촬영을 건너뜁니다.")
            else:
                self.append_log("[배경] 배경 이미지를 새로 촬영합니다...")
                
                # 1. 폴더 초기화 (기존 이미지 삭제)
                if BACKGROUND_DIR.exists():
                    shutil.rmtree(BACKGROUND_DIR)
                BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)
                
                QApplication.processEvents() # UI 갱신

                # 2. 촬영 루프
                for i in range(4):
                    time.sleep(1.0) # 1초 대기 (조명 변경 등)
                    self.append_log(f"[배경] 촬영 중... ({i+1}/4)")
                    QApplication.processEvents()

                    result = self.get_frame_func()
                    if result is None:
                        self.append_log(f"[배경] {i+1}번 프레임 획득 실패")
                        continue
                    
                    color, _, _ = result
                    if color is None:
                        continue
                    
                    save_path = BACKGROUND_DIR / f"bg_{i+1:02d}.jpg"
                    cv2.imwrite(str(save_path), color)
                    self.append_log(f"[배경] 저장됨: {save_path.name}")
                    QApplication.processEvents()
                
                self.append_log("[배경] 촬영 완료. 데이터 생성을 시작합니다.")

        # -----------------------------------------------------
        # 3. 워커 시작
        # -----------------------------------------------------
        self.worker = YoloTrainWorker()
        self.worker.log.connect(self.append_log)
        self.worker.stream_log.connect(self.append_stream_log)
        self.worker.done.connect(self.on_done)
        self.worker.start()
        
    def stop(self):
        if not self.worker or not self.worker.isRunning():
            return

        reply = QMessageBox.question(
            self, '중지 확인', 
            '현재 작업을 중지하시겠습니까?\n(데이터 생성 중이면 즉시 중단, 학습 중이면 강제 종료됩니다.)',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.append_log("[시스템] 중지 요청 중...")
        self.worker.stop()
        
        if not self.worker.wait(5000):
            self.append_log("[시스템] 응답이 없어 강제 종료합니다 (Terminate).")
            self.worker.terminate()
            self.worker.wait()
        
        self.worker = None
        self.on_done(False, "사용자에 의해 중지됨")

    def on_done(self, ok, msg):
        QMessageBox.information(self, "완료" if ok else "실패", msg)
        self.worker = None
        
    def append_log(self, msg):
        self.log_view.append(msg)

    def append_stream_log(self, msg):
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.End)

        if '\r' in msg:
            chunks = msg.split('\r')
            for i, chunk in enumerate(chunks):
                if i > 0: 
                    cursor.movePosition(cursor.StartOfBlock, cursor.KeepAnchor)
                    cursor.removeSelectedText()
                cursor.insertText(chunk)
        else:
            cursor.insertText(msg)
        
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()
        
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '종료 확인', 
                '작업이 진행 중입니다. 강제 종료하시겠습니까?',
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                self.worker.wait()
                
                if self.close_camera_func:
                    self.close_camera_func()
                    
                event.accept()
            else:
                event.ignore()
        else:
            if self.close_camera_func:
                self.close_camera_func()
                
            event.accept()
            
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     dlg = AutoYoloSegDialog()
#     dlg.show()
#     sys.exit(app.exec_())