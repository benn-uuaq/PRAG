import gc
import torch
import sys
import cv2
import yaml
import numpy as np
import random
import shutil
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QTextEdit, QPushButton, QLabel,
    QVBoxLayout, QHBoxLayout, QMessageBox, QApplication,
    QListWidget, QListWidgetItem, QWidget
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
    SHAPE_DB_DIR = PROJECT_ROOT / "PRAG_260721_test/VISION/data/template"
    BACKGROUND_DIR = PROJECT_ROOT / "PRAG_260721_test/VISION/data/backgrounds"
    
    ROI_YAML     = PROJECT_ROOT / "PRAG_260721_test/VISION/config/roi_config.yaml"
    DATASET_DIR  = PROJECT_ROOT / "PRAG_260721_test/VISION/data/dataset_seg"
    WEIGHT_PATH  = PROJECT_ROOT / "PRAG_260721_test/VISION/runs_seg"
    BASE_MODEL   = PROJECT_ROOT / "PRAG_260721_test/VISION/config/yolo11n-seg.pt"

IMG_TRAIN = DATASET_DIR / "images/train"
IMG_VAL   = DATASET_DIR / "images/val"
LBL_TRAIN = DATASET_DIR / "labels/train"
LBL_VAL   = DATASET_DIR / "labels/val"
DATA_YAML = DATASET_DIR / "dataset.yaml"

# ✅ 총 200장 생성
IMAGES_PER_CLASS = 200

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

def load_templates(selected_classes=None):
    templates, label_map = {}, {}
    idx = 0
    if not SHAPE_DB_DIR.exists(): return {}, {}

    for folder in sorted(SHAPE_DB_DIR.iterdir()):
        if not folder.is_dir(): continue
        cls = folder.name
        
        # ⭐ 선택된 클래스 목록이 있다면 해당 클래스만 로드
        if selected_classes and cls not in selected_classes:
            continue
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
# Worker (선택된 제품별 단독 학습)
# ==============================
class YoloTrainWorker(QThread):
    log = pyqtSignal(str)
    stream_log = pyqtSignal(str)
    done = pyqtSignal(bool, str)
    
    def __init__(self, selected_classes=None):
        super().__init__()
        self._is_running = True
        self.selected_classes = selected_classes or []

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

            # ⭐ [추가 1] 고객사 시스템 GPU 유무 자동 감지 및 하드웨어별 파라미터 최적화
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                target_device = 0
                train_batch = 16       # VRAM OOM 방지를 위해 32 -> 16으로 하향
                train_workers = 2      # 윈도우 핀메모리 에러 방지 (4 -> 2)
                self.log.emit("[SYSTEM] 🟢 GPU가 감지되었습니다. (CUDA GPU 학습 모드)")
            else:
                target_device = "cpu"
                train_batch = 8        # CPU 메모리 및 연산 부하 완화를 위해 8로 하향
                train_workers = 0      # CPU 환경에서는 0으로 해야 멀티프로세싱 병목 없음
                self.log.emit("[SYSTEM] 🟡 GPU가 없습니다. (고객사 CPU 전용 학습 모드)")

            roi = load_roi()
            templates, _ = load_templates(self.selected_classes)
            
            if not templates:
                self.done.emit(False, "선택한 템플릿 데이터를 찾을 수 없습니다.")
                return

            # 배경 이미지 로드
            real_backgrounds = []
            if BACKGROUND_DIR.exists():
                for ext in ["*.jpg", "*.png", "*.jpeg"]:
                    real_backgrounds.extend(list(BACKGROUND_DIR.glob(ext)))
            
            if not real_backgrounds:
                self.log.emit("[WARN] 실제 배경 이미지가 없습니다. (화이트/블랙/템플릿 배경으로만 생성)")

            W, H = 640, 480
            total_classes = len(templates)
            current_cls_idx = 0

            # ==========================================================
            # ⭐ 선택된 제품(클래스)별로 루프를 돌며 단독 모델 학습
            # ==========================================================
            for cls, tmpl_list in templates.items():
                if not self._is_running:
                    self.done.emit(False, "중지됨")
                    return

                current_cls_idx += 1
                self.log.emit(f"\n==================================================")
                self.log.emit(f"[YOLO] ({current_cls_idx}/{total_classes}) '{cls}' 제품 단독 모델 작업 시작")
                self.log.emit(f"==================================================")

                # 1. 제품별 임시 데이터셋 폴더 생성
                cls_dataset_dir = DATASET_DIR / cls
                if cls_dataset_dir.exists():
                    shutil.rmtree(cls_dataset_dir)

                img_train = cls_dataset_dir / "images/train"
                img_val   = cls_dataset_dir / "images/val"
                lbl_train = cls_dataset_dir / "labels/train"
                lbl_val   = cls_dataset_dir / "labels/val"

                # ⭐ [수정] 200장의 이미지가 모두 완벽히 생성되어 있는지 확실하게 검사
                existing_img_count = len(list(img_train.glob("*.jpg"))) if img_train.exists() else 0
                has_existing_data = (existing_img_count >= IMAGES_PER_CLASS)

                if has_existing_data:
                    # 200장 이상 완벽히 존재할 때만 보존 및 스킵
                    self.log.emit(f"[INFO] '{cls}' 기존 데이터셋({existing_img_count}장)이 완벽히 보존되어 있어 이미지 생성을 건너뜁니다.")
                else:
                    if existing_img_count > 0:
                        self.log.emit(f"[WARN] '{cls}' 불완전한 데이터셋({existing_img_count}/{IMAGES_PER_CLASS}장) 감지. 폴더를 비우고 재생성합니다.")
                        
                    # 데이터가 없거나 200장 미만으로 불완전한 경우 폴더 초기화 및 새로 생성
                    if cls_dataset_dir.exists():
                        shutil.rmtree(cls_dataset_dir)

                    for p in [img_train, img_val, lbl_train, lbl_val]:
                        p.mkdir(parents=True, exist_ok=True)

                    self.log.emit(f"[DATA] '{cls}' 합성 이미지 생성 중... (총 {IMAGES_PER_CLASS}장)")

                    # 2. 합성 이미지 및 라벨 생성
                    for i in range(IMAGES_PER_CLASS):
                        if not self._is_running: break

                        if i < 20:   # 화이트
                            bg = np.full((H, W, 3), 255, dtype=np.uint8)
                        elif i < 40: # 블랙
                            bg = np.zeros((H, W, 3), dtype=np.uint8)
                        elif i < 70: # 실제 촬영 배경
                            if real_backgrounds:
                                bg_path = random.choice(real_backgrounds)
                                bg_img = cv2.imread(str(bg_path))
                                bg = cv2.resize(bg_img, (W, H)) if bg_img is not None else np.full((H, W, 3), 255, dtype=np.uint8)
                            else:
                                bg = np.full((H, W, 3), 255, dtype=np.uint8)
                        else:        # 템플릿 배경
                            t_rgba, _ = random.choice(tmpl_list)
                            bg = cv2.resize(t_rgba[:, :, :3], (W, H))

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

                            bg[y:y+h_obj, x:x+w_obj] = (rgba[:, :, :3] * alpha + bg_roi * (1 - alpha)).astype(np.uint8)

                            mask = np.zeros((H, W), np.uint8)
                            mask[y:y+h_obj, x:x+w_obj] = hard
                            masks.append(mask)

                        # ⭐ 단독 모델이므로 무조건 클래스 ID는 0번 고정!
                        labels = []
                        front = np.zeros((H, W), np.uint8)

                        for m in reversed(masks):
                            vis = m.copy()
                            vis[front > 0] = 0 
                            if np.count_nonzero(vis) / max(1, np.count_nonzero(m)) < 0.7: continue

                            poly = mask_to_polygon(vis, W, H)
                            if poly:
                                labels.append("0 " + " ".join(map(str, poly)))
                                front[m > 0] = 1

                        if not labels: continue

                        is_val = (i % 5 == 0)
                        cv2.imwrite(str((img_val if is_val else img_train) / f"{i:06d}.jpg"), bg)
                        with open((lbl_val if is_val else lbl_train) / f"{i:06d}.txt", "w") as f:
                            f.write("\n".join(labels))

                # 3. 제품 단독 yaml 파일 생성
                cls_yaml = cls_dataset_dir / f"dataset_{cls}.yaml"
                with open(cls_yaml, "w", encoding="utf-8") as f:
                    yaml.safe_dump({
                        "path": str(cls_dataset_dir),
                        "train": "images/train",
                        "val": "images/val",
                        "names": {0: cls}
                    }, f)

                if not self._is_running: break

                # ⭐ [추가] 이미 학습된 최적 가중치 파일(best_{cls}.pt)이 존재하는지 확인
                target_best = WEIGHT_PATH / f"best_{cls}.pt"
                
                if target_best.exists():
                    self.log.emit(f"[INFO] '{cls}' 제품의 학습된 모델 파일({target_best.name})이 이미 존재합니다.")
                    self.log.emit(f"[INFO] 불필요한 재학습을 건너뛰고 기존 모델을 유지합니다.")
                else:
                    # 4. YOLO 개별 학습 시작 (가중치가 없을 때만 실행)
                    self.log.emit(f"[YOLO] '{cls}' 모델 학습 시작 (Epoch: 10 / Device: {target_device})...")
                    model = YOLO(str(BASE_MODEL))
                    run_name = f"seg_{cls}"
                    
                    model.train(
                        data=str(cls_yaml),
                        task="segment",
                        epochs=10,
                        imgsz=640,
                        batch=train_batch,      
                        device=target_device,   
                        project=str(WEIGHT_PATH),
                        name=run_name,
                        exist_ok=True,
                        workers=train_workers,  
                        verbose=True,
                        patience=5,
                        cache=False             
                    )

                    # 5. 학습 완료된 가중치를 best_제품명.pt로 복사
                    trained_best = WEIGHT_PATH / run_name / "weights/best.pt"
                    
                    if trained_best.exists():
                        shutil.copy(trained_best, target_best)
                        self.log.emit(f"[SUCCESS] ✅ '{cls}' 모델 저장 완료 -> {target_best.name}")
                    else:
                        self.log.emit(f"[ERROR] ⚠️ '{cls}' 가중치 파일을 찾지 못했습니다.")

                    # 루프 간 메모리 초기화
                    try:
                        del model
                    except NameError:
                        pass
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.log.emit("[MEMORY] 다음 학습을 위해 캐시가 초기화되었습니다.")

            if self._is_running:
                self.done.emit(True, f"선택한 {total_classes}개 제품의 모델 학습이 완료되었습니다!")

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
        self.setWindowTitle("YOLO Segmentation Training (레시피별 모델 학습)")
        self.setFixedSize(1400, 900)
        
        self.get_frame_func = get_frame_func
        self.close_camera_func = close_camera_func
        self.worker = None

        # -----------------------------------------------------
        # 좌측: 템플릿 선택 리스트 패널
        # -----------------------------------------------------
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        lbl_select = QLabel("📦 학습할 템플릿 선택")
        lbl_select.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        left_layout.addWidget(lbl_select)

        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet("""
            QListWidget { background-color: #2b2b2b; color: white; font-size: 16px; border: 1px solid #555; }
            QListWidget::item { padding: 8px; }
            QListWidget::item:hover { background-color: #3d3d3d; }
        """)
        left_layout.addWidget(self.list_widget)

        # 전체 선택 / 해제 버튼
        select_btns = QHBoxLayout()
        btn_select_all = QPushButton("전체 선택")
        btn_deselect_all = QPushButton("전체 해제")
        btn_select_all.setFixedHeight(40)
        btn_deselect_all.setFixedHeight(40)
        btn_select_all.clicked.connect(lambda: self.set_all_checked(True))
        btn_deselect_all.clicked.connect(lambda: self.set_all_checked(False))
        select_btns.addWidget(btn_select_all)
        select_btns.addWidget(btn_deselect_all)
        left_layout.addLayout(select_btns)

        # -----------------------------------------------------
        # 우측: 로그창 및 조작 버튼
        # -----------------------------------------------------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setLineWrapMode(QTextEdit.NoWrap) 
        self.log_view.setStyleSheet("background-color: white; color: black; font-family: Consolas; font-size: 14px;")
        right_layout.addWidget(self.log_view)

        # 하단 버튼
        self.btn_start = QPushButton("선택 항목 학습 시작")
        self.btn_stop = QPushButton("학습 중지")
        self.btn_close = QPushButton("닫기")
        
        self.btn_start.setFixedHeight(60)
        self.btn_stop.setFixedHeight(60)
        self.btn_close.setFixedHeight(60)
        
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-size: 18px; font-weight: bold;")
        self.btn_stop.setStyleSheet("background-color: #E53935; color: white; font-size: 18px; font-weight: bold;")
        self.btn_close.setStyleSheet("background-color: #757575; color: white; font-size: 18px; font-weight: bold;")
        
        self.btn_start.clicked.connect(self.start)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_close.clicked.connect(self.close)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_start)
        btns.addWidget(self.btn_stop)   
        btns.addWidget(self.btn_close)
        right_layout.addLayout(btns)

        # -----------------------------------------------------
        # 전체 레이아웃 조합 (좌측 300px, 우측 나머지)
        # -----------------------------------------------------
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(left_panel, stretch=1)
        main_layout.addWidget(right_panel, stretch=3)

        # 저장된 템플릿 목록 로드
        self.load_template_list()
        
    def load_template_list(self):
        """SHAPE_DB_DIR에서 템플릿 폴더 목록을 읽어와 체크박스 리스트 생성"""
        self.list_widget.clear()
        if not SHAPE_DB_DIR.exists():  #[cite: 7]
            self.append_log("[안내] 템플릿 폴더가 존재하지 않습니다.")
            return

        template_names = sorted([f.name for f in SHAPE_DB_DIR.iterdir() if f.is_dir()])  #[cite: 7]
        if not template_names:
            self.append_log("[안내] 저장된 템플릿이 없습니다. 먼저 템플릿을 생성해주세요.")
            return

        for name in template_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
            
        self.append_log(f"[안내] 총 {len(template_names)}개의 템플릿이 로드되었습니다. 학습할 항목을 체크하세요.")

    def set_all_checked(self, checked: bool):
        state = Qt.Checked if checked else Qt.Unchecked
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setCheckState(state)

    def start(self):
        """학습 시작: 배경 선택(기존/신규) -> 학습 워커 실행"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "경고", "이미 학습 중입니다.")
            return
        
        selected_classes = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected_classes.append(item.text())

        if not selected_classes:
            QMessageBox.warning(self, "선택 오류", "학습할 템플릿을 최소 1개 이상 선택해주세요!")
            return

        self.log_view.clear()
        self.append_log(f"[시작] 선택된 학습 대상: {selected_classes}")
        
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
        self.worker = YoloTrainWorker(selected_classes=selected_classes)
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
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = AutoYoloSegDialog()
    dlg.show()
    sys.exit(app.exec_())