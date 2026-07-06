# aruco_rigid_calibration_qt.py  (✅ aruco_rigid_calibration.py + qt 통합본)
# ✅ 외부에서 받는 건 get_frame_func 하나뿐
# ✅ DB에서 base pose / marker offset 로드 (RobotDB)
# ✅ 로봇 이동 + 샘플 캡처 + Rigid 계산 + 결과 저장까지 내부 처리
# ✅ subprocess 없음
# ✅ 프로젝트 루트 관리는 기존 방식 그대로 유지

import sys
import os
import time
import threading
import yaml
import cv2 as cv
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap


# ======================================================
# 경로 (루트 방식 유지)
# ======================================================
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    CONFIG_DIR = PROJECT_ROOT / "VISION" / "config"
    INTRINSICS_FILE = CONFIG_DIR / "calibration_intrinsics.yaml"
    RESULT_FILE = CONFIG_DIR / "aruco_rigid_result.yaml"
else:
    CONFIG_DIR = PROJECT_ROOT / "PRAG" / "VISION" / "config"
    INTRINSICS_FILE = CONFIG_DIR / "calibration_intrinsics.yaml"
    RESULT_FILE = CONFIG_DIR / "aruco_rigid_result.yaml"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# ======================================================
# 외부 모듈 (기존 코드에서 쓰던 그대로)
# ======================================================
from ROBOT_CONTROL.robot import ROBOT_SEND
from DB.robot_db import RobotDB


# ======================================================
# 설정/상수
# ======================================================
MARKER_ID = 0

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


# ======================================================
# 공용 함수들 (aruco_rigid_calibration.py에서 필요한 것만 정리)
# ======================================================
def load_camera_intrinsics() -> np.ndarray:
    if not INTRINSICS_FILE.exists():
        raise FileNotFoundError(INTRINSICS_FILE)

    with open(INTRINSICS_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return np.array(data["camera_matrix"], dtype=float)


def generate_27_poses(base_pose, dx=0.05, dy=0.05, dz=0.05):
    x0, y0, z0, rx, ry, rz = base_pose
    poses = []
    for k in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for i in [-1, 0, 1]:
                poses.append([x0 + i * dx, y0 + j * dy, z0 + k * dz, rx, ry, rz])
    return poses


def detect_aruco_3d_point(color_bgr, depth, K, depth_scale):
    gray = cv.cvtColor(color_bgr, cv.COLOR_BGR2GRAY)

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return None, None

    for i, mid in enumerate(ids.flatten()):
        if int(mid) != MARKER_ID:
            continue

        pts = corners[i][0]
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        z = float(depth[cy, cx]) * float(depth_scale)
        if z <= 0:
            return None, None

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx0, cy0 = float(K[0, 2]), float(K[1, 2])

        X = (cx - cx0) * z / fx
        Y = (cy - cy0) * z / fy

        return np.array([X, Y, z], dtype=float), corners[i]

    return None, None


def rigid_transform(cam_pts, base_pts):
    cam_pts = np.asarray(cam_pts, dtype=float)
    base_pts = np.asarray(base_pts, dtype=float)

    mu_c = cam_pts.mean(axis=0)
    mu_b = base_pts.mean(axis=0)

    X = cam_pts - mu_c
    Y = base_pts - mu_b

    U, _, Vt = np.linalg.svd(Y.T @ X)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Vt[-1, :] *= -1
        Rm = U @ Vt

    t = mu_b - Rm @ mu_c
    return Rm, t


def save_rigid_result(Rm, t):
    data = {"base_T_cam": {"R": Rm.tolist(), "t": t.tolist()}}
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# ======================================================
# Robot controller (기존 코드 흐름 복원: movel + running 대기)
# ======================================================
class RobotController:
    """
    ⚠️ ROBOT_SEND 객체를 생성하고, 백그라운드에서 poll()을 호출하여
       로봇 상태(running 등)를 지속적으로 최신화함.
    """

    def __init__(self):
        # 1. 로봇 연결
        try:
            self.robot_send = ROBOT_SEND("192.168.2.202")
            # 연결 즉시 한 번 poll 시도
            self.robot_send.poll()
            self.tcp_recv = self.robot_send.get_tcp()
        except TypeError:
            self.robot_send = ROBOT_SEND()
            self.robot_send.poll()
            self.tcp_recv = self.robot_send.get_tcp()

        self.running = None
        
        # 2. [추가] 폴링 스레드 설정 (데이터 갱신용)
        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

    def _poll_loop(self):
        """백그라운드에서 주기적으로 데이터를 갱신"""
        while not self._stop_event.is_set():
            try:
                # 이 함수가 호출되어야 get_state() 값이 변함
                self.robot_send.poll()
            except Exception as e:
                print(f"[RobotController] Poll error: {e}")
            
            # 너무 빠르면 CPU 점유율이 오르므로 약간 대기
            time.sleep(0.005)

    def close(self):
        """스레드 종료 및 자원 정리"""
        self._stop_event.set()
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)

    def wait_until_stop(self, timeout=30):
        start = time.time()
        was_running = False
        
        # 명령을 보낸 직후에는 아직 running이 False일 수 있음.
        # 따라서 "움직이기 시작할 때까지" 기다리는 최대 시간을 둡니다 (예: 3초)
        start_wait_limit = 3.0 
        motion_started = False

        while time.time() - start < timeout:
            try:
                state = self.robot_send.get_state()
                is_running = state.get("running") if state else None
                
                if is_running is None:
                    time.sleep(0.1)
                    continue

                self.running = is_running
                
                # 1. 움직임 시작 감지
                if self.running:
                    motion_started = True
                    was_running = True
                
                # 2. 움직임 시작 후 멈춤 감지 -> 완료
                if was_running and not self.running:
                    return True
                
                # 3. 예외 상황: 명령은 보냈는데 로봇이 꿈쩍도 안 하는 경우 (3초 지남)
                # 이미 목표 위치에 있어서 안 움직인 것일 수도 있음 -> True 반환
                if not motion_started and (time.time() - start > start_wait_limit):
                    print("[WARN] 로봇이 움직이지 않음 (이미 목표 위치거나 명령 무시됨). 다음 단계 진행.")
                    return True

            except Exception as e:
                print(f"상태 확인 중 에러 발생: {e}")
                pass

            time.sleep(0.05) # 주기를 조금 더 짧게 수정

        print("[WARN] Robot wait timeout!")
        return False

    def move_to_pose(self, target_pose6, a=1.0, v=0.4, timeout=30):
        cmd = f"movel({target_pose6}, a={a}, v={v})"
        self.robot_send.send_command(cmd)
        return self.wait_until_stop(timeout=timeout)

# ======================================================
# Worker Thread: 실제 캘리브레이션 수행
# ======================================================
class CalibWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool, str)

    def __init__(self, get_frame_func):
        super().__init__()
        self.get_frame_func = get_frame_func
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        robot = None  # 초기화
        try:
            # 1) DB에서 값 로드
            row_m = RobotDB().fetch_fk_pose("marker")
            marker_from_tcp = np.array([row_m[2] / 1000.0, row_m[3] / 1000.0, row_m[4] / 1000.0], dtype=float)

            row_c = RobotDB().fetch_fk_pose("calibration")
            base_pose = [row_c[2], row_c[3], row_c[4], row_c[5], row_c[6], row_c[7]]

            self.log.emit(f"[DB] marker_from_tcp = {marker_from_tcp.tolist()}")
            self.log.emit(f"[DB] base_pose = {base_pose}")

            # 2) intrinsics 로드
            K = load_camera_intrinsics()
            self.log.emit(f"[INTR] loaded: {INTRINSICS_FILE}")

            # 3) 로봇 컨트롤러 생성 + base pose 이동
            robot = RobotController()  # 여기서 내부 스레드 시작됨
            
            self.log.emit("[ROBOT] move to BASE_POSE ...")
            if not robot.move_to_pose(base_pose):
                self.done.emit(False, "로봇 BASE_POSE 이동 실패")
                return
            self.log.emit("[ROBOT] BASE_POSE reached")

            # 4) 27 pose 생성
            poses = generate_27_poses(base_pose)
            self.log.emit("[SEQ] generated 27 poses")

            cam_points = []
            base_points = []

            # 5) 시퀀스 수행
            for idx, pose in enumerate(poses, start=1):
                if self._stop:
                    self.done.emit(False, "사용자 중지")
                    return

                self.log.emit(f"[{idx}/27] move ...")
                if not robot.move_to_pose(pose):
                    self.log.emit("[WARN] move failed -> skip")
                    continue
                
                time.sleep(0.5)

                # 프레임 가져오기
                result = self.get_frame_func()
                if result is None:
                    self.log.emit("[WARN] frame=None -> skip")
                    continue

                color, depth, depth_scale = result
                if color is None or depth is None:
                    self.log.emit("[WARN] color/depth None -> skip")
                    continue

                cam_xyz, corners = detect_aruco_3d_point(color, depth, K, depth_scale)
                if cam_xyz is None:
                    self.log.emit("[WARN] aruco detect failed -> skip")
                    continue

                flange_xyz = np.array(pose[:3], dtype=float)
                marker_xyz = flange_xyz + marker_from_tcp

                cam_points.append(cam_xyz)
                base_points.append(marker_xyz)

                self.log.emit(f"[OK] cam={cam_xyz.tolist()} base={marker_xyz.tolist()}")

            if len(cam_points) < 6:
                self.done.emit(False, f"샘플 부족: {len(cam_points)}개 (최소 6개 필요)")
                return

            # 6) 계산 + 저장
            Rm, t = rigid_transform(cam_points, base_points)
            save_rigid_result(Rm, t)

            self.log.emit("[DONE] rigid saved")
            self.log.emit(f"R=\n{np.array(Rm)}")
            self.log.emit(f"t={np.array(t).tolist()}")
            self.done.emit(True, f"완료. 저장: {RESULT_FILE}")

        except Exception as e:
            self.done.emit(False, f"예외: {e}")
        
        finally:
            # [중요] 작업이 끝나거나 에러가 나도 로봇 폴링 스레드는 정리해야 함
            if robot:
                robot.close()


# ======================================================
# Qt Dialog: 시작/중지/로그 + (선택) 라이브뷰
# ======================================================
class ArucoCalibrationDialog(QDialog):
    """
    ✅ 외부에서 넘기는 것은 get_frame_func 하나만
      - get_frame_func() -> (color_bgr, depth, depth_scale)
    """

    def __init__(self, get_frame_func, parent=None):
        super().__init__(parent)

        self.get_frame_func = get_frame_func
        self.worker = None

        self.setWindowTitle("ArUco Rigid Calibration")
        self.setFixedSize(1100, 720)

        self._init_ui()

        # 라이브 뷰(동작 확인용) - 프레임만 사용, 로봇/DB 접근 없음
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_view)
        self.timer.start(30)

    def _init_ui(self):
        root = QHBoxLayout(self)

        # left: view
        left = QVBoxLayout()
        self.image = QLabel()
        self.image.setFixedSize(800, 600)
        self.image.setStyleSheet("background:black;")
        self.image.setAlignment(Qt.AlignCenter)
        left.addWidget(self.image)

        # right: buttons + log
        right = QVBoxLayout()

        self.btn_start = QPushButton("작업 시작")
        self.btn_stop = QPushButton("작업 중지")
        self.btn_close = QPushButton("닫기")

        for b in (self.btn_start, self.btn_stop, self.btn_close):
            b.setStyleSheet(BTN_STYLE)
            b.setFixedSize(250, 60)

        self.btn_start.clicked.connect(self.start_calibration)
        self.btn_stop.clicked.connect(self.stop_calibration)
        self.btn_close.clicked.connect(self.close)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("""
            QTextEdit {
                background-color: white;
                color: black;
                font-family: Consolas;
                font-size: 11pt;
                border: 1px solid #ccc;
            }
        """)

        right.addWidget(self.btn_start)
        right.addWidget(self.btn_stop)
        right.addWidget(self.btn_close)
        right.addSpacing(10)
        right.addWidget(self.log_box, 1)

        root.addLayout(left, 0)
        root.addLayout(right, 1)

    def append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())

    def update_view(self):
        # 캘리브레이션 동작 여부와 상관없이 "프레임만" 보여줌
        result = self.get_frame_func()
        if result is None:
            return

        color, depth, depth_scale = result
        if color is None:
            return

        # intrinsics 파일 없으면 그냥 화면만 표시
        vis = color.copy()
        try:
            K = load_camera_intrinsics()
            cam_xyz, corners = detect_aruco_3d_point(color, depth, K, depth_scale)
            if corners is not None:
                cv.aruco.drawDetectedMarkers(vis, [corners])
        except Exception:
            pass

        rgb = cv.cvtColor(vis, cv.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888)
        self.image.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def start_calibration(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "경고", "이미 실행 중입니다.")
            return

        self.log_box.clear()
        self.append_log("[QT] Calibration start")

        self.worker = CalibWorker(self.get_frame_func)
        self.worker.log.connect(self.append_log)
        self.worker.done.connect(self.on_done)
        self.worker.start()

    def stop_calibration(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.append_log("[QT] Stop requested")

    def on_done(self, ok: bool, msg: str):
        self.append_log(f"[QT] Done: {ok} / {msg}")
        self.worker = None
        if ok:
            QMessageBox.information(self, "완료", msg)
        else:
            QMessageBox.warning(self, "실패", msg)

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()
