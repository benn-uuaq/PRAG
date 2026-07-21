import sys
import ast
import time
import math
import socket
import select 
import traceback
import threading
import numpy as np
from pathlib import Path

from PyQt5.QtWidgets import *
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QMetaObject, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

PC_IP = "192.168.3.30"
ROBOT_IP = "192.168.2.202"
IO_IP = "192.168.1.150"
IO_PORT = 502

Z_LIMIT = 0.055 # 단위: m

# =============================
# Motion Speed Profile
# =============================
SPEED_FAST   = 5.0
SPEED_FAST_2ND   = 3.5
SPEED_NORMAL = 1.2
SPEED_SLOW   = 0.4
SPEED_PICK   = 0.1

ACC_FAST   = 5.0
ACC_FAST_2ND   = 3.5
ACC_NORMAL = 1.5
ACC_SLOW   = 0.6

SPEED_FAST_J  = 7.0
SPEED_NORMAL_J = 2.0
SPEED_SLOW_J   = 0.6
SPEED_PICK_J   = 0.2

ACC_FAST_J   = 5.0
ACC_NORMAL_J = 2.0
ACC_SLOW_J   = 1.0

def get_app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent

APP_ROOT = get_app_root()

# ⭐ sys.path 조작은 여기 한 번만
sys.path.insert(0, str(APP_ROOT))

from UI.ppap_ui_20251219_ui import Ui_MainWindow
from ROBOT_CONTROL.robot import Robot, ROBOT_SEND
from VISION.vision_main import VisionMain
from DB.robot_db import RobotDB
from IO.IOmodule import IO_Module_Class

class DBManager(RobotDB):
    def __init__(self):
        super().__init__()

class EmittingStream:
    def __init__(self, append_callback):
        self.append_callback = append_callback

    def write(self, text):
        if text.strip():
            self.append_callback(text)

    def flush(self):
        pass

class NumericKeypadDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("입력")
        self.setFixedSize(300, 500)
        self.value = ""

        # 🔹 부모 스타일 무시하고 기본 Qt 스타일 사용
        self.setStyleSheet("background-color: white; color: black;")

        layout = QVBoxLayout(self)
        self.display = QLineEdit(self)
        self.display.setAlignment(Qt.AlignRight)
        self.display.setReadOnly(True)
        self.display.setFixedHeight(50)
        self.display.setStyleSheet("font-size: 20px; background-color: white; color: black; border: 1px solid gray;")
        layout.addWidget(self.display)

        grid = QGridLayout()
        buttons = [
            ('7', 0, 0), ('8', 0, 1), ('9', 0, 2),
            ('4', 1, 0), ('5', 1, 1), ('6', 1, 2),
            ('1', 2, 0), ('2', 2, 1), ('3', 2, 2),
            ('0', 3, 1), ('-', 3, 0), ('.', 3, 2)
        ]
        for text, row, col in buttons:
            btn = QPushButton(text)
            btn.setFixedSize(80, 60)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0f0f0;
                    color: black;
                    font-size: 18px;
                    border-radius: 8px;
                    border: 1px solid gray;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
            """)
            btn.clicked.connect(lambda _, t=text: self.button_clicked(t))
            grid.addWidget(btn, row, col)
            
        layout.addLayout(grid)

        # 하단 OK/Cancel 버튼
        delete_btn = QPushButton("C")
        ok_btn = QPushButton("확인")
        cancel_btn = QPushButton("취소")
        delete_btn.setStyleSheet("background-color: #0000FF; color: white; font-size: 16px; border-radius: 8px; height: 40px;")
        ok_btn.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; border-radius: 8px; height: 40px;")
        cancel_btn.setStyleSheet("background-color: #E53935; color: white; font-size: 16px; border-radius: 8px; height: 40px;")
        delete_btn.clicked.connect(lambda: self.button_clicked('C'))
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        layout.addWidget(delete_btn)
        layout.addWidget(ok_btn)
        layout.addWidget(cancel_btn)
        
    def button_clicked(self, text):
        if text == 'C':
            self.value = ""
        else:
            self.value += text
        self.display.setText(self.value)

    def get_value(self):
        return self.value
    
class JobThread(QThread):
    def __init__(self, ui_ref):
        super().__init__()
        self.ui = ui_ref

    def run(self):
        print("[INFO] 로봇 작업 스레드 시작.")
        try:
            self.ui.job() 
        except Exception as e:
            print(f"[ERROR] 로봇 작업 스레드 에러: {e}")
        finally:
            print("[INFO] 로봇 작업 스레드 완료.")

    def stop(self):
        pass

class RobotStatusThread(QThread):
    status_signal = pyqtSignal(tuple)
    disconnected_signal = pyqtSignal()

    def __init__(self, robot_send):
        super().__init__()
        self.robot_send = robot_send
        self.running = True
        self.disconnected_emitted = False

    def check_socket_alive(self):
        """소켓 연결 상태 확인"""
        try:
            sock = self.robot_send.robot_30001.client_socket
        except AttributeError:
            return False
        
        if sock is None or sock.fileno() == -1:
            return False

        try:
            r, w, e = select.select([sock], [], [sock], 0)
            if e: return False
            if r:
                try:
                    data = sock.recv(1, socket.MSG_PEEK)
                    if len(data) == 0: return False
                except (BlockingIOError, OSError): return True
                except Exception: return False
            return True
        except Exception:
            return False

    def run(self):
        DISCONNECT_TIMEOUT = 10.0
        last_recv_time = time.time()
        
        # [핵심] 마지막으로 UI에 보낸 상태를 기억하는 변수
        self.last_emitted_state = None

        while self.running:
            try:
                # 1. 소켓 객체 가져오기
                sock = None
                if self.robot_send.robot_30001:
                    sock = self.robot_send.robot_30001.client_socket
                
                # 연결 없으면 잠시 대기
                if sock is None or sock.fileno() == -1:
                    self.msleep(100)
                    continue

                # 2. 데이터 대기 (최대 0.01초)
                # msleep 대신 select를 써서, 데이터가 오면 0.001초 만에도 즉시 반응함
                r, _, _ = select.select([sock], [], [], 0.01)

                if r:
                    # 3. 데이터 읽기 & 메모리 갱신 (Real-time)
                    # 여기서 poll()을 자주 해줘야 JobThread가 최신 정보를 가져갑니다.
                    if self.robot_send.poll():
                        last_recv_time = time.time()
                        
                        # 4. 현재 상태값 추출
                        state = self.robot_send.get_state()
                        mode = state.get("mode", 0)
                        
                        # 비교를 위한 튜플 생성
                        current_state = (
                            state.get("power", 0),
                            state.get("running", False),
                            state.get("speed", 0.0),
                            state.get("alarm", []),
                            mode,
                            state.get("control_mode", 0),
                        )
                        
                        # 5. [핵심] 값이 변했을 때만 UI 업데이트 (Event-driven)
                        # 이전 값과 다를 때만 emit 하므로 UI 렉이 획기적으로 줄어듭니다.
                        if current_state != self.last_emitted_state:
                            
                            self.status_signal.emit(current_state)
                            self.last_emitted_state = current_state
                            
                            # 연결 끊김 복구
                            if self.disconnected_emitted:
                                self.disconnected_emitted = False
                
            except Exception as e:
                pass

            # 타임아웃 체크 (데이터가 10초 이상 안 올 때)
            if time.time() - last_recv_time > DISCONNECT_TIMEOUT:
                if not self.check_socket_alive():
                    if not self.disconnected_emitted:
                        print(f"[WARN] Robot disconnected (Timeout).")
                        self.disconnected_signal.emit()
                        self.disconnected_emitted = True
                        self.last_emitted_state = None # 연결 끊기면 상태 초기화
                else:
                    last_recv_time = time.time()

class PPAPUI(QMainWindow):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super(PPAPUI, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.stack_window.setCurrentWidget(self.ui.main_page)
        
        self.db_manager = DBManager()
        
        # ✅ DB에서 IP 먼저 로드
        self.robot_ip = ROBOT_IP
        self.pc_ip = PC_IP
        print("[DEBUG] robot_ip =", self.robot_ip)
        print("[DEBUG] pc_ip =", self.pc_ip)
        
        self.io = IO_Module_Class(IO_IP, IO_PORT)
        self.io.start()
        print("[DEBUG] io_ip =", IO_IP)
        
        self.io.Unit_Reset("OFF")

        self.load_vision_from_db()  
        self.load_pose_from_db()
        
        self.ui.vision_reliability_setting_button.clicked.connect(
            lambda: self.on_vision_save("vision_reliability", [
                self.ui.vision_reliability_setting_edit_2])
        )
        
        self.ui.hoe_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("home", [
                self.ui.home_pose_X, self.ui.home_pose_Y, self.ui.home_pose_Z,
                self.ui.home_pose_Rx, self.ui.home_pose_Ry, self.ui.home_pose_Rz
            ])
        )

        self.ui.pick_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("pick", [
                self.ui.pick_pose_X, self.ui.pick_pose_Y, self.ui.pick_pose_Z,
                self.ui.pick_pose_Rx, self.ui.pick_pose_Ry, self.ui.pick_pose_Rz
            ])
        )

        self.ui.place_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("place", [
                self.ui.place_pose_X, self.ui.place_pose_Y, self.ui.place_pose_Z,
                self.ui.place_pose_Rx, self.ui.place_pose_Ry, self.ui.place_pose_Rz
            ])
        )

        self.ui.tcp_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("tcp", [
                self.ui.tcp_pose_X, self.ui.tcp_pose_Y, self.ui.tcp_pose_Z,
                self.ui.tcp_pose_Rx, self.ui.tcp_pose_Ry, self.ui.tcp_pose_Rz
            ])
        )
        
        self.ui.marker_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("marker", [
                self.ui.marker_pose_X, self.ui.marker_pose_Y, self.ui.marker_pose_Z,
                self.ui.marker_pose_Rx, self.ui.marker_pose_Ry, self.ui.marker_pose_Rz
            ])
        )
        
        self.ui.calibration_position_setting_button.clicked.connect(
            lambda: self.on_pose_save("calibration", [
                self.ui.calibration_pose_X, self.ui.calibration_pose_Y, self.ui.calibration_pose_Z,
                self.ui.calibration_pose_Rx, self.ui.calibration_pose_Ry, self.ui.calibration_pose_Rz
            ])
        )

        self.connect_keypad_events()
        
        self.pose_cache = {}
        retry_timeout = 10.0
        start_time = time.time()
        print("[INFO] DB 포즈 데이터 로딩 시도 중...")
        while time.time() - start_time < retry_timeout:
            self.load_pose_cache_from_db()
            if self.pose_cache:
                print(f"[INFO] Pose cache preloaded successfully: {list(self.pose_cache.keys())}")
                break
            time.sleep(0.1)
        if not self.pose_cache:
            print(f"[WARN] {retry_timeout}초 대기 후에도 DB 데이터를 찾지 못했습니다.")
            print("[WARN] 잠시 후 자동 계산 루틴(initial_pose_calibration)을 예약합니다.")
            QTimer.singleShot(2000, self.update_all_fk_poses)
        
        self.is_fk_calculating = False

        self.job_thread = None
        
        self.is_auto_running = False
        
        self.power = False
        self.running = False
        self.speed = 0
        self.alarm = False
        self.mode = 3         
        self.control_mode = 0
        self._speed_initialized = False
        
        self.alarm_popup_shown = False

        self.robot_status_initialized = False
        
        self.shaking_time = False
        self.shake_end_time = 0.0
        
        # self.robot_lock = threading.Lock()
                
        self.robot_send = ROBOT_SEND(self.robot_ip)
        
        self.ui.speed_slider.valueChanged.connect(self.on_slider_changed)
        self.on_slider_changed(self.ui.speed_slider.value())
        self.ui.speed_slider.valueChanged.connect(self.on_speed_changed)

        sys.stdout = EmittingStream(self.log_signal.emit)
        sys.stderr = EmittingStream(self.log_signal.emit)
        
        self.log_signal.connect(self.add_log)
        self.ui.log_delete_button.clicked.connect(lambda: self.ui.scrollArea.clear())
        
        self.ui.logo.installEventFilter(self)
        
        try:
            self.ui.start_button.clicked.disconnect()
            self.ui.robot_home_button.clicked.disconnect()
            # self.ui.pause_button.clicked.disconnect()
            self.ui.stop_button.clicked.disconnect()
        except Exception:
            pass
        self.ui.start_button.clicked.connect(self.on_start_button_clicked)
        self.ui.pause_button.toggled.connect(self.on_pause_button_toggled)
        self.ui.stop_button.clicked.connect(self.on_stop_button_clicked)
        self.ui.power_on_button.clicked.connect(lambda: self.on_robot_power_on_button_clicked(True))
        self.ui.power_off_button.clicked.connect(lambda: self.on_robot_power_off_button_clicked(True))
        self.ui.robot_home_button.clicked.connect(self.on_robot_home_button_clicked)
        
        self.setup_vision_label()
        self._last_frame = None
        
        #비전 버튼
        self.ui.vision_on_button.clicked.connect(self.vision_on)
        self.ui.vision_off_button.clicked.connect(self.vision_off)
        self.ui.vision_start_button.clicked.connect(self.vision_start)
        self.ui.vision_stop_button.clicked.connect(self.vision_stop)
        #비전 세팅 버튼
        self.ui.vision_ROI_setting_button.clicked.connect(self.on_roi_setting)
        self.ui.vision_cal_start_button.clicked.connect(self.on_camera_calib)
        self.ui.vision_robot_cal_start_button.clicked.connect(self.on_handeye_calib)
        self.ui.template_set_button.clicked.connect(self.on_template_set)
        self.ui.ai_train_start_button.clicked.connect(self.on_yolo_train)
        
        self.vision = VisionMain(APP_ROOT, parent=self)
        self.vision.frame_signal.connect(self.update_vision_label)
        
        self.ui_vision_enabled = False
        
        self.vision_pick_cache = {"pose": None, "valid": False, "timestamp": 0.0}
        
        
        #IO 버튼
        self.ui.box_shake_on.clicked.connect(lambda: self.io.BIN_Cylinder("ON"))
        self.ui.box_shake_off.clicked.connect(lambda: self.io.BIN_Cylinder("OFF"))
        self.ui.vacuum_on.clicked.connect(lambda: self.io.Vacuum("ON"))
        self.ui.vacuum_off.clicked.connect(lambda: self.io.Vacuum("OFF"))
        self.ui.blow_on.clicked.connect(lambda: self.io.Blow("ON"))
        self.ui.blow_off.clicked.connect(lambda: self.io.Blow("OFF"))
        self.ui.buzzer_off_button.clicked.connect(self.buzzer_off)
        self.ui.robot_alarm_reset_button.clicked.connect(self.robot_alarm_reset_button)
        
        self.robot_disconnected = False
        
        self.status_thread = RobotStatusThread(self.robot_send)
        self.status_thread.status_signal.connect(self.update_robot_status)
        self.status_thread.disconnected_signal.connect(self.on_robot_disconnected)
        self.status_thread.start()
        
        self.buzzer_value = False
        
        self.robot_send.robot_30001.on_disconnect = lambda: QMetaObject.invokeMethod(
            self, "on_robot_disconnected", Qt.QueuedConnection
        )
        
        self.robot_status_update()
        self.vision.start_camera()
        
    def robot_alarm_reset_button(self):
        try:
            self.buzzer_value = False
            self.io.Unit_Reset("ON")
            self.robot_send.robot_29999.send_command_29999("safety -r")
            self.robot_send.robot_29999.send_command_29999("closeSafetyDialog")
            time.sleep(0.1)
            self.io.Unit_Reset("OFF")
        except Exception as e:
            print("[Robot Alarm Reset ERROR]", e) 
        
    def buzzer_off(self):
        try:
            self.buzzer_value = False
            self.io.Buzzer("OFF")
        except Exception as e:
            print("[Robot Alarm Reset ERROR]", e) 
        
    def disable_all_buttons(self):
        buttons = [
            self.ui.start_button,
            self.ui.pause_button,
            self.ui.stop_button,
            self.ui.robot_home_button
        ]
        for btn in buttons:
            btn.setEnabled(False)


    def enable_buttons_after_alarm(self):
        if self.power:
            self.button_enable()
        else:
            self.button_disable()

    @pyqtSlot()
    def show_alarm_popup(self):
        QMessageBox.critical(
            self,
            "로봇 알람 발생",
            """
            <div style="color:white; font-size:22px; padding:20px;">
                <b>알람 발생</b><br><br>
                로봇 티치펜던트를 확인해주세요.
            </div>
            """
        )  
        
    def on_roi_setting(self):
        self.vision_off()
        self.vision.select_roi()

    def on_camera_calib(self):
        self.vision.calibrate_camera_intrinsics()

    def on_handeye_calib(self):
        # self.vision_off()
        self.vision.calibrate_handeye()
        
    def on_template_set(self):
        self.vision.capture_and_generate_templates()

    def on_yolo_train(self):
        self.vision.train_yolo_segmentation()
        
    def setup_vision_label(self):
        """비전 라벨 크기 자동 조절 설정"""
        from PyQt5.QtWidgets import QSizePolicy, QVBoxLayout
        
        if not self.ui.video_container.layout():
            container_layout = QVBoxLayout(self.ui.video_container)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            
            container_layout.addWidget(self.ui.label_video)
        
        self.ui.video_container.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        
        self.ui.view_frame.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        
        self.ui.label_video.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding
        )
        
        self.ui.label_video.setMinimumSize(320, 240)
        
        self.ui.label_video.setAlignment(Qt.AlignCenter)
        
        self.ui.label_video.setStyleSheet("""
            QLabel {
                background-color: #000000;
                border: 1px solid #555555;
                color: #666666;
            }
        """)
        
        self.ui.label_video.setText("카메라 OFF")
        
        print("[UI] 비전 라벨 설정 완료")

    def resizeEvent(self, event):
        """윈도우 크기 변경 시 비전 화면도 자동 조절"""
        super().resizeEvent(event)
        
        if hasattr(self, '_last_frame') and self._last_frame is not None:
            try:
                rgb = self._last_frame[:, :, ::-1]
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
                
                pixmap = QPixmap.fromImage(qimg).scaled(
                    self.ui.label_video.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.ui.label_video.setPixmap(pixmap)
            except:
                pass
            
    def vision_off(self):
        """UI 화면 OFF (카메라 유지, 객체 탐지 X)"""
        print("[UI] vision_off")

        self.ui_vision_enabled = False
        self.vision.set_visualize(False)
        self.vision.frame_signal.disconnect(self.update_vision_label)

        self._last_frame = None
        self.ui.label_video.clear()
        self.ui.label_video.setPixmap(QPixmap())
        self.ui.label_video.setText("카메라 OFF")

    def vision_on(self):
        """UI 화면 ON (객체 탐지 X)"""
        print("[UI] vision_on")

        self.ui_vision_enabled = True

        if not self.vision.worker.isRunning():
            self.vision.start_camera()
            
        self.vision.frame_signal.connect(self.update_vision_label)

        self.vision.set_visualize(False)

    def vision_start(self):
        """객체 탐지 시작 + 화면 표시"""
        print("[UI] vision_start")

        self.ui_vision_enabled = True

        if not self.vision.worker.isRunning():
            self.vision.start_camera()

        self.vision.set_visualize(True)
        
    def vision_stop(self):
        """객체 탐지 중지 (화면 유지)"""
        print("[UI] vision_stop")

        self.vision.set_visualize(False)

    def detect_object_and_get_pose(self):
            """비전 시스템을 통해 물체를 감지하고 픽 위치를 반환"""
            try:
                success, base_xyz, result, img = self.vision.detect_pick_once()
                
                if success and base_xyz is not None:
                    print(f"[INFO] Pick point detected: {np.round(base_xyz, 6)}")

                    pose = [
                        float(base_xyz[0]), 
                        float(base_xyz[1]), 
                        float(base_xyz[2]),
                        -3.14159265358979, 0.0, -1.5707963267949
                    ]
                    
                    self.vision_pick_cache["pose"] = pose
                    self.vision_pick_cache["valid"] = True
                    self.vision_pick_cache["timestamp"] = time.time()
                    
                    return pose
                else:
                    print("[WARN] No pick point detected by vision.")
                    return None
                    
            except Exception as e:
                print(f"[ERROR] Vision detection failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    def update_vision_label(self, frame: np.ndarray):
        if not self.ui_vision_enabled:
            return
        """UI에 프레임 표시"""
        try:
            self._last_frame = frame.copy()
            
            rgb = frame[:, :, ::-1]
            h, w, ch = rgb.shape
            qimg = QImage(rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888)

            pixmap = QPixmap.fromImage(qimg).scaled(
                self.ui.label_video.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.ui.label_video.setPixmap(pixmap)

        except Exception as e:
            print("[VISION UI ERROR]", e) 
        
    @pyqtSlot()
    def on_robot_disconnected(self):
        self.robot_disconnected = True
        
        # [핵심 수정 3] 연결 끊김 시 자동 운전 강제 종료 플래그 설정
        if self.is_auto_running:
            print("[WARN] 연결 끊김 감지! 작업을 강제 중단합니다.")
            self.is_auto_running = False  # 작업 루프 탈출용
            self.robot_send.robot_stop()  # 로봇 정지 명령 시도
        
        try:
            self.ui.scrollArea.append("[WARN] 로봇 연결 끊김")
        except:
            pass  
        self.robot_status_update()
    
    def get_forward_kinematic(self, joint_pose):
        try:
            pc_ip = PC_IP
            robot_ip = ROBOT_IP
            pc_port = 30010
            robot = Robot(ip=robot_ip)

            recv_thread = threading.Thread(target=robot.RecvPopup, args=(pc_port,), daemon=True)
            recv_thread.start()
            time.sleep(0.5)

            robot.get_forward_kinematic(joint_pose, pc_ip=pc_ip, pc_port=pc_port)
            print(f"📤 정기구학 계산 명령을 로봇에게 전달 중...")

            start_time = time.time()
            while time.time() - start_time < 5:
                if robot.recv_data:
                    print(f"✅ 정기구학 계산 결과: {robot.recv_data}")
                    pose_list = ast.literal_eval(robot.recv_data)
                    if isinstance(pose_list, list) and len(pose_list) == 6:
                        return [float(x) for x in pose_list]
                time.sleep(0.2)

            print("⚠️ 정기구학 데이터가 수신되지 않았습니다. (timeout)")
            return None

        except Exception as e:
            print(f"[ERROR] 정기구학 계산 실패: {e}")
            return None
        
    def get_inverse_kinematic(self, tcp_pose):
        """로봇에 역방향 기구학 요청 (tcp -> joint)"""
        try:
            pc_ip = PC_IP
            robot_ip = ROBOT_IP
            pc_port = 30010

            robot = Robot(ip=robot_ip)
           
            recv_thread = threading.Thread(target=robot.RecvPopup, args=(pc_port,), daemon=True)
            recv_thread.start()
            time.sleep(0.5)

            robot.get_inverse_kinematic(tcp_pose, pc_ip=pc_ip, pc_port=pc_port)
            print(f"📤 역기구학 계산 명령을 로봇에게 전달 중...")

            start_time = time.time()
            while time.time() - start_time < 5:
                if robot.recv_data:
                    print(f"✅ 역기구학 계산 결과: {robot.recv_data}")
                    pose_list = ast.literal_eval(robot.recv_data)
                    if isinstance(pose_list, list) and len(pose_list) == 6:
                        return [float(x) for x in pose_list]
                time.sleep(0.2)

            print("⚠️ 역기구학 데이터가 수신되지 않았습니다. (timeout)")
            return None

        except Exception as e:
            print(f"[ERROR] 역기구학 계산 실패: {e}")
            return None
            
    def deg_to_rad(self, degrees):
        return [math.radians(deg) for deg in degrees] 
        
    def connect_keypad_events(self):
        # Pose 입력창들 리스트
        all_fields = [
            self.ui.vision_reliability_setting_edit_2,
            self.ui.home_pose_X, self.ui.home_pose_Y, self.ui.home_pose_Z,
            self.ui.home_pose_Rx, self.ui.home_pose_Ry, self.ui.home_pose_Rz,
            self.ui.pick_pose_X, self.ui.pick_pose_Y, self.ui.pick_pose_Z,
            self.ui.pick_pose_Rx, self.ui.pick_pose_Ry, self.ui.pick_pose_Rz,
            self.ui.place_pose_X, self.ui.place_pose_Y, self.ui.place_pose_Z,
            self.ui.place_pose_Rx, self.ui.place_pose_Ry, self.ui.place_pose_Rz,
            self.ui.tcp_pose_X, self.ui.tcp_pose_Y, self.ui.tcp_pose_Z,
            self.ui.tcp_pose_Rx, self.ui.tcp_pose_Ry, self.ui.tcp_pose_Rz,
            self.ui.marker_pose_X, self.ui.marker_pose_Y, self.ui.marker_pose_Z,
            self.ui.marker_pose_Rx, self.ui.marker_pose_Ry, self.ui.marker_pose_Rz,
            self.ui.calibration_pose_X, self.ui.calibration_pose_Y, self.ui.calibration_pose_Z,
            self.ui.calibration_pose_Rx, self.ui.calibration_pose_Ry, self.ui.calibration_pose_Rz
        ]

        # 각 입력창마다 클릭 시 show_keypad 실행
        for field in all_fields:
            field.mousePressEvent = lambda event, f=field: self.show_keypad(event, f)

    def show_keypad(self, event, field):
        """해당 입력창 클릭 시 숫자 키패드 다이얼로그 표시"""
        dialog = NumericKeypadDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            value = dialog.get_value()
            field.setText(value)
            
    def load_vision_from_db(self):
        row = self.db_manager.fetch_vision("vision_reliability")
        if row:
            _, _, value = row
            self.ui.vision_reliability_setting_edit_2.setText(str(value))
        
    def load_pose_from_db(self):
        for name, fields in {
            "home": [self.ui.home_pose_X, self.ui.home_pose_Y, self.ui.home_pose_Z,
                    self.ui.home_pose_Rx, self.ui.home_pose_Ry, self.ui.home_pose_Rz],
            "pick": [self.ui.pick_pose_X, self.ui.pick_pose_Y, self.ui.pick_pose_Z,
                    self.ui.pick_pose_Rx, self.ui.pick_pose_Ry, self.ui.pick_pose_Rz],
            "place": [self.ui.place_pose_X, self.ui.place_pose_Y, self.ui.place_pose_Z,
                    self.ui.place_pose_Rx, self.ui.place_pose_Ry, self.ui.place_pose_Rz],
            "tcp": [self.ui.tcp_pose_X, self.ui.tcp_pose_Y, self.ui.tcp_pose_Z,
                    self.ui.tcp_pose_Rx, self.ui.tcp_pose_Ry, self.ui.tcp_pose_Rz],
            "marker": [self.ui.marker_pose_X, self.ui.marker_pose_Y, self.ui.marker_pose_Z,
                    self.ui.marker_pose_Rx, self.ui.marker_pose_Ry, self.ui.marker_pose_Rz],
            "calibration": [self.ui.calibration_pose_X, self.ui.calibration_pose_Y, self.ui.calibration_pose_Z,
                    self.ui.calibration_pose_Rx, self.ui.calibration_pose_Ry, self.ui.calibration_pose_Rz]
        }.items():
            row = self.db_manager.fetch_pose(name)
            if row:
                _, _, X, Y, Z, Rx, Ry, Rz = row
                values = [X, Y, Z, Rx, Ry, Rz]
                for field, value in zip(fields, values):
                    field.setText(str(value))
                    
    def on_pose_save(self, pose_name, fields):
        allowed_names = ["home", "pick", "place", "tcp", "marker", "calibration"]
        try:
            if pose_name not in allowed_names:
                raise ValueError(f"Invalid pose name: {pose_name}")
            
            pose_values = [float(field.toPlainText() or 0) for field in fields]
            self.db_manager.insert_pose(pose_name, pose_values)
            
            print(f"[DB] {pose_name} pose saved:", pose_values)
            if pose_name in ["home", "pick", "place", "tcp", "marker", "calibration"]:
                self.update_all_fk_poses()
            self.show_message(
                title="저장 완료",
                message=f"{pose_name} 이(가) DB에 저장되었습니다.",
                text_color="#FFFFFF",
                font_size=20
            )
        except Exception as e:
            print(f"[ERROR] {pose_name} 저장 실패:", e)
            self.show_message(
                title="오류",
                message=f"{pose_name} 저장 실패",
                icon=QMessageBox.Warning,
                text_color="#FFFFFF",
                font_size=20
            )
                        
    def on_vision_save(self, setting_name, fields):
        try:
            field = fields[0]

            if isinstance(field, QTextEdit):
                raw = field.toPlainText().strip()
            elif isinstance(field, QLineEdit):
                raw = field.text().strip()
            else:
                raise TypeError(f"Unsupported widget type: {type(field)}")

            value = float(raw) if raw else 0.0

            self.db_manager.insert_vision(setting_name, value)
            print(f"[DB] {setting_name} saved:", value)

            self.show_message(
                title="저장 완료",
                message=f"{setting_name} 이(가) DB에 저장되었습니다.",
                text_color="#FFFFFF",
                font_size=20
            )

        except Exception as e:
            print(f"[ERROR] {setting_name} 저장 실패:", e)
            self.show_message(
                title="오류",
                message=str(e),
                icon=QMessageBox.Warning,
                text_color="#FFFFFF",
                font_size=20
            )

    def show_message(self, title, message, icon=QMessageBox.Information,
        text_color="#FFFFFF",     # ← 글씨 색상
        font_size=22,             # ← 글씨 크기
        button_color="#0078D7",   # ← 버튼 색상
        button_text_color="#FFFFFF"):
        msg = QMessageBox(self)
        msg.setIcon(icon)
        msg.setWindowTitle(title)

        # ✅ 텍스트 설정 (HTML 사용)
        msg.setTextFormat(Qt.RichText)
        msg.setText(
            f"""
            <div style="
                color:{text_color};
                font-size:{font_size}px;
                padding:10px;
            ">
                {message}
            </div>
            """
        )

        msg.setStandardButtons(QMessageBox.Ok)
        ok_btn = msg.button(QMessageBox.Ok)
        ok_btn.setText("확인")

        # ✅ 버튼 스타일 강제 지정
        ok_btn.setFixedSize(140, 60)
        ok_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {button_color};
                color: {button_text_color};
                font-size: {font_size - 2}px;
                padding: 8px 28px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: #005a9e;
            }}
        """)

        # ✅ 메시지 박스 자체 스타일 (배경 포함)
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: #1e1e1e;
            }}
        """)

        msg.resize(1200, 700)
        msg.exec_()

        
    def eventFilter(self, source, event):
        if source == self.ui.logo and event.type() == QtCore.QEvent.MouseButtonPress:
            self.toggle_page()
            return True
        return super().eventFilter(source, event)
        
    def toggle_page(self):
        cur = self.ui.stack_window.currentWidget()
        next_w = self.ui.set_page if cur is self.ui.main_page else self.ui.main_page
        self.ui.stack_window.setCurrentWidget(next_w)
        
    def load_pose_cache_from_db(self):
        """DB에 저장된 포즈를 pose_cache에 로드"""
        try:
            fk_pose_names = ["home", "pick", "place", "pick_above", "place_above", "place_above_j", "calibration", "tcp", "marker"]
            for fk_name in fk_pose_names:
                fk_row = self.db_manager.fetch_fk_pose(fk_name)
                if fk_row:
                    self.pose_cache[fk_name] = [float(x) for x in fk_row[2:]]
            if self.pose_cache:
                print(f"[INFO] Pose cache loaded from DB: {list(self.pose_cache.keys())}")
            else:
                print("[WARN] No FK pose data found in DB.")
        except Exception as e:
            print(f"[ERROR] load_pose_cache_from_db failed: {e}")
    
    def update_all_fk_poses(self):
        """모든 포즈를 DB에서 불러와 FK 계산 후 fk_poses 테이블에 저장"""
        try:
            self.is_fk_calculating = True
            self.ui.status_circle.setStyleSheet("border-radius: 30px; background-color: yellow;")
            self.ui.status_label.setText("로봇 좌표 계산 및 저장 중...")

            QApplication.processEvents()
            print("[INFO] Updating FK poses...")

            # --- DB에서 조인트(degree) 및 포즈(mm/degree)읽기 ---
            calling_home = self.db_manager.fetch_pose("home")
            calling_pick = self.db_manager.fetch_pose("pick")
            calling_place = self.db_manager.fetch_pose("place")
            calling_tcp = self.db_manager.fetch_pose("tcp")
            calling_marker = self.db_manager.fetch_pose("marker")
            calling_calibration = self.db_manager.fetch_pose("calibration")

            if not all([calling_home, calling_pick, calling_place, calling_tcp, calling_marker, calling_calibration]):
                print("[WARN] Some base pose not found in DB.")
                return

            # --- FK 변환 ---
            home_pose_j   = self.deg_to_rad(calling_home[2:])
            pick_pose_j = self.deg_to_rad(calling_pick[2:])
            place_pose_j  = self.deg_to_rad(calling_place[2:])
            calibration_pose_j = self.deg_to_rad(calling_calibration[2:])

            self.pose_cache["home"]   = self.get_forward_kinematic(home_pose_j)
            print("홈 포즈 계산 후 저장:", self.pose_cache["home"])
            self.pose_cache["pick"]   = self.get_forward_kinematic(pick_pose_j)
            print("픽 포즈 계산 후 저장:", self.pose_cache["pick"])
            self.pose_cache["place"]  = self.get_forward_kinematic(place_pose_j)
            print("플레이싱 포즈 계산 후 저장:", self.pose_cache["place"])
            self.pose_cache["calibration"] = self.get_forward_kinematic(calibration_pose_j)
            print("캘리브레이션 포즈 계산 후 저장:", self.pose_cache["calibration"])
            self.pose_cache["tcp"]    = [float(x) for x in calling_tcp[2:]]
            print("TCP 포즈 저장:", self.pose_cache["tcp"])
            self.pose_cache["marker"] = [float(x) for x in calling_marker[2:]]
            print("마커 포즈 저장:", self.pose_cache["marker"])
            
            place_pose_fk = self.pose_cache["place"].copy()
            place_above_pose_fk = place_pose_fk.copy()
            place_above_pose_fk[2] += 0.05
            self.pose_cache["place_above"] = place_above_pose_fk
            print("플레이스 어보브 포즈 계산 후 저장:", self.pose_cache["place_above"])
            self.pose_cache["place_above_j"] = self.get_inverse_kinematic(place_above_pose_fk)
            print("플레이스 어보브 조인트 포즈 계산 후 저장:", self.pose_cache["place_above_j"])

            if not all([self.pose_cache["pick"], self.pose_cache["place"], self.pose_cache["place_above"], self.pose_cache["place_above_j"], self.pose_cache["home"], self.pose_cache["calibration"], 
                        self.pose_cache["tcp"], self.pose_cache["marker"]]):
                print("[ERROR] FK 계산 실패 및 정보 미흡으로 pose_cache에 저장하지 못함.")
                return

            # ✅ FK 전용 테이블에만 저장
            for name in ["home", "pick", "place", "calibration", "tcp", "marker"]:
                if name in self.pose_cache and self.pose_cache[name]:
                    self.db_manager.insert_fk_pose(name, self.pose_cache[name])

            print("[INFO] FK 포즈는 fk_poses 테이블에 저장되고 Excel로 내보내집니다.")

        except Exception as e:
            print(f"[ERROR] update_all_fk_poses가 실패했습니다.: {e}")
            
        finally:
            self.set_fk_status(active=False)
            self.is_fk_calculating = False

    def set_fk_status(self, active: bool):
        """FK 계산 중임을 시각적으로 표시"""
        if active:
            # FK 작업 시작 시: 노란색 원 + 텍스트
            self.ui.status_circle.setStyleSheet(
                "border-radius: 30px; background-color: yellow;"
            )
            self.ui.status_label.setText("로봇 좌표 계산 중...")
            # 다른 버튼 비활성화
            for btn in [
                self.ui.start_button, self.ui.pause_button, self.ui.stop_button,
                self.ui.power_on_button, self.ui.power_off_button,
                self.ui.robot_home_button
            ]:
                btn.setEnabled(False)
        else:
            # FK 완료 후: 원래 상태로 복귀
            self.robot_status_update()
            for btn in [
                self.ui.start_button, self.ui.pause_button, self.ui.stop_button,
                self.ui.power_on_button, self.ui.power_off_button,
                self.ui.robot_home_button
            ]:
                btn.setEnabled(True)
        
    def job(self):
        paused = self.ui.pause_button.isChecked()
        if paused:
            return
        else:
            if not self.power:
                print("[INFO] 로봇 전원이 꺼져있으므로 작업을 시작할 수 없습니다.")
            elif self.power and not self.running:
                print("[INFO] 작업을 시작합니다.")
            elif self.running and not self.alarm:
                print("[INFO] 작업 중입니다..")
            else:
                print("[INFO] 로봇 알람이 발생하였습니다. 작업을 정지합니다.")
        
        # self.job_command()
        job_result = self.job_command()
        
        if job_result == "no_objects":
            self.robot_send.robot_stop()
            self.job_thread.stop()
            QMetaObject.invokeMethod(
                self, 
                "show_job_done_popup", 
                Qt.QueuedConnection
            )
        
    @pyqtSlot()
    def show_job_done_popup(self):
        """작업 완료 팝업"""
        self.show_message(
            title="작업 완료",
            message="감지 가능한 물체가 없어 작업을 종료합니다.",
            text_color="#FFFFFF",
            font_size=20
        )
        
    def is_grip_lost(self):
        try:
            di_values = self.io.Read_Input_Data()
            if di_values[1] == 0:
                return True
        except Exception as e:
            print("[WARN] 흡착 감지 센서 에러:", e)
    
    def send_moveL_and_wait_for_move_up(self, target_pose, a=1.5, v=1.0,
                            pos_tol=0.005, timeout=20.0):

        print(f"[INFO] moveL_wait started: target={target_pose}")
        
        # with self.robot_lock:
        self.send_moveL(target_pose, a, v)

        start_time = time.time()

        motion_started = False
        reached_once = False

        while True:
            if self.ui.pause_button.isChecked():
                print("[INFO] 일시 정지 상태 진입... 대기 중")
                
                self.wait_if_paused() 

                print("[INFO] 재시작됨 → 이동 명령 재전송 (Resending moveL)")
                
                self.send_moveL(target_pose, a, v)
                
                start_time = time.time()
                motion_started = False
                reached_once = False
                
                time.sleep(0.1) 
                continue
            
            if time.time() - start_time > timeout:
                print("[ERROR] moveL timeout")
                return False

            tcp = self.robot_send.get_tcp()
            state = self.robot_send.get_state()
                
            if tcp is None or any(v is None for v in tcp):
                time.sleep(0.1)
                continue

            cur_xyz = np.array(tcp[:3], dtype=float)
            tgt_xyz = np.array(target_pose[:3], dtype=float)
            dist = np.linalg.norm(cur_xyz - tgt_xyz)

            running = state.get("running") if state else None
            
            if not motion_started and dist < pos_tol:
                print("[INFO] already at target")
                return True

            if not motion_started:
                if running is True and dist > pos_tol:
                    motion_started = True
                    print("[INFO] motion started")
                time.sleep(0.1)
                continue

            if dist < pos_tol:
                reached_once = True

            if running is False:
                if reached_once:
                    print("[INFO] ✓ moveL finished")
                    return True
                else:
                    print(f"[WARN] moveL stopped early (dist={dist:.6f})")
                    return False

            time.sleep(0.1)
                   
    def send_moveL_and_wait(self, target_pose, a=1.5, v=1.0,
                            pos_tol=0.005, timeout=20.0):
        
        if not self.is_auto_running:
            print("[WARN] 작업 정지 상태이므로 이동 명령 취소")
            return False

        print(f"[INFO] moveL_wait started: target={target_pose}")
        
        # with self.robot_lock:
        self.send_moveL(target_pose, a, v)

        start_time = time.time()

        motion_started = False
        reached_once = False

        while True:
            if not self.is_auto_running:
                print("[INFO] 작업 정지 플래그 감지 -> 이동 대기 중단")
                self.robot_send.robot_stop()
                return False
            
            if self.ui.pause_button.isChecked():
                print("[INFO] 일시 정지 상태 진입... 대기 중")
                
                self.wait_if_paused() 
                
                if not self.is_auto_running: 
                    return False

                print("[INFO] 재시작됨 → 이동 명령 재전송 (Resending moveL)")
                
                self.send_moveL(target_pose, a, v)
                
                start_time = time.time()
                motion_started = False
                reached_once = False
                
                time.sleep(0.1) 
                continue
            
            if time.time() - start_time > timeout:
                print("[ERROR] moveL timeout")
                return False

            if self.job_thread is None: 
                 pass

            # with self.robot_lock:
            tcp = self.robot_send.get_tcp()
            state = self.robot_send.get_state()
                
            if tcp is None or any(v is None for v in tcp):
                time.sleep(0.1)
                continue

            cur_xyz = np.array(tcp[:3], dtype=float)
            tgt_xyz = np.array(target_pose[:3], dtype=float)
            dist = np.linalg.norm(cur_xyz - tgt_xyz)

            running = state.get("running") if state else None
            
            if not motion_started and dist < pos_tol:
                print("[INFO] already at target")
                return True

            if not motion_started:
                if running is True and dist > pos_tol:
                    motion_started = True
                    print("[INFO] motion started")
                time.sleep(0.1)
                continue

            if dist < pos_tol:
                reached_once = True

            if running is False:
                if reached_once:
                    print("[INFO] ✓ moveL finished")
                    return True
                else:
                    print(f"[WARN] moveL stopped early (dist={dist:.6f})")
                    return False

            time.sleep(0.1)
            
    def send_moveL(self, pose, a=1.5, v=1.0):
        cmd = f"movel({pose}, a={a}, v={v})\n"
        script = f"def m():\n    {cmd}end\n"
        self.robot_send.send_command(script)
        
    def send_moveJ_and_wait(self, target_pose_l, target_pose_j, a=1.5, v=1.0,
                            pos_tol=0.005, timeout=20.0):
        if not self.is_auto_running:
            print("[WARN] 작업 정지 상태이므로 이동 명령 취소")
            return False

        print(f"[INFO] moveJ_wait started: target={target_pose_j}")
        
        # with self.robot_lock:
        self.send_moveJ(target_pose_j, a, v)

        start_time = time.time()
        motion_started = False
        reached_once = False

        while True:
            if not self.is_auto_running:
                print("[INFO] 작업 정지 플래그 감지 -> 이동 대기 중단")
                self.robot_send.robot_stop()
                return False
            
            if self.ui.pause_button.isChecked():
                print("[INFO] 일시 정지 상태 진입... 대기 중")
                
                self.wait_if_paused() 
                
                if not self.is_auto_running: 
                    return False

                print("[INFO] 재시작됨 → 이동 명령 재전송 (Resending moveL)")
                
                self.send_moveJ(target_pose_j, a, v)
                
                start_time = time.time()
                motion_started = False
                reached_once = False
                
                time.sleep(0.1) 
                continue

            if time.time() - start_time > timeout:
                print("[ERROR] moveJ timeout")
                return False

            # with self.robot_lock:
            tcp = self.robot_send.get_tcp()
            state = self.robot_send.get_state()
            
            if tcp is None:
                time.sleep(0.1)
                continue

            cur_xyz = np.array(tcp[:3], dtype=float)
            tgt_xyz = np.array(target_pose_l[:3], dtype=float)
            dist = np.linalg.norm(cur_xyz - tgt_xyz)

            running = state.get("running") if state else None
            
            if not motion_started and dist < pos_tol:
                print("[INFO] already at target")
                return True

            if not motion_started:
                if running is True and dist > pos_tol:
                    motion_started = True
                    print("[INFO] motion started")
                time.sleep(0.1)
                continue

            if dist < pos_tol:
                reached_once = True

            if running is False:
                if reached_once:
                    print("[INFO] ✓ moveJ finished")
                    return True
                else:
                    print(f"[WARN] moveJ stopped early (dist={dist:.6f})")
                    return False

            time.sleep(0.1)
            
    def send_moveJ(self, pose, a=1.5, v=1.0):
        cmd = f"movej({pose}, a={a}, v={v})\n"
        script = f"def m():\n    {cmd}end\n"
        self.robot_send.send_command(script)

    def job_move_home(self):
        try:
            cache = self.pose_cache
            self.job_move_up()
            # print(cache)
            if not cache:
                print("[ERROR] 로봇 자세에 대한 정보가 비어있습니다. 로봇 좌표 계산을 먼저 진행해주세요.")
                return
            home_pose_l   = cache["home"]
            print("[INFO] 로봇을 홈 위치로 이동합니다..")
            return self.send_moveL_and_wait(home_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 로봇 홈 위치 이동 오류: {e}")
            
    def job_move_home_after_picking(self):
        try:
            cache = self.pose_cache
            # print(cache)
            if not cache:
                print("[ERROR] 로봇 자세에 대한 정보가 비어있습니다. 로봇 좌표 계산을 먼저 진행해주세요.")
                return
            home_pose_l   = cache["home"]
            print("[INFO] 로봇을 홈 위치로 이동합니다..")
            return self.send_moveL_and_wait(home_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 로봇 홈 위치 이동 오류: {e}")

    def job_move_home_after_placing(self):
        try:
            cache = self.pose_cache
            # print(cache)
            if not cache:
                print("[ERROR] 로봇 자세에 대한 정보가 비어있습니다. 로봇 좌표 계산을 먼저 진행해주세요.")
                return
            home_pose_l   = cache["home"]
            print("[INFO] 로봇을 홈 위치로 이동합니다..")
            return self.send_moveL_and_wait(home_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 로봇 홈 위치 이동 오류: {e}")
          
    def job_move_pick_home(self):
        try:
            cache = self.pose_cache
            if not cache:
                return False
            pick_pose_l = cache["pick"]
            print("[INFO] → 픽 홈 이동")
            return self.send_moveL_and_wait(pick_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 픽 홈 이동 오류: {e}")
            return False
    
    def job_move_pick_down(self, vision_pick_pose):
        try:
            cache = self.pose_cache
            if not cache:
                return False
            pick_pose_l = cache["pick"]
            tcp = cache["tcp"]
            tcp_pose_z = float(tcp[2]) / 1000.0
            picking_pose = [
                vision_pick_pose[0],
                vision_pick_pose[1],
                vision_pick_pose[2] + tcp_pose_z,
                pick_pose_l[3],
                pick_pose_l[4],
                pick_pose_l[5],
            ]
            if picking_pose[2] < Z_LIMIT:
                picking_pose[2] = Z_LIMIT
            cache["picking"] = picking_pose
            print("[INFO] → 픽 하강")
            return self.send_moveL_and_wait(picking_pose, a=ACC_FAST_2ND, v=SPEED_FAST_2ND)
        except Exception as e:
            print(f"[ERROR] 픽 하강 오류: {e}")
            return False 
        
    # def job_move_picking(self, step=0.01):
    #     try:
    #         tcp = self.robot_send.get_tcp()
    #         pose = list(tcp)            
    #         print("pose =", pose, type(pose))

    #         z_limit = Z_LIMIT
            
    #         # 1. Z limit까지 무조건 하강 (중간 센서 체크 주석 처리)
    #         while pose[2] > z_limit:
    #             # 테스트용: 하강 중 센서 감지 로직 주석 처리 (끝까지 내려가기 위함)
    #             # di_values = self.io.Read_Input_Data()
    #             # if di_values[1] == 1:
    #             #     print("그리퍼 흡착 확인 완료")
    #             #     return True
                
    #             pose[2] -= step
    #             self.send_moveL_and_wait(pose, a=ACC_SLOW, v=SPEED_PICK)

    #         # 2. 하강 완료 후 무조건 성공 처리
    #         print("[TEST] Z_LIMIT 도달 완료 -> 강제 흡착 성공(True) 반환")
    #         return True

    #     except Exception as e:
    #         print(f"[ERROR] 로봇 픽 위치 이동 오류: {e}")
    #         return False
    
            
    def job_move_picking(self, step=0.01):
        try:
            tcp = self.robot_send.get_tcp()
            pose = list(tcp)            
            print("pose =", pose, type(pose))

            z_limit = Z_LIMIT
            SUCTION_WAIT_TIME = 0.3   # 🔥 흡착 대기 시간 (초)
            CHECK_INTERVAL = 0.01

            while pose[2] > z_limit:
                di_values = self.io.Read_Input_Data()

                # 🔹 흡착 감지
                if di_values[1] == 1:
                    print("그리퍼 흡착 확인 완료")
                    return True
                
                pose[2] -= step
                self.send_moveL_and_wait(pose, a=ACC_SLOW, v=SPEED_PICK)

            # ================================
            # 🟡 접촉 후 흡착 대기 구간
            # ================================
            print("[INFO] 접촉 후 흡착 대기 중...")

            start_time = time.time()
            while time.time() - start_time < SUCTION_WAIT_TIME:
                di_values = self.io.Read_Input_Data()
                if di_values[1] == 1:
                    print("그리퍼 흡착 확인 완료 (지연 감지)")
                    return True
                time.sleep(CHECK_INTERVAL)
                
            print("[WARN] 픽 실패 (흡착 신호 없음)")
            return False

        except Exception as e:
            print(f"[ERROR] 로봇 픽 위치 이동 오류: {e}")

    def move_up(self):
        try:
            tcp = self.robot_send.get_tcp()
            pose = list(tcp)
            print("pose =", pose, type(pose))
            
            cache = self.pose_cache
            home_pose_l   = cache["home"]
            z_limit = home_pose_l[2]
                
            pose[2] = z_limit
            print("[INFO] 로봇 복귀 시작")
            return self.send_moveL_and_wait_for_move_up(pose, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 로봇 상단 위치 이동 오류: {e}")
            
    def job_move_up(self):
        try:
            tcp = self.robot_send.get_tcp()
            pose = list(tcp)
            print("pose =", pose, type(pose))
            
            cache = self.pose_cache
            home_pose_l   = cache["home"]
            z_limit = home_pose_l[2]
                
            pose[2] = z_limit
            print("[INFO] 로봇 복귀 시작")
            return self.send_moveL_and_wait(pose, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 로봇 상단 위치 이동 오류: {e}")
    
    def job_move_place(self):
        try:
            cache = self.pose_cache
            if not cache:
                return False
            place_pose_l = cache["place"]
            print("[INFO] → 플레이스 위치")
            return self.send_moveL_and_wait(place_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 플레이스 이동 오류: {e}")
            return False
        
    def job_move_place_above(self):
        try:
            cache = self.pose_cache
            if not cache:
                return False
            place_above_pose_l = cache["place_above"]
            print("[INFO] → 플레이스 상단")
            return self.send_moveL_and_wait(place_above_pose_l, a=ACC_FAST, v=SPEED_FAST)
        except Exception as e:
            print(f"[ERROR] 플레이스 상단 이동 오류: {e}")
            return False
        
    def job_move_J_place_above(self):
        try:
            # print(self.pose_cache.keys())
            cache = self.pose_cache
            if not cache:
                return False
            place_above_pose = cache["place_above"]
            place_above_pose_j = cache["place_above_j"]
            print("[INFO] → 플레이스 상단")
            return self.send_moveJ_and_wait(place_above_pose, place_above_pose_j, a=ACC_FAST_J, v=SPEED_FAST_J)
        except Exception as e:
            print(f"[ERROR] 플레이스 상단 move J 이동 오류: {e}")
            return False
            
    def get_vision_pick_blocking(self):
        """비전으로 물체 감지 (무한 대기, 중단 가능)"""
        while self.job_thread:
            try:
                pose = self.detect_object_and_get_pose()
                if pose is not None:
                    return pose
                else:
                    self.BIN_Cylinder_move()
                    return None
            except Exception as e:
                print(f"[WARN] 비전 감지 오류: {e}")

            time.sleep(0.1)

        print("[INFO] 비전 감지 중단")
        return None
    
    def BIN_Cylinder_move(self):
        try:
            # self.shaking_time = True
            print("[ACTION] BIN cylinder shaking start")

            self.io.BIN_Cylinder("OFF")
            time.sleep(0.5)
            self.io.BIN_Cylinder("ON")
            time.sleep(0.5)

            print("[ACTION] BIN cylinder shaking end")
        except Exception as e:
            print(f"[WARN] 박스 실린더 제어 오류: {e}")
        finally:
            self.shaking_time = False
            
    def job_command(self):
        try:
            if not self.pose_cache:
                print("[ERROR] 로봇 자세 정보 없음")
                return

            max_no_object_attempts = 2
            no_object_count = 0

            print("[INFO] 반복 작업 시작")
            cycle_count = 0

            while self.is_auto_running:

                # ⛔ 쉐이킹 중이면 루프 정지
                if self.shaking_time:
                    self.BIN_Cylinder_move()
                    continue

                self.io.Blow("OFF")
                self.io.Vacuum("OFF")

                if not self.power:
                    print("[INFO] 로봇 상태 이상 → 작업 종료")
                    break

                # ===============================
                # 1. HOME
                # ===============================
                if not self.job_move_home():
                    break
                
                # =======================================================
                # 외부 PLC 작업 수량 도달(정지) 신호 확인 (DI 8번 핀)
                # =======================================================
                try:
                    di_values = self.io.Read_Input_Data()
                    # 8번 핀(plc_work_done) 신호가 1(ON)인지 확인
                    if di_values and len(di_values) > 8 and di_values[self.io.plc_work_done] == 1:
                        print("[INFO] 외부 PLC 작업 수량 도달(정지) 신호 감지(8번 핀) → 작업을 정지합니다.")
                        self.is_auto_running = False
                        self.robot_send.robot_stop()
                        break
                except Exception as e:
                    print(f"[WARN] PLC 정지 신호 확인 중 오류 발생: {e}")
                # =======================================================

                # ===============================
                # 2. VISION
                # ===============================
                vision_pick_pose = self.get_vision_pick_blocking()

                if vision_pick_pose is None:
                    no_object_count += 1
                    print(f"[WARN] 작업물 감지 실패 ({no_object_count}/{max_no_object_attempts})")

                    # 홈 → 쉐이크
                    self.job_move_home()
                    self.BIN_Cylinder_move()
                    self.shaking_time = True

                    if no_object_count >= max_no_object_attempts:
                        print("[END] 감지 실패 → no_object")
                        return "no_objects"

                    continue

                no_object_count = 0

                # ===============================
                # 3. PICK
                # ===============================
                if not self.job_move_pick_home():
                    continue

                if not self.job_move_pick_down(vision_pick_pose):
                    continue

                self.io.Vacuum("ON")
                
                
                if not self.job_move_picking():
                    self.io.Vacuum("OFF")
                    if not self.job_move_home():
                        print("[ERROR] 홈 이동 실패 → BIN 흔들기 취소")
                        break

                    self.shaking_time = True
                    continue

                # ===============================
                # 4. PLACE
                # ===============================
                if not self.job_move_pick_home() or self.is_grip_lost():
                # if not self.job_move_pick_home():
                    continue

                if not self.job_move_J_place_above() or self.is_grip_lost():
                # if not self.job_move_J_place_above():
                    continue

                if not self.job_move_place():
                    continue

                self.io.Vacuum("OFF")
                self.io.Blow("ON")
                time.sleep(0.25)
                self.io.Blow("OFF")

                if not self.job_move_home_after_placing():
                    continue

                self.vision_pick_cache["valid"] = False
                cycle_count += 1
                print(f"[INFO] ✅ Cycle {cycle_count} 완료\n")

        except Exception as e:
            print(f"[ERROR] job_command 오류: {e}")
            self.is_auto_running = False
            import traceback
            traceback.print_exc()
            
        finally:
            self.is_auto_running = False
            print(f"[INFO] 로봇 모든 작업 완료\n")
        
    def update_robot_status(self, values):
        try:
            if self.robot_disconnected and values is not None:
                print("[INFO] 로봇 재연결 감지! UI를 복구합니다.")
                self.robot_disconnected = False
                self.ui.scrollArea.append("[INFO] 로봇 재연결됨")
                
            if self.robot_disconnected:
                return
            
            if values is None or len(values) < 6:
                print("[WARN] 로봇 상태 이상 발생:", values)
                return
            
            self.power, self.running, self.speed, self.alarm, self.mode, self.control_mode = values

            if not self.robot_status_initialized:
                self.robot_status_initialized = True
                print("[SYNC] Robot status initialized:", values)
            
            if not self.is_alarm_state():
                self.alarm_popup_shown = False
                
            if not self._speed_initialized and self.speed is not None:
                self._speed_initialized = True

                self.ui.speed_slider.blockSignals(True)
                self.ui.speed_slider.setValue(int(self.speed))
                self.ui.speed_slider.blockSignals(False)

                self.ui.speed_label.setText(f"{int(self.speed)}%")
                print(f"[SYNC] UI speed synced to robot: {self.speed}%")
                
            self.robot_status_update()
            
            if self.power:
                self.button_enable()
            else:
                self.button_disable()
        except Exception as e:
            import traceback
            print("[WARN] 로봇 상태 업데이트 오류:", e)
            traceback.print_exc()

    def button_enable(self):
        buttons = [
            self.ui.start_button,
            self.ui.power_off_button,
            self.ui.pause_button,
            self.ui.stop_button,
            self.ui.robot_home_button,
        ]
        enable = self.mode in (4, 7)

        for btn in buttons:
            btn.setEnabled(enable)
            
    def button_disable(self):
        buttons = [
            self.ui.start_button,
            self.ui.power_off_button,
            self.ui.pause_button,
            self.ui.stop_button,
            self.ui.robot_home_button,
        ]
        for btn in buttons:
            btn.setDisabled(not bool(self.power))
            
    def is_alarm_state(self) -> bool:
        """
        세이프티 모드 기준 알람 판별
        NORMAL(1)이 아니면 전부 알람
        """
        SAFETY_MODE_STOP = [3, 5, 6, 7, 12, 13]
        if self.alarm in SAFETY_MODE_STOP:
            return True
        else:
            return False
    
    def should_show_alarm_popup(self) -> bool:
        return self.is_alarm_state() and not self.alarm_popup_shown
    
    def is_idle_state(self):
        return self.power and self.mode == 5

    def is_booting_state(self):
        return self.power and self.mode == 2
    
    def on_robot_status_received(self, status):
        self.robot_status = status
        self.robot_status_initialized = True

    def robot_status_update(self):
        if not getattr(self, "robot_status_initialized", False):
            return
        # FK 계산 중이면 UI 갱신 금지
        if getattr(self, "is_fk_calculating", False):
            return

        # 타워램프 기본값 (모두 OFF)
        lamp_status = ["OFF", "OFF", "OFF"]

        # ================================
        # 1️⃣ 로봇 연결 끊김 (통신 불가)
        # ================================
        if self.robot_disconnected:
            color = "red"
            text = "로봇 연결 끊김"
            self.disable_all_buttons()

            # 🔴 적색등 ON / 부저 OFF 
            lamp_status = ["ON", "OFF", "OFF"]
            
            self.io.Tower_Lamp(lamp_status)
            self.ui.status_circle.setStyleSheet(
                f"border-radius: 30px; background-color: {color};"
            )
            self.ui.status_label.setText(text)
            
            return
            
        # ================================
        # 2️⃣ 로봇 알람 발생 (통신 OK)
        # ================================
        if self.is_alarm_state():
            color = "red"
            text = "알람 발생"
            self.disable_all_buttons()

            # 🔴 적색등 ON + 부저 ON
            lamp_status = ["ON", "OFF", "OFF"]
            self.buzzer_value = True
            self.io.Buzzer("ON")

            # 🚨 알람 팝업 (1회만)
            if not self.alarm_popup_shown:
                self.alarm_popup_shown = True
                self.show_alarm_popup()

        # ================================
        # 2️⃣ 전원 OFF (정지 상태)
        # ================================
        elif not self.power:
            color = "red"
            text = "전원 꺼짐"
            self.disable_all_buttons()
            self.ui.power_on_button.setEnabled(True)
            
            # 🛑 적색등만 ON
            lamp_status = ["ON", "OFF", "OFF"]
            
        # ================================
        # 3️⃣ Booting (부팅 중)
        # ================================
        elif self.is_booting_state():
            color = "yellow"
            text = "부팅 중"
            self.disable_all_buttons()
            self.ui.power_on_button.setEnabled(True)
            
            # ⚠️ 황색등 ON
            lamp_status = ["OFF", "ON", "OFF"]

        # ================================
        # 3️⃣ Idle (대기/준비 중)
        # ================================
        elif self.is_idle_state():
            color = "yellow"
            text = "대기 상태"
            self.disable_all_buttons()
            self.ui.power_on_button.setEnabled(True)
            
            # ⚠️ 황색등 ON
            lamp_status = ["OFF", "ON", "OFF"]

        # ================================
        # 4️⃣ Ready / Running (정상 가동)
        # ================================
        else:
            if self.running or self.is_auto_running:
                color = "blue"
                text = "로봇 동작 중"
                # 🔵 동작 중에는 녹색등 ON (필요시 황색 혼합 가능)
                lamp_status = ["OFF", "OFF", "ON"]
            else:
                color = "green"
                text = "전원 켜짐"
                # ✅ 대기 중 녹색등 ON
                lamp_status = ["OFF", "ON", "OFF"]

            self.button_enable()

        # ================================
        # IO 모듈에 타워램프 명령 전송
        # ================================
        try:
            self.io.Tower_Lamp(lamp_status)
        except Exception as e:
            print(f"[IO_ERROR] 타워램프 제어 실패: {e}")

        # ================================
        # UI 반영
        # ================================
        self.ui.status_circle.setStyleSheet(
            f"border-radius: 30px; background-color: {color};"
        )
        self.ui.status_label.setText(text)

    def on_start_button_clicked(self, checked=False):
        try:
            print("[INFO] 작업 시작 요청 전송")
            if self.is_auto_running:
                print("[WARN] 이미 작업이 실행 중입니다.")
                return

            if not hasattr(self, 'pose_cache') or not self.pose_cache:
                print("[WARN] 포즈 캐시가 비어있어 DB에서 다시 로드합니다.")
                self.load_pose_cache_from_db()

            if not hasattr(self, 'pose_cache') or not self.pose_cache:
                print("[WARN] 포즈 캐시 없음. update_all_fk_poses() 필요.")
                QMessageBox.warning(self, "데이터 없음", "로봇 좌표 데이터가 없습니다.\n초기화 후 다시 시도하세요.")
                return

            # 1. 상태 잠금 (가장 먼저)
            self.is_auto_running = True

            # 3. 스레드 시작
            self.job_thread = JobThread(self)
            self.job_thread.start()
            print("[INFO] 작업 시작됨 (Auto Running: ON)")

        except Exception as e:
            print(f"[ERROR] 시작 명령 실패: {e}")

        finally:
            self.ui.pause_button.setChecked(False)
            self.ui.pause_button.setText("일시 정지")
            
    def wait_if_paused(self):
        while self.ui.pause_button.isChecked():
            time.sleep(0.05)

    def on_pause_button_toggled(self, checked=False):
        if checked:
            try:
                self.robot_send.robot_pause()
                self.ui.pause_button.setText("재시작")
                print("[INFO] 로봇 일시 정지")
            except Exception as e:
                print(f"[ERROR] 일시 정지 오류: {e}")
        else:
            try:
                self.ui.pause_button.setText("일시 정지")
                print("[INFO] 로봇 재시작")
            except Exception as e:
                print(f"[ERROR] 재시작 오류: {e}")
            
    def on_stop_button_clicked(self):
        try:
            print("[INFO] 로봇 정지")

            self.is_auto_running = False
            
            self.robot_send.robot_stop()

            self.ui.start_button.setEnabled(True)
            self.ui.robot_home_button.setEnabled(True)

            print("[INFO] 로봇 정지 명령 전송 및 상태 플래그 해제")

        except Exception as e:
            print(f"[ERROR] 정지 오류: {e}")
        finally:
            self.ui.pause_button.setChecked(False)
            self.ui.pause_button.setText("일시 정지")
            self.robot_status_update()

    def on_robot_power_on_button_clicked(self, pressed):
        if not pressed:
            return
        
        state_dict = self.robot_send.get_state()

        current_mode = state_dict.get("mode")
        is_power_on = state_dict.get("power")

        print(f"[INFO] Power sequence start. Current: Power={is_power_on}, Mode={current_mode}")

        # 이미 Ready 상태
        if is_power_on and current_mode == 7:
            print("[INFO] 로봇이 이미 Ready 상태입니다.")
            return

        # 타이머 초기화
        if hasattr(self, 'power_on_timer') and self.power_on_timer.isActive():
            self.power_on_timer.stop()
        
        self.power_on_timer = QtCore.QTimer(self)
        self.power_on_timer.timeout.connect(self.robot_power_on)

        # 재전송 딜레이를 위한 카운터 변수 추가
        self.retry_counter = 0 

        # 시작 단계 설정
        if is_power_on and current_mode == 5:
            print("[INFO] Idle 상태 감지 → 브레이크 해제 단계(STEP 3)부터 시작")
            self.power_on_step = 3
        else:
            print("[INFO] 전원 OFF 상태 감지 → 전체 시퀀스(STEP 0) 시작")
            self.power_on_step = 0

        # 타이머 시작 (0.2초 간격)
        self.power_on_timer.start(200)
        
    def robot_power_on(self):
        try:
            # ✅ [수정] 딕셔너리 방식으로 안전하게 값 가져오기
            state_dict = self.robot_send.get_state()
            mode = state_dict.get("mode")  
            power = state_dict.get("power")

            # =========================================================
            # STEP 0: 제어권 요청
            # =========================================================
            if self.power_on_step == 0:
                self.robot_send.robot_29999.send_command_29999("remoteControl -on")
                self.robot_send.robot_29999.send_command_29999("robotControl -on")
                print("[STEP 0] Remote / Robot control ON 요청")
                self.power_on_step = 1
                return

            # =========================================================
            # STEP 1: 전원 켜기
            # =========================================================
            if self.power_on_step == 1:
                # 전원이 꺼져있거나(False), 아직 상태를 못 읽었으면(None) 명령 전송
                if not power: 
                    self.robot_send.robot_power_on()
                    # (로그가 너무 많이 뜨면 주석 처리)
                    # print(f"[STEP 1] 전원 켜기 시도 중... (현재 Power: {power})")
                else:
                    print(f"[STEP 1] Power ON 확인됨 (Power: {power}) → Idle 대기 진입")
                    self.power_on_step = 2
                return

            # =========================================================
            # STEP 2: Idle 모드 대기 (여기가 문제였던 곳)
            # =========================================================
            if self.power_on_step == 2:
                # Idle 상태 (5) 확인
                if mode == 5:
                    print(f"[STEP 2] Idle 모드 진입 확인 (Mode: {mode}) → 브레이크 해제 진입")
                    self.power_on_step = 3
                    self.retry_counter = 0
                else:
                    # ✅ [추가] 왜 안 넘어가는지 확인하기 위한 로그
                    # 로봇 부팅 중에는 0(None), 1(Init), 3(PowerOff) 등이 뜰 수 있음
                    print(f"[STEP 2] Idle 모드 대기 중... 현재 Mode: {mode}")
                
                return

            # =========================================================
            # STEP 3: 브레이크 해제
            # =========================================================
            if self.power_on_step == 3:
                # 1. Ready(7)가 되었다면 성공
                if mode == 7:
                    print("[STEP 3] 로봇 Ready 상태 변경 확인! → 완료 단계로 이동")
                    self.power_on_step = 4
                    return

                # 2. 아직 Idle(5) 혹은 다른 상태라면 반복 전송
                # (1초에 한 번 정도만 전송하도록 조건 추가)
                if self.retry_counter % 5 == 0:
                    self.robot_send.robot_29999.send_command_29999("brakeRelease")
                    print(f"[STEP 3] 브레이크 해제 명령 전송... (현재 Mode: {mode})")
                
                self.retry_counter += 1
                return

            # =========================================================
            # STEP 4: 최종 완료 처리
            # =========================================================
            if self.power_on_step == 4:
                print("[STEP 4] 로봇 부팅 시퀀스 완료.")
                self.power_on_timer.stop()

                if not getattr(self, "fk_updated", False):
                    # self.update_all_fk_poses()
                    self.db_manager.export_to_excel()
                    self.fk_updated = True
                    print("[INFO] FK 및 좌표 백업 완료")
                return

        except Exception as e:
            print(f"[ERROR] Power ON 시퀀스 오류: {e}")
            # 디버깅을 위해 에러 전체 출력
            import traceback
            traceback.print_exc()

    def on_robot_power_off_button_clicked(self, pressed):
        if pressed:    
            try:
                self.robot_send.robot_power_off()
                self.button_disable()
                print("[INFO] 전원 끄기")
                self.pose_cache.clear()
            except Exception as e:
                print(f"[ERROR] 전원 끄기 실패: {e}")

    def on_robot_home_button_clicked(self):
        if self.is_auto_running:
            print("[WARN] 자동 작업 중입니다! 홈 이동 명령 무시됨.")
            return
            
        if self.mode == 7:
            pass
        
        try:
            calling_home_pose = self.db_manager.fetch_pose("home")
            home_pose = calling_home_pose[2:]
            print(home_pose)
            home_pose_rad = self.deg_to_rad(home_pose)
            print(home_pose_rad)
            self.move_up()
            command = f"movej({home_pose_rad}, a={ACC_FAST}, v={SPEED_FAST})"
            self.robot_send.send_command(command)
            print("[INFO] 홈 포즈로 이동합니다.")
        except Exception as e:
            print(f"[ERROR] 홈 포즈 이동 실패: {e}")
            

    def on_slider_changed(self, value):
        self.ui.speed_label.setText(f"{value}%")

    def on_speed_changed(self, value):
        self.ui.speed_label.setText(f"{value}%")
        self.robot_send.robot_speed(speed=value)
        print(f"[INFO] Speed set to {value}%")

    def add_log(self, text):
        self.ui.scrollArea.append(text)
        self.ui.scrollArea.moveCursor(QtGui.QTextCursor.End)

        max_lines = 250
        if self.ui.scrollArea.document().blockCount() > max_lines:
            cursor = self.ui.scrollArea.textCursor()
            cursor.movePosition(QtGui.QTextCursor.Start)
            cursor.select(QtGui.QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

if __name__ == "__main__":
    def qt_excepthook(exc_type, exc_value, exc_tb):
        print("[UNCAUGHT EXCEPTION]")
        traceback.print_exception(exc_type, exc_value, exc_tb)

    sys.excepthook = qt_excepthook
    app = QApplication(sys.argv)
    window = PPAPUI()
    window.showMaximized()
    sys.exit(app.exec_())
