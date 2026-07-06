import struct
import socket
import select
import threading

import sys, os, time
from pathlib import Path

def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    EXCEL_PATH = PROJECT_ROOT / "ROBOT_CONTROL" / "RobotStateMessage.xlsx"
else:
    EXCEL_PATH = PROJECT_ROOT / "PRAG" / "ROBOT_CONTROL" / "RobotStateMessage.xlsx"

from ROBOT_CONTROL.RobotData import *

SHEET_NAME = "v2.6.0"

HOST = None
PORT1 = 30001
PORT2 = 29999

PC_IP = None
PC_PORT = 30010

DEFAULT_TIMEOUT = 10.0
ROBOT_STATE_TYPE = 16

class Robot():
    def __init__(self, ip): 
        self.ip = ip
        self.recv_data = None  # Store received message

    def RecvPopup(self, port):
        HOST = "0.0.0.0"
        PORT = port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.bind((HOST, PORT))
            s.listen()
            print(f"🟢 Listening for robot on port {PORT}...")
            conn, addr = s.accept()
            with conn:
                print(f"🔁 Connected by {addr}")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    self.recv_data = data.decode().strip()
                    print(f"📥 Received from robot: {self.recv_data}")
                    break  # Exit after one message

    def connectETController(self, ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((ip, port))
            return (True, sock)
        except Exception as e:
            sock.close()
            return (False, None)

    def get_inverse_kinematic(self, p, pc_ip=None, pc_port=None):
        command = f'''
def test():
    socket_open("{pc_ip}", {pc_port})
    sleep(0.5)
    socket_send_string(str(get_inverse_kin({p})))
    socket_close()
end
'''
        conSuc, sock = self.connectETController(self.ip, PORT1)
        if conSuc:
            try:
                print("📤 Sending inverse kinematic request to robot...")
                sock.sendall(command.encode())
            except Exception as e:
                print("❌ Failed to send command:", e)
            finally:
                sock.close()
        else:
            print("❌ Connection to robot failed")

    def get_forward_kinematic(self, p, pc_ip=None, pc_port=None):
        command = f'''
def test():
    socket_open("{pc_ip}", {pc_port})
    sleep(0.5)
    socket_send_string(str(get_forward_kin({p})))
    socket_close()
end
'''
        conSuc, sock = self.connectETController(self.ip, PORT1)
        if conSuc:
            try:
                print("📤 Sending forward kinematic request to robot...")
                sock.sendall(command.encode())
            except Exception as e:
                print("❌ Failed to send command:", e)
            finally:
                sock.close()
        else:
            print("❌ Connection to robot failed")

class Robot_29999():
    def __init__(self, host_ip):
        self.sock = None
        self.host = host_ip
        self.port = PORT2

    def socket_connect_29999(self):
        try:
            if self.sock:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, PORT2))  
            print(f"Connected to {self.host} on port {PORT2}")
            return self.sock
        except Exception as e:
            print(f"Error connecting to {self.host} on port {PORT2}: {e}")
            return None

    def send_command_29999(self, command):
        try:
            if self.sock is None:
                if self.socket_connect_29999() is None:
                    print("[WARN] 29999 not connected; cannot send command")
                    return
            self.sock.sendall(f"{command}\n".encode("utf-8"))
        except Exception as e:
            print(f"Error sending command: {e}")
            # 에러 발생 시 재연결 시도를 위해 소켓 초기화
            self.socket_disconnect_29999()

    def socket_disconnect_29999(self):
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        self.sock = None

class Robot_30001():
    def __init__(self, host_ip, excel_path, sheet_name):
        self.target_ip = host_ip
        self.target_port = PORT1
        self.__data_config = RobotDataConfig.get_config(excel_path, sheet_name)
        
        self.client_socket = None
        self.__buf = b""
        self.latest_data = None
        
    def socket_connect(self):
        # [수정 1] 기존 소켓이 있다면 닫고 새로 생성 (메모리 누수 방지)
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None

        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5.0) # 연결 타임아웃 설정
            self.client_socket.connect((self.target_ip, self.target_port))  
            self.client_socket.setblocking(False) # Non-blocking 모드 전환
            print(f"Connected to {self.target_ip} on port {self.target_port}")
            return self.client_socket
        except Exception as e:
            print(f"Error connecting to {self.target_ip} on port {self.target_port}: {e}")
            return None

    def on_disconnect(self):
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        self.latest_data = None

    def send_command(self, command):
        if not self.client_socket:
            return

        try:
            cmd_bytes = f"{command}\n".encode("utf-8")
            # sendall은 blocking 모드에서 동작하므로 잠시 타임아웃 설정
            self.client_socket.settimeout(1.0)
            self.client_socket.sendall(cmd_bytes)
            self.client_socket.setblocking(False) # 다시 논블로킹 복구
        except Exception as e:
            print(f"[ERROR] Send command failed: {e}")
            self.on_disconnect()

    def update(self):
        if self.client_socket is None:
            return False

        if self.client_socket.fileno() == -1:
            return False

        try:
            chunk = self.client_socket.recv(4096)
            if not chunk: 
                return False
            self.__buf += chunk
        except socket.timeout:
            return False
        except BlockingIOError:
            return False
        except OSError as e:
            if e.errno in [10057, 10038]:
                return False
            else:
                # print(f"[ERROR] Socket OSError: {e}")
                self.on_disconnect()
                return False
        except Exception as e:
            print(f"[ERROR] General Socket error: {e}")
            self.on_disconnect()
            return False

        last_valid_packet = None
        
        # 버퍼가 너무 커지면(1MB 이상) 초기화 (안전장치)
        if len(self.__buf) > 1024 * 1024:
            self.__buf = b""
            return False

        while len(self.__buf) > 5:
            try:
                head = RobotHeader.unpack(self.__buf)
                if len(self.__buf) < head.size:
                    break 

                packet = self.__buf[:head.size]
                self.__buf = self.__buf[head.size:]

                if head.type == ROBOT_STATE_TYPE:
                    last_valid_packet = packet
            except:
                self.__buf = b""
                break
        
        if last_valid_packet:
            try:
                self.latest_data = RobotData.unpack(last_valid_packet, self.__data_config)
                return True
            except Exception as e:
                print(f"[ERROR] Data Unpack Error: {e}")
                return False
            
        return False

    def snapshot(self):
            if self.latest_data is None:
                return None
            
            d = self.latest_data
            
            return {
                "power": getattr(d, "is_robot_power_on", None),
                "running": getattr(d, "is_program_running", None),
                "speed": (
                    int(getattr(d, "get_target_speed_fraction", 0) * 100)
                    if getattr(d, "get_target_speed_fraction", None) is not None
                    else None
                ),
                "alarm": getattr(d, "bord_safe_mode", None),
                "mode": getattr(d, "get_robot_mode", None),
                "control_mode": getattr(d, "get_robot_control_mode", None),
                
                "tcp_x": getattr(d, "tcp_x", None),
                "tcp_y": getattr(d, "tcp_y", None),
                "tcp_z": getattr(d, "tcp_z", None),
                "rot_x": getattr(d, "rot_x", None),
                "rot_y": getattr(d, "rot_y", None),
                "rot_z": getattr(d, "rot_z", None),
            }

class ROBOT_SEND:
    def __init__(self, host_ip):
        self.robot_29999 = Robot_29999(host_ip)
        self.robot_30001 = Robot_30001(host_ip, EXCEL_PATH, SHEET_NAME)
        self.robot_29999.socket_connect_29999()
        self.robot_30001.socket_connect()

        self.state = {
            "power": None,
            "running": None,
            "speed": None,
            "alarm": None,
            "mode": None,
            "control_mode": None,
        }

        self.tcp = {
            "tcp_x": None,
            "tcp_y": None,
            "tcp_z": None,
            "rot_x": None,
            "rot_y": None,
            "rot_z": None,
        }

        self._lock = threading.Lock()
            
    def poll(self):
        if not self.robot_30001.update():
            return False

        snap = self.robot_30001.snapshot()
        if snap is None:
            return False

        with self._lock:
            for k in self.state:
                if snap[k] is not None:
                    self.state[k] = snap[k]

            for k in self.tcp:
                if snap[k] is not None:
                    self.tcp[k] = snap[k]

        return True

    # ===== 외부 접근용 =====
    def get_state(self):
        with self._lock:
            return dict(self.state)

    def get_tcp(self):
        with self._lock:
            return (
                self.tcp["tcp_x"],
                self.tcp["tcp_y"],
                self.tcp["tcp_z"],
                self.tcp["rot_x"],
                self.tcp["rot_y"],
                self.tcp["rot_z"],
            )

    # ===== 로봇 제어 =====
    def robot_power_on(self):
        self.robot_29999.send_command_29999("robotControl -on")

    def robot_power_off(self):
        self.robot_29999.send_command_29999("robotControl -off")

    def robot_play(self):
        self.robot_29999.send_command_29999("play")

    def robot_pause(self):
        self.robot_29999.send_command_29999("pause")

    def robot_stop(self):
        self.robot_29999.send_command_29999("stop")

    def robot_speed(self, speed):
        self.robot_29999.send_command_29999(f"speed -v {speed}")

    def send_command(self, command):
        # [핵심 수정 2] 매번 재연결하지 않고, 연결이 끊겨있을 때만 재연결 시도
        # 기존: sock = self.robot_30001.socket_connect() (무조건 재연결 -> 소켓 누수 원인)
        
        if (self.robot_30001.client_socket is None or 
            self.robot_30001.client_socket.fileno() == -1):
            print("[INFO] Robot command socket disconnected. Reconnecting...")
            self.robot_30001.socket_connect()
            
        self.robot_30001.send_command(command)