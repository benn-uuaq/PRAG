import sys
import yaml
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
#  프로젝트 경로 / 경로 설정
# ------------------------------------------------------------
def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    # ⭐ 단일 best.pt 경로 대신 runs_seg 폴더 전체를 참조하도록 수정
    WEIGHT_DIR      = PROJECT_ROOT / "VISION" / "runs_seg"
    HANDEYE_YAML    = PROJECT_ROOT / "VISION" / "config" / "aruco_rigid_result.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "VISION" / "config" / "calibration_intrinsics.yaml"
else:
    WEIGHT_DIR      = PROJECT_ROOT / "PRAG" / "VISION" / "runs_seg"
    HANDEYE_YAML    = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "aruco_rigid_result.yaml"
    INTRINSICS_YAML = PROJECT_ROOT / "PRAG" / "VISION" / "config" / "calibration_intrinsics.yaml"


from VISION.yolo_object_segmentation_qt import YOLOSegDetector, load_roi_from_yaml

# ------------------------------------------------------------
#  Hand–Eye 로드
# ------------------------------------------------------------
def load_base_T_cam(yaml_path: Path) -> np.ndarray:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    bc = data.get("base_T_cam") or data.get("T_base_cam")
    if bc is None:
        raise KeyError("handeye_result.yaml 에 base_T_cam 없음")

    R_bc = np.array(bc["R"], dtype=float)
    t_bc = np.array(bc["t"], dtype=float).reshape(3)

    T = np.eye(4, dtype=float)
    T[:3, :3] = R_bc
    T[:3, 3]  = t_bc
    return T


def cam_to_base(cam_xyz: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    cam_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=float)
    base_h = T_base_cam @ cam_h
    return base_h[:3]


class YOLOToRobot:
    def __init__(self,
                 weight_path=None,
                 handeye_yaml=HANDEYE_YAML,
                 intrinsics_yaml=INTRINSICS_YAML,
                 use_external_frames=False):
        self.detector = YOLOSegDetector(
            weight_path=weight_path,
            intrinsics_yaml=intrinsics_yaml,
            roi=load_roi_from_yaml()
        )
        self.handeye_yaml = handeye_yaml
        self.T_base_cam = load_base_T_cam(handeye_yaml)

    def set_model(self, weight_path):
        """하위 YOLOSegDetector 검출기의 모델 동적 교체"""
        return self.detector.set_model(weight_path)

    def detect_from_frames(self, color, depth, depth_scale):
        detections, img = self.detector.detect_from_frames(
            color, depth, depth_scale
        )

        # 1️⃣ detection 자체가 없으면
        if detections is None or len(detections) == 0:
            return None, None, img

        # 2️⃣ best object 선택
        result = self.select_top_object(detections)
        if result is None:
            return None, None, img

        # 3️⃣ center_xyz 검증
        cam_xyz = result.get("center_xyz", None)
        if cam_xyz is None or len(cam_xyz) != 3:
            return None, None, img

        cam_xyz = np.array(cam_xyz, dtype=float)
        if not np.isfinite(cam_xyz).all():
            return None, None, img

        # 4️⃣ Hand–Eye 변환
        base_xyz = cam_to_base(cam_xyz, self.T_base_cam)

        return result, base_xyz, img

    def select_top_object(self, objs):
        """
        objs: list of detection dicts
        return: best object or None
        """
        if not objs:
            return None

        scored = []
        for o in objs:
            center = o.get("center_xyz", None)
            if center is None or len(center) != 3:
                continue
            z = center[2]

            if "mask_area" in o:
                area = o["mask_area"]
            elif "mask" in o and o["mask"] is not None:
                area = int(np.count_nonzero(o["mask"]))
            else:
                area = 0

            conf = o.get("confidence", 0.0)

            score = (
                -100.0 * z +        # depth 최우선
                0.0001 * area +     # 보이는 면적
                0.1 * conf          # confidence 보조
            )

            scored.append((score, o))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


class YOLOToRobotQt:
    """
    ✔ 카메라 접근 ❌
    ✔ 프레임을 외부(VisionMain)에서 받음
    """

    def __init__(
        self,
        weight_path: Path = None,
        handeye_yaml: Path = HANDEYE_YAML,
        intrinsics_yaml: Path = INTRINSICS_YAML
    ):
        self.current_weight_path = weight_path
        self.handeye_yaml = handeye_yaml
        self.intrinsics_yaml = intrinsics_yaml

        self.yolo_robot = YOLOToRobot(
            weight_path=self.current_weight_path,
            handeye_yaml=self.handeye_yaml,
            intrinsics_yaml=self.intrinsics_yaml
        )

        self.last_result = None
        self.last_base_xyz = None
        self.last_image = None

        print("[VISION][QT] YOLOToRobotQt initialized")
        
    def update_confidence(self):
        """DB에서 신뢰도 값을 다시 읽어와 감지기에 반영"""
        if hasattr(self, 'yolo_robot') and hasattr(self.yolo_robot, 'detector'):
            self.yolo_robot.detector.update_confidence()
            print("[VISION][QT] 비전 신뢰도(Confidence) 갱신 완료")

    def change_target_model(self, weight_path):
        """⭐ 메인 UI에서 선택한 레시피(모델 경로)를 비전 시스템에 적용"""
        self.current_weight_path = weight_path
        return self.yolo_robot.set_model(weight_path)

    # --------------------------------------------------
    # 1️⃣ 실시간 시각화 전용 (좌표 계산 ❌)
    # --------------------------------------------------
    def visualize(self, color, depth, depth_scale):
        try:
            _, vis_img = self.yolo_robot.detector.detect_from_frames(
                color.copy(), depth, depth_scale
            )
            return vis_img
        except Exception as e:
            print("[VISION][QT][VISUALIZE ERROR]", e)
            return color

    # --------------------------------------------------
    # 2️⃣ 로봇 작업용 1회 검출 (좌표 계산 ⭕)
    # --------------------------------------------------
    def detect_once(self, color, depth, depth_scale):
        if color is None or depth is None:
            return False, None, None, None

        try:
            result, base_xyz, img = self.yolo_robot.detect_from_frames(
                color, depth, depth_scale
            )

            if base_xyz is None:
                self._clear_last()
                print("[VISION][QT] Detection failed: base_xyz is None")
                return False, None, None, img

            self.last_result = result
            self.last_base_xyz = base_xyz
            self.last_image = img

            print("[VISION][QT] Pick point:", np.round(base_xyz, 6))
            return True, base_xyz, result, img

        except Exception as e:
            print("[VISION][QT][ERROR] detect_once failed:", e)
            self._clear_last()
            return False, None, None, None

    def reload_config(self):
        """
        [핵심 수정] 설정 재로드 시 현재 작업자가 선택해 둔 모델(current_weight_path)을 유지함
        """
        print("[VISION][QT] Reloading configuration files...")
        try:
            self.yolo_robot = YOLOToRobot(
                weight_path=self.current_weight_path,
                handeye_yaml=self.handeye_yaml,
                intrinsics_yaml=self.intrinsics_yaml
            )
            print("[VISION][QT] Configuration reloaded successfully.")
        except Exception as e:
            print(f"[VISION][QT][ERROR] Config reload failed: {e}")

    # --------------------------------------------------
    # 내부 상태
    # --------------------------------------------------
    def _clear_last(self):
        self.last_result = None
        self.last_base_xyz = None
        self.last_image = None

    # --------------------------------------------------
    # 디버그용
    # --------------------------------------------------
    def has_last_result(self):
        return self.last_base_xyz is not None

    def get_last_base_xyz(self):
        return self.last_base_xyz

    def get_last_result(self):
        return self.last_result

    def get_last_image(self):
        return self.last_image