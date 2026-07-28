# PyInstaller 커스텀 런타임 훅
# PyInstaller가 자동으로 끼워 넣는 PyQt5 런타임 훅(pyi_rth_pyqt5.py)이
# 메인 스크립트보다 먼저 실행되면서 Qt DLL 경로를 먼저 초기화해버림.
# 이로 인해 torch(cu128)/ultralytics를 나중에 로드할 때
# c10.dll 로딩 access violation이 발생함.
# --runtime-hook으로 이 파일을 넘기면 PyInstaller 부트로더 초기 단계에서
# torch를 가장 먼저 강제로 import하여 DLL 로딩 순서를 고정한다.
import torch  # noqa: F401,E402
