import sys
import time
from pathlib import Path
import sqlite3
from openpyxl import Workbook

def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    else:
        return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = project_root()

if getattr(sys, "frozen", False):
    DB_PATH = PROJECT_ROOT / "DB" / "robot_system.db"
    BACKUP_XLSX = PROJECT_ROOT / "DB" / "backup.xlsx"
else:
    DB_PATH = PROJECT_ROOT / "PRAG" / "DB" / "robot_system.db"
    BACKUP_XLSX = PROJECT_ROOT / "PRAG" / "DB" / "backup.xlsx"

class RobotDB:
    def __init__(self, db_path=DB_PATH, backup_xlsx=BACKUP_XLSX):
        self.db_path = db_path
        self.backup_xlsx = backup_xlsx
        self._init_db()

    def _get_connection(self):
        # print("DB_PATH =", self.db_path)
        # print("EXISTS =", Path(self.db_path).exists())
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        """DB 초기화: 테이블이 없으면 생성"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS vision (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vision_name TEXT NOT NULL UNIQUE,
            value REAL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS poses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pose_name TEXT NOT NULL UNIQUE,
            X REAL, Y REAL, Z REAL, Rx REAL, Ry REAL, Rz REAL
        )
        """)
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fk_poses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fk_name TEXT NOT NULL UNIQUE,
            X REAL, Y REAL, Z REAL, Rx REAL, Ry REAL, Rz REAL
        )
        """)
        cursor.execute("PRAGMA table_info(vision)")
        cols = [row[1] for row in cursor.fetchall()]
        if "value" not in cols:
            cursor.execute("ALTER TABLE vision ADD COLUMN value REAL")

        conn.commit()
        conn.close()

    # === 엑셀 백업 ===
    def export_to_excel(self):
        conn = self._get_connection()
        cursor = conn.cursor()

        wb = Workbook()
        wb.remove(wb.active)

        for table in ["vision", "poses", "fk_poses"]:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]

            ws = wb.create_sheet(title=table)
            ws.append(col_names)
            for row in rows:
                ws.append(row)

        wb.save(self.backup_xlsx)
        conn.close()
        print(f"[백업 완료] {self.backup_xlsx}")
        
    # === 비전 추가/갱신 ===
    def insert_vision(self, vision_name, value):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO vision (vision_name, value)
            VALUES (?, ?)
            ON CONFLICT(vision_name) DO UPDATE SET
                value=excluded.value
        """, (vision_name, value))
        conn.commit()
        conn.close()
        time.sleep(0.1)
        self.export_to_excel()

    # === 포즈 추가/갱신 ===
    def insert_pose(self, pose_name, poses):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO poses (pose_name, X, Y, Z, Rx, Ry, Rz)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(pose_name) DO UPDATE SET
                X=excluded.X, Y=excluded.Y, Z=excluded.Z,
                Rx=excluded.Rx, Ry=excluded.Ry, Rz=excluded.Rz
        """, (pose_name, *poses))
        conn.commit()
        conn.close()
        self.export_to_excel()
        
    def insert_fk_pose(self, fk_name, fk_values):
        """FK 계산된 좌표값 (x,y,z,Rx,Ry,Rz) 저장"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO fk_poses (fk_name, X, Y, Z, Rx, Ry, Rz)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(fk_name) DO UPDATE SET
                X=excluded.X, Y=excluded.Y, Z=excluded.Z,
                Rx=excluded.Rx, Ry=excluded.Ry, Rz=excluded.Rz
        """, (fk_name, *fk_values))
        conn.commit()
        conn.close()
        self.export_to_excel()
        print(f"[DB] FK Pose saved: {fk_name} → {fk_values}")

    # # === 작업량 추가/갱신 ===
    # def insert_workload(self, work_name, work_load):
    #     """제품별 작업 횟수를 1회 누적, 날짜 바뀌면 초기화"""
    #     conn = self._get_connection()
    #     cursor = conn.cursor()
    #     today = datetime.now().strftime("%Y-%m-%d")

    #     # 현재 날짜 확인
    #     cursor.execute("SELECT DISTINCT date FROM workload")
    #     row = cursor.fetchone()
    #     if row and row[0] != today:
    #         print(f"[INFO] 날짜 변경 감지 → 전체 초기화")
    #         cursor.execute("UPDATE workload SET work_load = 0, date = ?", (today,))

    #     # 해당 제품의 현재 작업량
    #     cursor.execute("SELECT work_load FROM workload WHERE work_name = ?", (work_name,))
    #     row = cursor.fetchone()
    #     current = int(row[0]) if row else 0
    #     new_value = current + 1

    #     # 갱신
    #     cursor.execute("""
    #         INSERT INTO workload (work_name, work_load, date)
    #         VALUES (?, ?, ?)
    #         ON CONFLICT(work_name) DO UPDATE SET
    #             work_load = excluded.work_load,
    #             date = excluded.date
    #     """, (work_name, new_value, today))

    #     conn.commit()
    #     conn.close()
    #     print(f"[DB] {work_name} → {new_value}회 완료")
    #     self.export_to_excel()

    # === 데이터 조회 ===
    def fetch_table(self, table_name):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        conn.close()
        return rows
    
    def fetch_vision(self, vision_name):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vision WHERE vision_name=?", (vision_name,))
        row = cursor.fetchone()
        conn.close()
        return row

    def fetch_pose(self, pose_name):
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM poses WHERE pose_name=?", (pose_name,))
        row = cursor.fetchone()
        conn.close()
        return list(row) if row else None
    
    def fetch_fk_pose(self, fk_name):
        """FK 포즈 데이터 조회"""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM fk_poses WHERE fk_name=?", (fk_name,))
        row = cursor.fetchone()
        conn.close()
        return list(row) if row else None

    # def fetch_workload(self, work_name):
    #     conn = self._get_connection()
    #     cursor = conn.cursor()
    #     cursor.execute("SELECT * FROM workload WHERE work_name=?", (work_name,))
    #     row = cursor.fetchone()
    #     conn.close()
    #     return row


# # === 사용 예시 ===
# if __name__ == "__main__":
#     db = RobotDB()

#     db.insert_pose("home", [0, 0, 0, 0, 90, 0])
#     db.insert_ip("robot", "192", "168", "0", "101")
#     db.insert_workload("today", 450)

#     print(db.fetch_pose("pick1"))
#     print(db.fetch_ip("robot"))
#     print(db.fetch_workload("today"))
