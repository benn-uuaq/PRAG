import time
from pyModbusTCP.server import ModbusServer
from threading import Thread

class IOSimulatorServer:
    def __init__(self, host="0.0.0.0", port=502):
        self.requested_host = host
        self.port = port
        self.server = None
        self.running = False
        self.sim_thread = None

    def start(self):
        try:
            self.server = ModbusServer(host=self.requested_host, port=self.port, no_block=True)
            self.server.start()
            print(f"[SIMULATOR] IO 모듈 시뮬레이터 서버 시작 (IP: {self.requested_host}, Port: {self.port})")
        except Exception as e:
            print(f"[WARN] IP '{self.requested_host}' 바인딩 실패: {e}")
            print(f"[INFO] 대상을 '0.0.0.0' (모든 IP 수신 허용)으로 자동 전환하여 시작합니다.")
            self.server = ModbusServer(host="0.0.0.0", port=self.port, no_block=True)
            self.server.start()
            print(f"[SIMULATOR] IO 모듈 시뮬레이터 서버 시작 (IP: 0.0.0.0, Port: {self.port})")

        self.running = True
        self.sim_thread = Thread(target=self._simulation_logic, daemon=True)
        self.sim_thread.start()
        print("[SIMULATOR] IO 모듈 자동 응답 로직 스레드 가동 완료")

    def stop(self):
        self.running = False
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)
        if self.server:
            self.server.stop()
        print("[SIMULATOR] IO 모듈 시뮬레이터 종료")

    def _bits_to_word(self, bit_list):
        high_8 = bit_list[0:8][::-1]
        low_8 = bit_list[8:16][::-1]
        combined = high_8 + low_8
        return int("".join(str(b) for b in combined), 2)

    def _word_to_bits(self, word_val):
        temp = bin(word_val)[2:].zfill(16)
        di = [int(c) for c in temp]
        high_8 = di[0:8][::-1]
        low_8 = di[8:16][::-1]
        return high_8 + low_8

    def _simulation_logic(self):
        address = 0x07D0  # 2000번지

        while self.running:
            try:
                # ⭐ server 인스턴스 내의 data_bank 객체를 수신하여 참조
                if not self.server or not hasattr(self.server, "data_bank"):
                    time.sleep(0.05)
                    continue

                db = self.server.data_bank
                holding_regs = db.get_holding_registers(address, 1)

                if not holding_regs:
                    time.sleep(0.05)
                    continue

                do_word = holding_regs[0]
                do_bits = self._word_to_bits(do_word)

                io_map = {0:7, 1:6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:0,
                          8:15, 9:14, 10:13, 11:12, 12:11, 13:10, 14:9, 15:8}

                vacuum_do = do_bits[io_map[0]]
                bin_on_do = do_bits[io_map[2]]
                bin_off_do = do_bits[io_map[3]]

                di_bits = [0] * 16

                # 吸着 ON 시 -> 흡착 감지(0번) & 흡착 성공(1번) 센서 자동 ON
                if vacuum_do == 1:
                    di_bits[io_map[0]] = 1
                    di_bits[io_map[1]] = 1
                else:
                    di_bits[io_map[0]] = 0
                    di_bits[io_map[1]] = 0

                # 실린더 동작 시 센서 시뮬레이션
                if bin_on_do == 1 and bin_off_do == 0:
                    di_bits[io_map[2]] = 1
                    di_bits[io_map[3]] = 0
                elif bin_off_do == 1 and bin_on_do == 0:
                    di_bits[io_map[2]] = 0
                    di_bits[io_map[3]] = 1

                di_bits[io_map[8]] = 0 

                di_word = self._bits_to_word(di_bits)
                db.set_input_registers(address, [di_word])

            except Exception as e:
                print(f"[SIMULATOR ERROR] 시뮬레이션 루프 예외: {e}")

            time.sleep(0.05)


if __name__ == "__main__":
    server = IOSimulatorServer(host="0.0.0.0", port=502)
    server.start()
    
    print("[INFO] IO 모듈 시뮬레이터가 실행중입니다. 종료하려면 Ctrl+C를 누르세요.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        server.stop()
        print("[INFO] 시뮬레이터가 안전하게 종료되었습니다.")