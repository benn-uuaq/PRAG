import time
from PyQt5.QtCore import *
from pyModbusTCP.client import ModbusClient
from threading import Lock
from queue import Queue


class IO_Module_Class(QThread):

    def __init__(self,IP,Port):
        super().__init__()
        
        self.modbus_lock = Lock()
        self.Send_que = Queue()
        self.stop_signal=False

        #   IO모듈 정보
        self.IO_Module=ModbusClient(host=IP,port=Port,timeout=2)
        print("\n",end="")
        print("IO모듈 IP 및 Port:",self.IO_Module)

        #   IO주소
        self.Input_Address=0x07D0
        self.Output_Address=0x07D0

        self.IO_map={0:7, 1:6, 2:5, 3:4, 4:3, 5:2, 6:1, 7:0,
                     8:15, 9:14, 10:13, 11:12, 12:11, 13:10, 14:9, 15:8}
        
        # Input번호
        self.vacuum_ON_serson=0
        self.vacuum_Success_serson=1
        self.BIN_cylinder_ON_serson=2
        self.BIN_cylinder_OFF_serson=3
        self.emergency_stop=10
        
        
        # Output 번호
        self.vacuum=0
        self.blow=1
        
        self.BIN_cylinder_ON=2
        self.BIN_cylinder_OFF=3
        self.unit_reset=10
        self.lamp_red=12
        self.lamp_yellow=13
        self.lamp_green=14
        self.lamp_buzzer=15

        
        #흡착 동작 변수
        self.vacuum_run=0
        self.vacuum_step=0
        
        #파기 동작 변수
        self.blow_run=0
        self.blow_step=0

        #BIN 실린더 동작 변수
        self.BIN_cylinder_run=0
        self.BIN_cylinder_step=0

        #유닛 리셋 동작 변수
        self.unit_reset_run=0
        self.unit_reset_step=0

        #타워램프 동작 변수
        self.lamp_run=[0,0,0]
        self.lamp_step=[0,0,0]
        self.tower_lamp=[self.lamp_red,self.lamp_yellow,self.lamp_green]
        
        self.buzzer_run=0
        self.buzzer_step=0


        
    def run(self):
        while not self.stop_signal:
            count=1
            self.connection_status=False
            while not self.connection_status:
                self.connection_status=self.IO_Module.open()
                print("\n",end="")
                print("IO모듈 연결 상태: {0}, 연결 시도 횟수: {1}".format(self.connection_status,count))
                if self.connection_status:
                    print("\n",end="")
                    print("IO모듈 연결 성공")
                    time.sleep(0.01)
                    with self.modbus_lock:
                        feedback=self.IO_Module.write_multiple_registers(self.Output_Address,[0,0])
                else:
                    count+=1
                    time.sleep(0.01)
            
            #HSK LOOP################################################################################################3
            while not self.stop_signal:
                self.Read_DI = self.Read_Input()
                # print(f"DI: {self.Read_DI}")
                self.Read_DO = self.Read_Output()
                # print(f"DO: {self.Read_DO}")
                self.send_do=list(map(int,bin(self.DO_data[0])[2:].zfill(16)))
                
                #####흡착 On/Off
                if self.vacuum_run:
                    if self.vacuum_step==1:
                        self.send_do[self.IO_map[self.vacuum]] = 1
                        self.vacuum_run=0
                        self.vacuum_step=0
                    elif self.vacuum_step==2:
                        self.send_do[self.IO_map[self.vacuum]] = 0
                        self.vacuum_run=0
                        self.vacuum_step=0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))
                    
                #####파기 On/Off
                if self.blow_run:
                    if self.blow_step==1:
                        self.send_do[self.IO_map[self.blow]] = 1
                        self.blow_run=0
                        self.blow_step=0
                    elif self.blow_step==2:
                        self.send_do[self.IO_map[self.blow]] = 0
                        self.blow_run=0
                        self.blow_step=0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))
                
                
                #####BIN 실린더 전진/후진
                if self.BIN_cylinder_run:
                    if self.BIN_cylinder_step==1:
                        self.send_do[self.IO_map[self.BIN_cylinder_ON]] = 1
                        self.send_do[self.IO_map[self.BIN_cylinder_OFF]] = 0
                        self.BIN_cylinder_run=0
                        self.BIN_cylinder_step=0
                    elif self.BIN_cylinder_step==2:
                        self.send_do[self.IO_map[self.BIN_cylinder_ON]] = 0
                        self.send_do[self.IO_map[self.BIN_cylinder_OFF]] = 1
                        self.BIN_cylinder_run=0
                        self.BIN_cylinder_step=0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))

                
                #####Relay Unit Reset 신호
                if self.unit_reset_run:
                    if self.unit_reset_step==1:
                        self.send_do[self.IO_map[self.unit_reset]] = 1
                        self.unit_reset_run=0
                        self.unit_reset_step=0
                    elif self.unit_reset_step==2:
                        self.send_do[self.IO_map[self.unit_reset]] = 0
                        self.unit_reset_run=0
                        self.unit_reset_step=0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))
                    
                if self.buzzer_run:
                    if self.buzzer_step==1:
                        self.send_do[self.IO_map[self.lamp_buzzer]] = 1
                        self.buzzer_run=0
                        self.buzzer_step=0
                    elif self.buzzer_step==2:
                        self.send_do[self.IO_map[self.lamp_buzzer]] = 0
                        self.buzzer_run=0
                        self.buzzer_step=0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))
                

                #####타워 램프 모듈 동작
                if 1 in self.lamp_run:
                    for i in range(3):
                        if self.lamp_run[i]==1:
                            if self.lamp_step[i] == 1:
                                self.send_do[self.IO_map[self.tower_lamp[i]]] = 1
                                self.lamp_run[i] = 0
                                self.lamp_step[i] = 0
                            elif self.lamp_step[i] == 2:
                                self.send_do[self.IO_map[self.tower_lamp[i]]] = 0
                                self.lamp_run[i] = 0
                                self.lamp_step[i] = 0
                    self.Send_que.put(self.Write_DO_Date(self.Output_Address,self.send_do))
                    
                                                 
                # 명령어 전송 큐~~#######################################################                
                while not self.Send_que.empty():
                    func = self.Send_que.get()
                    if callable(func):
                        func()
                
                #기준틱 용 지연 맨마지막에 의미없음
                time.sleep(0.1)

        self.IO_Module.write_multiple_registers(self.Output_Address,[0])
        self.stop_signal=False
                         
    def stop(self):
        self.stop_signal=True
        self.wait()
        
    def Read_Input_Data(self):
        return self.Read_DI
    
    def Read_Output_Data(self):
        return self.Read_DO

    def Read_Input(self):
        self.DI_data=self.IO_Module.read_input_registers(self.Input_Address,1)
        di = [[0] * 16 for _ in range(1)]
        if self.DI_data!=None:
            for i in range(len(self.DI_data)):
                temp = bin(self.DI_data[i])[2:].zfill(16)
                for j in range(16):
                    di[i][j] = int(temp[j])
                high_8bit=[di[i][j] for j in range(0,8)][::-1]
                low_8bit=[di[i][j] for j in range(8,16)][::-1]
                di[i]=high_8bit+low_8bit
            return di[0]
        else:
            return None
        
    def Read_Output(self):
        self.DO_data=self.IO_Module.read_holding_registers(self.Input_Address,1)
        do = [[0] * 16 for _ in range(1)]
        if self.DO_data!=None:
            for i in range(len(self.DO_data)):
                temp = bin(self.DO_data[i])[2:].zfill(16)
                for j in range(16):
                    do[i][j] = int(temp[j])
                high_8bit=[do[i][j] for j in range(0,8)][::-1]
                low_8bit=[do[i][j] for j in range(8,16)][::-1]
                do[i]=high_8bit+low_8bit
            return do[0]
        else:
            return None
        
    def Write_DO_Date(self, address, data):
        def task():
            send_data = int("".join(str(b) for b in data), 2)
            self.IO_Module.write_single_register(address, send_data)
        return task
    
    
    def Vacuum(self, data):
        if data == "ON":
            if self.vacuum_run == 0:
                self.vacuum_run = 1
                self.vacuum_step = 1
        elif data == "OFF":
            if self.vacuum_run == 0:
                self.vacuum_run = 1
                self.vacuum_step = 2
                
    def Blow(self, data):
        if data == "ON":
            if self.blow_run == 0:
                self.blow_run = 1
                self.blow_step = 1
        elif data == "OFF":
            if self.blow_run == 0:
                self.blow_run = 1
                self.blow_step = 2
                
    def Buzzer(self, data):
        if data == "ON":
            if self.buzzer_run == 0:
                self.buzzer_run = 1
                self.buzzer_step = 1
        elif data == "OFF":
            if self.buzzer_run == 0:
                self.buzzer_run = 1
                self.buzzer_step = 2
        
    
    def Gripper(self, data):

        '''
        data= 명령

        명령: "ON"(흡착), "OFF"(파기)

        설명: 그리퍼 흡착/파기
        '''

        if data == "ON":
            if self.gripper_run == 0:
                self.gripper_run = 1
                self.gripper_step = 1
        elif data == "OFF":
            if self.gripper_run == 0:
                self.gripper_run = 1
                self.gripper_step = 2

    def BIN_Cylinder(self, data):

        '''
        data= 명령

        명령: "ON"(전진), "OFF"(후진)

        설명: BIN 실린더 전진/후진
        '''

        if data == "ON":
            if self.BIN_cylinder_run == 0:
                self.BIN_cylinder_run = 1
                self.BIN_cylinder_step = 1
        elif data == "OFF":
            if self.BIN_cylinder_run == 0:
                self.BIN_cylinder_run = 1
                self.BIN_cylinder_step = 2

    def Unit_Reset(self, data):

        '''
        data= 명령

        명령: "ON", "OFF"

        설명: Relay Unit Reset 신호
        '''

        if data == "ON":
            if self.unit_reset_run == 0:
                self.unit_reset_run = 1
                self.unit_reset_step = 1
        elif data == "OFF":
            if self.unit_reset_run == 0:
                self.unit_reset_run = 1
                self.unit_reset_step = 2
            

    def Tower_Lamp(self, data):

        '''
        data= [명령,명령,명령,명령] -> 적색, 녹색, 황색, 부저

        명령: "ON"(켜기), "OFF"(끄기)

        설명: 타워 램프 키고 끄기
        '''

        if data[0] == "ON":
            if self.lamp_run[0] == 0:
                self.lamp_run[0] = 1
                self.lamp_step[0] = 1
        elif data[0] == "OFF":
            if self.lamp_run[0] == 0:
                self.lamp_run[0] = 1
                self.lamp_step[0] = 2

        if data[1] == "ON":
            if self.lamp_run[1] == 0:
                self.lamp_run[1] = 1
                self.lamp_step[1] = 1
        elif data[1] == "OFF":
            if self.lamp_run[1] == 0:
                self.lamp_run[1] = 1
                self.lamp_step[1] = 2

        if data[2] == "ON":
            if self.lamp_run[2] == 0:
                self.lamp_run[2] = 1
                self.lamp_step[2] = 1
        elif data[2] == "OFF":
            if self.lamp_run[2] == 0:
                self.lamp_run[2] = 1
                self.lamp_step[2] = 2

