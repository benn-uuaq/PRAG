import pandas
import struct

class RobotHeader():
    __slots__ = ['type', 'size',]
    @staticmethod
    def unpack(buf):
        rmd = RobotHeader()
        (rmd.size, rmd.type) = struct.unpack_from('>iB', buf)
        return rmd

class RobotDataConfig():
    __slots__ = ['fmt', 'name']
    
    @staticmethod
    def __type_to_pack(vartype : str):
        if vartype == 'int' or vartype == 'int32_t':
            return 'i'
        elif vartype == 'int8_t' :
            return 'b'
        elif vartype == 'uint8_t':
            return 'B'
        elif vartype == 'uint64_t':
            return'Q'
        elif vartype == 'bool':
            return '?'
        elif vartype == 'double':
            return 'd'
        elif vartype == 'uint32_t':
            return 'I'
        elif vartype == 'float':
            return 'f'
        elif vartype == 'uint32_t':
            return 'I'
        else:
            raise("Unkow Type")

    @staticmethod
    def get_config(file, sheet):
        config = RobotDataConfig()
        excel = pandas.read_excel(file, sheet_name=sheet)
        config.fmt = '>'
        config.name = []
        is_foreach = False
        temp_fmt = ''
        temp_name = []
        elite_internel_count = 0
        for i in range(len(excel['type'])):
            if type(excel['name'][i]) == float:
                pass
            if type(excel['type'][i]) == str:
                excel['type'][i].replace(" ", "")
            if excel['type'][i] == 'bytes':
                config.fmt += 'B' * excel['bytes'][i]
                for j in range(excel['bytes'][i]):
                    config.name.append(excel['name'][i] + '_' +str(j) + '_' + str(elite_internel_count))
                    elite_internel_count += 1
            elif excel['type'][i] == 'foreach':
                is_foreach = True
            elif excel['type'][i] == 'end' and is_foreach == True:
                config.fmt += (temp_fmt * 6)
                for j in range(6):
                    for k in temp_name:
                        config.name.append(k + str(j))
                temp_fmt = ''
                temp_name = []
                is_foreach = False
            else:
                if is_foreach:
                    temp_name.append(excel['name'][i])
                    temp_fmt += RobotDataConfig.__type_to_pack(excel['type'][i])
                else:
                    config.name.append(excel['name'][i])
                    config.fmt += RobotDataConfig.__type_to_pack(excel['type'][i])
        return config

class RobotData():
    @staticmethod
    def unpack(buf, config : RobotDataConfig):
        data = RobotData()
        unpack = struct.unpack_from(config.fmt, buf)
        for i in range(len(config.name)):
            name = config.name[i]
            data.__dict__[name] = unpack[i]
        return data