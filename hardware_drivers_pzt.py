
import clr
import time
import sys
import copy
clr.AddReference('System')
from System import Decimal 
from System.Threading import Thread
from logger11 import get_logger
logger = get_logger(__name__)

# 添加必要的DLL引用
clr.AddReference('System')
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.DeviceManagerCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\Thorlabs.MotionControl.GenericPiezoCLI.dll")
clr.AddReference("C:\\Program Files\\Thorlabs\\Kinesis\\ThorLabs.MotionControl.Benchtop.PiezoCLI.dll")
# from System.Threading import Thread
from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
from Thorlabs.MotionControl.Benchtop.PiezoCLI import BenchtopPiezo
# from Thorlabs.MotionControl.GenericPiezoCLI import PiezoControlModeTypes
from Thorlabs.MotionControl.GenericPiezoCLI.Piezo import PiezoControlModeTypes

def zero_channels(device, controller_name):
    """安全归零并初始化所有通道，根据控制器类型选择通道数"""
    # 根据控制器名称确定要调零的通道
    if "位置" in controller_name:  # 位置控制器（控制器1和控制器3）
        channels = [1, 2, 3]  # 三个通道都调零
    else:  # 角度控制器（控制器2和控制器4）
        channels = [1, 2]  # 只调零前两个通道
        
    success = True

    try:
        for channel_number in channels:
            channel = device.GetChannel(channel_number)
            if channel is None:
                print(f"通道 {channel_number} 不存在")
                success = False
                continue

            # === 初始化设置 ===
            if not channel.IsSettingsInitialized():
                try:
                    channel.WaitForSettingsInitialized(5000)
                except Exception as ex:
                    print(f"通道 {channel_number} 设置初始化失败: {ex}")
                    success = False
                    continue

            # === 执行归零 ===
            try:
                print(f"正在归零通道 {channel_number}...")
                channel.SetZero()  # 执行硬件归零
                
                # 等待归零完成
                Thread.Sleep(1000)
            except Exception as ex:
                print(f"通道 {channel_number} 归零失败: {ex}")
                success = False
                continue
            # === 停止轮询 ===
            finally:
                channel.StopPolling()

    except Exception as ex:
        print(f"全局错误: {ex}")
        success = False

    # === 最终状态检查 ===
    if success:
        print(f"{controller_name} 所有通道已归零并初始化完成")
    else:
        print(f"{controller_name} 部分通道操作未完成，请检查日志")
    
    return success
def mode_change_test(channel,mode) :
    """测试模式切换功能"""
    try:
        if mode == 1:
            channel.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
        elif mode == 2:
            channel.SetPositionControlMode(PiezoControlModeTypes.CloseLoop)
        new_mode = channel.GetPositionControlMode()
        print(f"新模式: {new_mode}")

        return True
    except Exception as e:
        print(f"模式切换测试失败: {e}")
        return False
def set_piezo_voltage(channel, voltage) -> bool:
    """
    安全快速设置压电通道输出电压
    返回: True表示成功，False表示失败
    """
    try:
        # 动态获取最大允许电压
        max_voltage = channel.GetMaxOutputVoltage()
        # print(f"检测到执行器最大允许电压: {max_voltage}V")

        # 确保电压是 System.Decimal 类型
        if not isinstance(voltage, Decimal):
            try:
                # 转换为 System.Decimal
                voltage = Decimal(float(voltage))
                # print(f"已将电压值转换为 System.Decimal: {voltage}")
            except:
                # print(f"无法将电压值 {voltage} 转换为 System.Decimal")
                return False
        # mode = PiezoControlModeTypes.OpenLoop  # 定义目标模式为开环
        # # mode_test = channel.GetPositionControlMode()  # 正确调用方法获取当前模式
        # # print(mode_test)  # 打印当前模式（用于调试）

        # # 错误的条件判断：比较方法对象而不是方法返回值
        # if channel.GetPositionControlMode != mode:
        #     print("当前为闭环模式，正在切换至开环模式...")
        #     channel.SetPositionControlMode(PiezoControlModeTypes.OpenLoop)
        #     # time.sleep(0.5)

        # 设置电压
        channel.SetOutputVoltage(voltage)
        time.sleep(0.1)  # 确保命令被处理
        print(f"已设置电压: {voltage}V")
        return True  # 成功

    except Exception as e:
        print(f"电压设置失败: {e}")
        return False  # 失败
def map_value_to_voltage(value, val_min, val_max, volt_max=75.0):
    """将输入值线性映射到电压范围，返回 System.Decimal 类型"""
    value_range = val_max - val_min
    voltage = (value - val_min) / value_range * volt_max
    # 返回 System.Decimal 类型
    return Decimal(voltage)

# def get_position_from_voltage(voltage, val_min, val_max, volt_max=75.0):
#     """将电压值线性映射回位置范围"""
#     # 将System.Decimal转换为Python float
#     volt_py = float(str(voltage))  # 通过字符串转换
#     volt_max_float = float(volt_max)
#     val_min_float = float(val_min)
#     val_max_float = float(val_max)
    
#     # 使用Python float进行计算
#     position = val_min_float + (volt_py / volt_max_float) * (val_max_float - val_min_float)
#     return position

class PiezoController:
    def __init__(self, controller_name, serial_no):
        self.controller_name = controller_name
        self.serial_no = serial_no
        self.device = None
        self.channels = {
            1: None,  # 通道1
            2: None,  # 通道2
            3: None   # 通道3
        }
        self.ranges = {
            'x': (0, 30),
            'y': (0, 30),
            'z': (0, 30),
            'rx': (0, 0.03),
            'ry': (0, 0.03),
            'bx': (0, 30),
            'by': (0, 30),
            'bz': (0, 30),
            'brx': (0, 0.03),
            'bry': (0, 0.03)
        }
        self.is_connected = False
        self.is_zeroed = False
        self.initial_positions = {}  # 添加初始位置存储

    def connect(self):
        """连接压电控制器并初始化通道"""
        try:
            logger.info(f"开始连接PZT控制器 {self.controller_name} ({self.serial_no})...")
            
            # 构建设备列表
            logger.info("正在构建设备列表...")
            DeviceManagerCLI.BuildDeviceList()
            logger.info("设备列表构建完成")
            
            # 连接设备
            logger.info(f"正在创建BenchtopPiezo实例...")
            self.device = BenchtopPiezo.CreateBenchtopPiezo(self.serial_no)
            logger.info(f"正在连接设备 {self.serial_no}...")
            self.device.Connect(self.serial_no)
            logger.info("设备连接成功")
            
            # 初始化所有通道
            for ch_num in [1, 2, 3]:
                logger.info(f"正在初始化通道 {ch_num}...")
                if ch_num in self.channels:
                    channel = self.device.GetChannel(ch_num)
                    self.channels[ch_num] = channel
                    
                    # 确保设置初始化
                    if not channel.IsSettingsInitialized():
                        logger.info(f"等待通道 {ch_num} 设置初始化...")
                        try:
                            if not channel.WaitForSettingsInitialized(10000):
                                logger.warning(f"通道 {ch_num} 设置初始化超时")
                                continue
                            logger.info(f"通道 {ch_num} 设置初始化完成")
                        except Exception as e:
                            logger.error(f"通道 {ch_num} 设置初始化错误: {e}")
                            continue
                    
                    # 启动轮询并启用设备
                    logger.info(f"启动通道 {ch_num} 轮询...")
                    channel.StartPolling(250)
                    time.sleep(0.25)
                    logger.info(f"启用通道 {ch_num}...")
                    channel.EnableDevice()
                    time.sleep(0.25)
                    logger.info(f"通道 {ch_num} 启用完成")
            
            self.is_connected = True
            logger.info(f"{self.controller_name} ({self.serial_no}) 已连接并初始化")
            return True
        except Exception as e:
            logger.error(f"连接失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_connected = False
            return False
    def zero(self):
        """执行调零操作"""
        if not self.is_connected:
            print("设备未连接，无法调零")
            return False
        
        try:
            # 传递控制器名称给 zero_channels 函数
            success = zero_channels(self.device, self.controller_name)
            if success:
                self.is_zeroed = True
                print(f"{self.controller_name} 调零成功")
            return success
        except Exception as e:
            print(f"调零失败: {str(e)}")
            return False

    def set_initial_position(self, position_dict):
        """设置初始位置"""
        self.initial_positions = position_dict.copy()
        print(f"{self.controller_name} 初始位置已设置: {self.initial_positions}")
        
    def back_to_initial_position(self):
        """回归到初始位置（使用参数设置界面中的初始位置）"""
        if not self.is_connected:
            print("设备未连接，无法回归初始位置")
            return False
        
        if not self.initial_positions:
            print("未设置初始位置，无法回归")
            print(self.initial_positions)
            return False
        
        try:
            # 根据控制器类型决定要操作的轴
            if "A端" in self.controller_name:
                if "位置" in self.controller_name:  # 位置控制器
                    axes_to_set = ['x', 'y', 'z']
                else:  # 角度控制器
                    axes_to_set = ['rx', 'ry']
            else:  # B端控制器
                if "位置" in self.controller_name:  # 位置控制器
                    axes_to_set = ['bx', 'by', 'bz']
                else:  # 角度控制器
                    axes_to_set = ['brx', 'bry']
            
            # 提取该控制器负责的初始位置
            controller_initial_pos = {}
            for axis in axes_to_set:
                if axis in self.initial_positions:
                    controller_initial_pos[axis] = self.initial_positions[axis]
            
            if controller_initial_pos:
                return self.set_position(controller_initial_pos)
            else:
                print(f"{self.controller_name} 没有需要设置的初始位置")
                return True
                
        except Exception as e:
            print(f"回归初始位置失败: {str(e)}")
            return False
    def mode_change(self,mode,channels):
        """测试模式切换功能"""
        if not self.is_connected:
            print("设备未连接，无法切换模式")
            return False
        
        try:
            # 仅测试第一个通道的模式切换
            for ch_num in channels:
                
                channel = self.channels.get(ch_num)
                if not channel:
                    print(f"{self.controller_name} 通道 {ch_num} 未初始化")
                    return False
                
                if not mode_change_test(channel,mode):
                    print(f"模式切换失败在通道 {ch_num}")
                # time.sleep(0.5)
                    return False
            
            return True
        except Exception as e:
            print(f"模式切换测试失败: {str(e)}")
            return False
    def set_position(self, position_dict):
        """设置位置参数到对应的控制器通道，并等待直到到达目标位置"""
        if not self.is_connected:
            print("设备未连接，无法设置位置")
            return False

        target_positions = {}
        success = True  # 跟踪所有设置是否成功

        for axis, value in position_dict.items():
            if axis not in self.ranges:
                print(f"警告: 未知轴 '{axis}'，跳过")
                continue
                
            # 获取轴范围
            val_min, val_max = self.ranges[axis]
            # print(f"{self.controller_name} 轴 {axis} 范围: {val_min} - {val_max}")
            # 映射位置值到电压
            voltage = map_value_to_voltage(value, val_min, val_max)
            # print(f"{self.controller_name} 轴 {axis} 目标值: {value} 映射电压: {voltage}V")
            
            
            if axis in ['x', 'y', 'z']:
                ch_num = {'x': 1, 'y': 2, 'z': 3}.get(axis)
            elif axis in ['rx', 'ry']:
                ch_num = {'rx': 1, 'ry': 2}.get(axis)
            elif axis in ['bx', 'by', 'bz']:
                ch_num = {'bx': 1, 'by': 2, 'bz': 3}.get(axis)
            elif axis in ['brx', 'bry']:
                ch_num = {'brx': 1, 'bry': 2}.get(axis)  # 或者您需要的其他通道号
            else:
                print(f"错误: 无法为轴 '{axis}' 分配通道")
                success = False
                continue
            channel = self.channels.get(ch_num)
            
            if not channel:
                print(f"错误: {self.controller_name} 通道 {ch_num} 未初始化")
                success = False
                continue
                
            print(f"{self.controller_name} 设置 {axis} 到 {value} (电压: {voltage}V)")
            
            # 设置电压，并检查是否成功
            if not set_piezo_voltage(channel, voltage):
                print(f"设置 {axis} 的电压失败")
                success = False
                continue
            
            # 记录目标位置
            target_positions[axis] = (value, ch_num, val_min, val_max)
        
        # 如果设置电压时有失败，直接返回False
        if not success:
            print(f"{self.controller_name} 部分轴设置失败，无法继续")
            return False
        print(f"{self.controller_name} 所有轴已到达目标位置")
        return True
        # # 等待位置到达
        # start_time = time.time()
        # all_reached = False
        
        # # while time.time() - start_time < timeout:
        # #     all_reached = True
        # #     for axis, (target_value, ch_num, val_min, val_max) in target_positions.items():
        # #         # 获取当前电压
        # #         current_voltage = self.channels[ch_num].GetOutputVoltage()
        # #         # 转换回位置
        # #         current_position = get_position_from_voltage(current_voltage, val_min, val_max)
                
        # #         # 检查是否在容差范围内
        # #         if abs(current_position - target_value) > tolerance:
        # #             all_reached = False
        # #             break
                
        #     # if all_reached:
        #     #     print(f"{self.controller_name} 所有轴已到达目标位置")
        #     #     return True
            
        #     # time.sleep(0.01)  # 10ms检查一次
        
        # print(f"{self.controller_name} 部分轴未能在超时时间内到达目标位置")
        # return False

    def disconnect(self):
        """断开设备连接"""
        if not self.is_connected:
            return True
        
        try:
            for ch_num in [1, 2, 3]:
                try:
                    if ch_num in self.channels and self.channels[ch_num]:
                        channel = self.channels[ch_num]
                        channel.StopPolling()
                        channel.DisableDevice()
                except Exception as e:
                    print(f"断开 {self.controller_name} 通道 {ch_num} 时出错: {str(e)}")
            
            if self.device:
                self.device.Disconnect()
                print(f"{self.controller_name} ({self.serial_no}) 已断开")
            
            self.is_connected = False
            return True
        except Exception as e:
            print(f"断开连接失败: {str(e)}")
            return False
