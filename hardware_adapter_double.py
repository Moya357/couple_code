from typing import Dict, List, Callable, Optional
from core_abstract import IHardwareController
from device_manager_double import GlobalDeviceManager
from thread_manager import ThreadManager
from PowerMeter import get_power_meter
import queue
import time

class HardwareAdapter(IHardwareController):
    """硬件控制适配器"""
    
    def __init__(self, mode="single", thread_manager: ThreadManager = None,
                 progress_callback: Optional[Callable] = None,
                 finished_callback: Optional[Callable] = None):
        self.mode = mode
        self.device_manager = GlobalDeviceManager()
        self.thread_manager = thread_manager or ThreadManager()
        self.initial_positions = {}
        self._position_queue = queue.Queue()
        self._power_queue = queue.Queue()
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.debug_mode = False  # 调试模式开关
    
    def set_callbacks(self, progress_callback: Callable, finished_callback: Callable):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
    
    def measure_power(self, position: Dict[str, float]) -> float:
        """测量功率 - 直接调用硬件控制器功能"""
        # 直接设置位置
        print("测量功率，设置位置:", position)
        if not self.set_position(position):
            print("设置位置失败，无法进行功率测量")
            return 0.0
        
        # 等待位置稳定（根据实际情况调整等待时间）
        time.sleep(1.2)
        
        try:
            # 直接调用功率计进行测量
            power_meter = self.device_manager.get_power_meter()
            result = power_meter.measure_power_fast()
            
            # 处理功率计返回的字典格式
            if isinstance(result, dict):
                power_value = result.get("power", 0.0)
                if self.debug_mode:
                    engineering_notation = result.get("engineering_notation", "")
                    print(f"功率测量结果: {engineering_notation}")
                return power_value
            else:
                # 兼容旧版本：直接返回数值
                return result
        except Exception as e:
            print(f"功率测量失败: {str(e)}")
            return 0.0
    
    def measure_power_average(self, position: Dict[str, float]) -> float:
        """测量功率 - 直接调用硬件控制器功能"""
        # 直接设置位置
        if not self.set_position(position):
            print("设置位置失败，无法进行功率测量")
            return 0.0
        
        # 等待位置稳定（根据实际情况调整等待时间）
        time.sleep(0.8)
        
        try:
            # 直接调用功率计进行测量
            power_meter = self.device_manager.get_power_meter()
            result = power_meter.measure_power(samples=5)
            
            # 处理功率计返回的字典格式
            if isinstance(result, dict):
                power_value = result.get("power", 0.0)
                if self.debug_mode:
                    engineering_notation = result.get("engineering_notation", "")
                    print(f"平均功率测量结果: {engineering_notation}")
                return power_value
            else:
                # 兼容旧版本：直接返回数值
                return result
        except Exception as e:
            print(f"功率测量失败: {str(e)}")
            return 0.0
    
    def measure_current_power(self):
        """
        测量当前功率（不移动位置）
        
        返回:
            power: 当前功率值
        """
        try:
            # 使用功率计测量当前功率
            power_meter = self.device_manager.get_power_meter()
            power_result = power_meter.measure_power_fast()
            
            # 处理功率计返回的字典格式
            if isinstance(power_result, dict):
                power_value = power_result.get("power", 0.0)
                # 可选：记录工程单位显示用于调试
                if self.debug_mode:
                    engineering_notation = power_result.get("engineering_notation", "")
                    print(f"当前功率: {engineering_notation}")
                return power_value
            else:
                # 兼容旧版本：直接返回数值
                return power_result
        except Exception as e:
            print(f"测量功率失败: {e}")
            return 0.0
    
    def get_power_value(self, power_result):
        """
        从功率计返回结果中提取功率值
        支持新旧两种格式
        
        参数:
            power_result: 功率计返回的结果，可能是字典或数值
            
        返回:
            float: 提取的功率值
        """
        if power_result is None:
            return 0.0
            
        if isinstance(power_result, dict):
            # 新格式：字典包含功率值和其他信息
            power_value = power_result.get("power", 0.0)
            
            # 可选：记录详细的功率信息
            if self.debug_mode:
                engineering_notation = power_result.get("engineering_notation", "N/A")
                scientific_notation = power_result.get("scientific_notation", "N/A")
                print(f"功率详情: {engineering_notation} ({scientific_notation})")
                
            return power_value
        else:
            # 旧格式：直接返回功率数值
            return float(power_result)
    
    def mode_switch(self, mode) -> bool:
        """切换模式"""
        success = True
        
        # 设置A端位置控制器
        a_pos_controller = self.device_manager.get_pzt_controller("A端位置控制器")
        if a_pos_controller:
            channels = [1, 2, 3]  # A端位置控制器有3个通道
            if not a_pos_controller.mode_change(mode, channels):
                print("设置A端位置控制器模式失败")
                success = False
        
        # 设置A端角度控制器
        a_angle_controller = self.device_manager.get_pzt_controller("A端角度控制器")
        if a_angle_controller:
            channels = [1, 2]  # A端角度控制器有2个通道
            if not a_angle_controller.mode_change(mode, channels):
                print("设置A端角度控制器模式失败")
                success = False
        
        # 双端模式下设置B端控制器
        if self.mode == "dual":
            # 设置B端位置控制器
            b_pos_controller = self.device_manager.get_pzt_controller("B端位置控制器")
            if b_pos_controller:
                channels = [1, 2, 3]  # B端位置控制器有3个通道
                if not b_pos_controller.mode_change(mode, channels):
                    print("设置B端位置控制器模式失败")
                    success = False
            
            # 设置B端角度控制器
            b_angle_controller = self.device_manager.get_pzt_controller("B端角度控制器")
            if b_angle_controller:
                channels = [1, 2]  # B端角度控制器有2个通道
                if not b_angle_controller.mode_change(mode, channels):
                    print("设置B端角度控制器模式失败")
                    success = False
        
        return success
    
    def set_position(self, position: Dict[str, float]) -> bool:
        """设置位置 - 直接通过PZT控制器实现"""
        # 将位置参数转换为控制器可理解的格式
        position_dict = self._convert_state_to_position(position)
        
        # 根据控制器类型拆分位置参数
        success = True
        
        # 设置A端位置控制器
        a_pos_controller = self.device_manager.get_pzt_controller("A端位置控制器")
        if a_pos_controller:
            a_pos = {k: v for k, v in position_dict.items() if k in ['x', 'y', 'z']}
            if a_pos and not a_pos_controller.set_position(a_pos):
                print("设置A端位置失败")
                success = False
        
        # 设置A端角度控制器
        a_angle_controller = self.device_manager.get_pzt_controller("A端角度控制器")
        if a_angle_controller:
            a_angle = {k: v for k, v in position_dict.items() if k in ['rx', 'ry']}
            if a_angle and not a_angle_controller.set_position(a_angle):
                print("设置A端角度失败")
                success = False
        
        # 双端模式下设置B端控制器
        if self.mode == "dual":
            # 设置B端位置控制器
            b_pos_controller = self.device_manager.get_pzt_controller("B端位置控制器")
            if b_pos_controller:
                b_pos = {k: v for k, v in position_dict.items() if k in ['bx', 'by', 'bz']}
                print("设置B端位置:", b_pos)
                if b_pos and not b_pos_controller.set_position(b_pos):
                    print("设置B端位置失败")
                    success = False
            
            # 设置B端角度控制器
            b_angle_controller = self.device_manager.get_pzt_controller("B端角度控制器")
            if b_angle_controller:
                b_angle = {k: v for k, v in position_dict.items() if k in ['brx', 'bry']}
                print("设置B端角度:", b_angle)
                if b_angle and not b_angle_controller.set_position(b_angle):
                    print("设置B端角度失败")
                    success = False
        
        return success
    
    def set_initial_positions(self, positions):
        """设置所有控制器的初始位置 - 改进版本"""
        # 先进行坐标转换
        converted_positions = self._convert_state_to_position(positions)
        self.initial_positions = converted_positions
        
        print(f"设置初始位置 - 转换前: {positions}")
        print(f"设置初始位置 - 转换后: {converted_positions}")
        
        # 设置A端位置控制器的初始位置
        a_pos_controller = self.device_manager.get_pzt_controller("A端位置控制器")
        if a_pos_controller:
            a_pos = {k: v for k, v in converted_positions.items() if k in ['x', 'y', 'z']}
            print("设置A端位置初始位置:", a_pos)
            a_pos_controller.set_initial_position(a_pos)
        
        # 设置A端角度控制器的初始位置
        a_angle_controller = self.device_manager.get_pzt_controller("A端角度控制器")
        if a_angle_controller:
            a_angle = {k: v for k, v in converted_positions.items() if k in ['rx', 'ry']}
            print("设置A端角度初始位置:", a_angle)
            a_angle_controller.set_initial_position(a_angle)
        
        # 如果是双端模式，设置B端控制器的初始位置
        if self.mode == "dual":
            b_pos_controller = self.device_manager.get_pzt_controller("B端位置控制器")
            if b_pos_controller:
                b_pos = {k: v for k, v in converted_positions.items() if k in ['bx', 'by', 'bz']}
                print("设置B端位置初始位置:", b_pos)
                b_pos_controller.set_initial_position(b_pos)
            
            b_angle_controller = self.device_manager.get_pzt_controller("B端角度控制器")
            if b_angle_controller:
                b_angle = {k: v for k, v in converted_positions.items() if k in ['brx', 'bry']}
                print("设置B端角度初始位置:", b_angle)
                b_angle_controller.set_initial_position(b_angle)
    
    def back_to_initial_positions(self):
        """所有控制器回归到初始位置"""
        success = True
        
        # A端位置控制器回归初始位置
        a_pos_controller = self.device_manager.get_pzt_controller("A端位置控制器")
        if a_pos_controller and not a_pos_controller.back_to_initial_position():
            success = False
        
        # A端角度控制器回归初始位置
        a_angle_controller = self.device_manager.get_pzt_controller("A端角度控制器")
        if a_angle_controller and not a_angle_controller.back_to_initial_position():
            success = False
        
        # 如果是双端模式，B端控制器也回归初始位置
        if self.mode == "dual":
            b_pos_controller = self.device_manager.get_pzt_controller("B端位置控制器")
            if b_pos_controller and not b_pos_controller.back_to_initial_position():
                success = False
            
            b_angle_controller = self.device_manager.get_pzt_controller("B端角度控制器")
            if b_angle_controller and not b_angle_controller.back_to_initial_position():
                success = False
        
        return success
    
    def _convert_state_to_position(self, state: Dict[str, float]) -> Dict[str, float]:
        """将算法状态转换为硬件位置格式 - 改进版本"""
        converted = {}
        
        # 定义坐标映射关系 - 更清晰的映射
        coordinate_mapping = {
            # A端位置映射
            'A_x': 'x', 'A_y': 'y', 'A_z': 'z', 
            'A_rx': 'rx', 'A_ry': 'ry',
            # B端位置映射  
            'B_x': 'bx', 'B_y': 'by', 'B_z': 'bz', 
            'B_rx': 'brx', 'B_ry': 'bry',
            # 单端模式兼容
            'x': 'x', 'y': 'y', 'z': 'z', 'rx': 'rx', 'ry': 'ry'
        }
        
        # 转换所有坐标
        for key, value in state.items():
            if key in coordinate_mapping:
                new_key = coordinate_mapping[key]
                converted[new_key] = value
            else:
                print(f"⚠️ 未知坐标键: {key}，跳过")
        
        # 确保所有必需坐标都有默认值
        default_positions = {
            'x': 0, 'y': 0, 'z': 0, 'rx': 0, 'ry': 0,
            'bx': 0, 'by': 0, 'bz': 0, 'brx': 0, 'bry': 0
        }
        
        # 用实际值覆盖默认值
        for key, default_value in default_positions.items():
            if key not in converted:
                converted[key] = default_value
        
        print(f"🔧 坐标转换: {state} -> {converted}")
        return converted
    
    def _get_controller_axes(self, controller_name: str) -> List[str]:
        """获取控制器负责的轴"""
        if "位置" in controller_name:
            return ['x', 'y', 'z'] if "A端" in controller_name else ['bx', 'by', 'bz']
        else:
            return ['rx', 'ry'] if "A端" in controller_name else ['brx', 'bry']
    
    def zero_all(self) -> bool:
        """所有轴调零"""
        success = True
        controllers = [
            self.device_manager.get_pzt_controller("A端位置控制器"),
            self.device_manager.get_pzt_controller("A端角度控制器"),
        ]
        print(controllers)
        if self.mode == "dual":
            controllers.extend([
                self.device_manager.get_pzt_controller("B端位置控制器"),
                self.device_manager.get_pzt_controller("B端角度控制器"),
            ])
            print(controllers)
        for controller in controllers:
            if controller and not controller.zero():
                success = False
        
        return success
    
    def disconnect(self) -> bool:
        """断开连接"""
        self.device_manager.disconnect_all()
        return True
    
    def get_current_position(self) -> Dict[str, float]:
        """获取当前位置"""
        position = {}
        
        # 获取A端位置控制器状态
        a_pos_controller = self.device_manager.get_pzt_controller("A端位置控制器")
        if a_pos_controller:
            a_pos = a_pos_controller.get_current_position()
            position.update({k: v for k, v in a_pos.items() if k in ['x', 'y', 'z']})
        
        # 获取A端角度控制器状态
        a_angle_controller = self.device_manager.get_pzt_controller("A端角度控制器")
        if a_angle_controller:
            a_angle = a_angle_controller.get_current_position()
            position.update({k: v for k, v in a_angle.items() if k in ['rx', 'ry']})
        
        # 如果是双端模式，获取B端状态
        if self.mode == "dual":
            b_pos_controller = self.device_manager.get_pzt_controller("B端位置控制器")
            if b_pos_controller:
                b_pos = b_pos_controller.get_current_position()
                position.update({f"b_{k}": v for k, v in b_pos.items() if k in ['x', 'y', 'z']})
            
            b_angle_controller = self.device_manager.get_pzt_controller("B端角度控制器")
            if b_angle_controller:
                b_angle = b_angle_controller.get_current_position()
                position.update({f"b_{k}": v for k, v in b_angle.items() if k in ['rx', 'ry']})
        
        return position
    
    def enable_debug_mode(self, enable: bool = True):
        """启用或禁用调试模式"""
        self.debug_mode = enable
        print(f"调试模式: {'启用' if enable else '禁用'}")