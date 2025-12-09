import threading
from typing import Dict, Optional, Tuple, List
from hardware_drivers_pzt import PiezoController
from PowerMeter import PowerMeter
from logger11 import get_logger

logger = get_logger(__name__)

class GlobalDeviceManager:
    """全局设备管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(GlobalDeviceManager, cls).__new__(cls)
                cls._instance._init()
        return cls._instance
    
    def _init(self):
        """初始化"""
        self._power_meter = None
        self._pzt_controllers = {}
    
    def initialize_power_meter(self, wavelength=1550) -> Tuple[bool, str]:
        """初始化功率计"""
        try:
            self._power_meter = PowerMeter(wavelength=wavelength)
            logger.info("功率计初始化成功")
            return True, "功率计初始化成功"
        except Exception as e:
            logger.error(f"功率计初始化失败: {e}")
            return False, f"功率计初始化失败: {e}"
    
    def initialize_pzt_controller(self, name: str, serial_no: str, max_retries=3) -> Tuple[bool, str]:
        """初始化PZT控制器，带重试机制"""
        import time

        for attempt in range(max_retries):
            try:
                if name in self._pzt_controllers:
                    logger.info(f"{name} 已初始化")
                    return True, f"{name} 已初始化"

                logger.info(f"尝试连接 {name} (尝试 {attempt+1}/{max_retries})...")

                controller = PiezoController(name, serial_no)

                # 设置连接超时
                import threading
                result = [None]
                exception = [None]

                def connect_thread():
                    try:
                        result[0] = controller.connect()
                    except Exception as e:
                        exception[0] = e

                thread = threading.Thread(target=connect_thread)
                thread.daemon = True
                thread.start()
                thread.join(15)  # 15秒超时

                if thread.is_alive():
                    logger.error(f"{name} 连接超时")
                    try:
                        controller.disconnect()
                    except:
                        pass
                    continue

                if exception[0] is not None:
                    raise exception[0]

                if result[0]:
                    self._pzt_controllers[name] = controller
                    logger.info(f"{name} 初始化成功")
                    return True, f"{name} 初始化成功"
                else:
                    logger.warning(f"{name} 连接失败，将重试")
                    time.sleep(2)

            except Exception as e:
                logger.error(f"{name} 初始化失败: {e}")
                time.sleep(2)

        logger.error(f"{name} 所有连接尝试均失败")
        return False, f"{name} 连接失败，所有尝试均未成功"
    
    def initialize_all_pzt_controllers(self, mode: str = "single", config: Dict = None) -> Tuple[bool, str]:
        """初始化所有需要的PZT控制器"""
        if config is None:
            config = {}
        
        # 基础控制器配置
        base_controllers = {
            "A端位置控制器": config.get("a_position_serial", "A_POS_SERIAL"),
            "A端角度控制器": config.get("a_angle_serial", "A_ANGLE_SERIAL")
        }
        
        # 双端模式下添加B端控制器
        if mode == "dual":
            base_controllers.update({
                "B端位置控制器": config.get("b_position_serial", "B_POS_SERIAL"),
                "B端角度控制器": config.get("b_angle_serial", "B_ANGLE_SERIAL")
            })
        
        # 初始化所有控制器
        all_success = True
        messages = []
        
        for name, serial_no in base_controllers.items():
            success, msg = self.initialize_pzt_controller(name, serial_no)
            if not success:
                all_success = False
                messages.append(f"{name} 初始化失败: {msg}")
            else:
                messages.append(f"{name} 初始化成功")
        
        return all_success, "; ".join(messages)

    def disconnect_failed_controllers(self):
        """断开连接失败的控制器"""
        failed_controllers = []
        for name, controller in list(self._pzt_controllers.items()):
            if not controller.is_connected:
                try:
                    controller.disconnect()
                    failed_controllers.append(name)
                except:
                    pass
        
        for name in failed_controllers:
            self._pzt_controllers.pop(name, None)
        
        return failed_controllers
    
    def get_power_meter(self) -> Optional[PowerMeter]:
        """获取功率计实例"""
        return self._power_meter
    
    def get_pzt_controller(self, name: str) -> Optional[PiezoController]:
        """获取指定名称的PZT控制器"""
        return self._pzt_controllers.get(name)
    
    def get_all_pzt_controllers(self) -> Dict[str, PiezoController]:
        """获取所有PZT控制器"""
        return self._pzt_controllers.copy()
    
    def check_devices_ready(self, mode: str = "single") -> Tuple[bool, str]:
        """检查设备是否就绪"""
        required_controllers = ["A端位置控制器", "A端角度控制器"]
        
        if mode == "dual":
            required_controllers.extend(["B端位置控制器", "B端角度控制器"])
        
        # 检查PZT控制器
        for name in required_controllers:
            controller = self.get_pzt_controller(name)
            if not controller or not controller.is_connected:
                return False, f"{name} 未连接"
        
        # 检查功率计
        if not self.get_power_meter():
            return False, "功率计未连接"
        
        return True, "所有设备就绪"
    
    def disconnect_all(self):
        """断开所有设备连接"""
        try:
            if self._power_meter:
                self._power_meter.close()
                self._power_meter = None
                logger.info("功率计已断开")
            
            for name, controller in self._pzt_controllers.items():
                try:
                    controller.disconnect()
                    logger.info(f"{name} 已断开")
                except Exception as e:
                    logger.error(f"断开 {name} 时出错: {e}")
            
            self._pzt_controllers.clear()
        except Exception as e:
            logger.error(f"断开所有设备时出错: {e}")
    
    def check_connection_status(self):
        """检查所有设备的连接状态"""
        status = {}
        
        # 检查功率计
        if self._power_meter:
            status["功率计"] = "已连接"
        else:
            status["功率计"] = "未连接"
        
        # 检查PZT控制器
        for name, controller in self._pzt_controllers.items():
            if controller and hasattr(controller, 'is_connected') and controller.is_connected:
                status[name] = "已连接"
            else:
                status[name] = "未连接"
        
        return status