from abc import ABC, abstractmethod
from typing import Dict

class IPowerMeter(ABC):
    """功率计抽象接口"""
    
    @abstractmethod
    def measure_power(self, samples: int = 5, interval: float = 0.02) -> Dict:
        """测量功率"""
        pass
    
    @abstractmethod
    def set_wavelength(self, wavelength: float) -> bool:
        """设置波长"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭连接"""
        pass

class IPZTController(ABC):
    """PZT控制器抽象接口"""
    
    @abstractmethod
    def connect(self) -> bool:
        """连接设备"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass
    
    @abstractmethod
    def zero(self) -> bool:
        """调零"""
        pass
    
    @abstractmethod
    def set_position(self, position_dict: Dict[str, float]) -> bool:
        """设置位置"""
        pass
    
    @abstractmethod
    def back_to_initial_position(self) -> bool:
        """回归初始位置"""
        pass