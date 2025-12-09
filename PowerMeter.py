from datetime import datetime
from ctypes import c_long, c_uint32, byref, create_string_buffer, c_bool, c_char_p, c_int, c_double
import time
from TLPM import TLPM  # 直接导入实际库，不处理模拟情况
from ctypes import c_int16
import numpy as np

class PowerMeter:
    DEFAULT_WAVELENGTH = 1560  # 默认波长1550nm（内部以纳米为单位）
    
    def __init__(self, wavelength=DEFAULT_WAVELENGTH):
        """初始化功率计并自动连接设备
        :param wavelength: 测量波长，单位米（例如1550nm = 1550e-9m）
        """
        self.tlPM = None  # 功率计库实例
        self.device_count = c_uint32()  # 设备数量
        self.resource_name = None  # 设备资源名称
        self.wavelength = wavelength  # 当前波长（纳米）
        self.current_range = None  # 当前量程
        self._find_device()  # 搜索设备
        self._find_and_connect_device()  # 初始化时自动连接设备
    
    def _find_device(self):
        """搜索可用功率计设备"""
        self.tlPM = TLPM()
        self.tlPM.findRsrc(byref(self.device_count))
        print(f"发现 {self.device_count.value} 个功率计设备")
    
    def _get_device_name(self, index=0):
        """获取指定索引的设备资源名称"""
        if index >= self.device_count.value:
            raise IndexError(f"设备索引 {index} 超出范围（0-{self.device_count.value - 1}）")
        
        resource_buffer = create_string_buffer(1024)
        self.tlPM.getRsrcName(c_int(index), resource_buffer)
        device_name = c_char_p(resource_buffer.raw).value.decode('utf-8')
        print(f"设备 {index} 资源名称: {device_name}")
        return device_name
    
    def _find_and_connect_device(self):
        """搜索并连接功率计设备（优先连接第一个，失败则尝试第二个）"""
        try:
            # 搜索设备
            self._find_device()
            
            if self.device_count.value == 0:
                raise ConnectionError("未发现任何功率计设备")
            
            # 尝试连接第一个设备
            self.resource_name = self._get_device_name(0)
            resource_buffer = create_string_buffer(self.resource_name.encode('utf-8'))
            self.tlPM.open(resource_buffer, c_bool(True), c_bool(True))
            
            # 初始化设备参数
            self.tlPM.setPowerAutoRange(c_int16(1))  # 启用自动量程 (TLPM_AUTORANGE_POWER_ON = 1)
            self.set_wavelength(self.wavelength)  # 设置初始波长
            
            # 获取初始量程
            self._update_current_range()
            
            # 打印校准信息
            calib_buffer = create_string_buffer(1024)
            self.tlPM.getCalibrationMsg(calib_buffer)
            calib_info = c_char_p(calib_buffer.raw).value.decode('utf-8')
            print(f"设备校准信息: {calib_info}")
            
            print("功率计连接成功")
            
        except Exception as e:
            print(f"连接第一个设备失败: {str(e)}")
            
            # 若存在多个设备，尝试连接第二个
            if self.device_count.value > 1:
                try:
                    print("尝试连接第二个设备...")
                    self.resource_name = self._get_device_name(1)
                    resource_buffer = create_string_buffer(self.resource_name.encode('utf-8'))
                    self.tlPM.open(resource_buffer, c_bool(True), c_bool(True))
                    
                    # 初始化参数
                    self.tlPM.setPowerAutoRange(c_int16(1))  # 启用自动量程
                    self.set_wavelength(self.wavelength)
                    self._update_current_range()
                    
                    print("第二个设备连接成功")
                except Exception as e2:
                    raise ConnectionError(f"所有设备连接失败: {str(e2)}") from e
            else:
                raise ConnectionError(f"设备连接失败: {str(e)}") from e
    
    def _update_current_range(self):
        """更新当前功率量程信息"""
        try:
            power_range = c_double()
            # 获取当前设置的功率量程 (TLPM_ATTR_SET_VAL = 0)
            self.tlPM.getPowerRange(c_int16(0), byref(power_range))
            self.current_range = power_range.value
            return self.current_range
        except Exception as e:
            print(f"获取功率量程失败: {str(e)}")
            self.current_range = None
            return None
    
    def get_current_range(self):
        """获取当前功率量程"""
        return self._update_current_range()
    
    def _to_scientific_notation(self, value):
        """将数值转换为科学计数法表示 (mantissa, exponent)"""
        if value == 0:
            return 0.0, 0
        exponent = np.floor(np.log10(abs(value)))
        mantissa = value / (10 ** exponent)
        return mantissa, int(exponent)
    
    def _get_scientific_display_info(self, power_value, current_range):
        """
        根据功率值确定科学计数法显示信息
        :return: 包含显示信息的字典
        """
        if power_value == 0:
            return {
                'value': 0.0,
                'exponent': 0,
                'unit': 'W',
                'scientific': '0.000 W',
                'engineering': '0.000 W'
            }
        
        # 计算10的指数
        exponent = int(np.floor(np.log10(abs(power_value))))
        
        # 确定最佳单位和对应的指数调整
        unit_info = [
            (-12, 'pW'),  # 皮瓦
            (-9, 'nW'),   # 纳瓦
            (-6, 'μW'),   # 微瓦
            (-3, 'mW'),   # 毫瓦
            (0, 'W')      # 瓦
        ]
        
        # 选择最合适的单位
        selected_unit = 'W'
        adjusted_exponent = 0
        for exp_offset, unit_name in unit_info:
            if exponent <= exp_offset + 3:  # 在单位范围内显示1-999的值
                selected_unit = unit_name
                adjusted_exponent = exp_offset
                break
        
        # 计算显示值
        display_value = power_value / (10 ** adjusted_exponent)
        
        # 确定小数位数
        if abs(display_value) < 1:
            precision = 4
        elif abs(display_value) < 10:
            precision = 3
        elif abs(display_value) < 100:
            precision = 2
        else:
            precision = 1
        
        # 生成科学计数法字符串
        scientific_str = f"{power_value:.3e} W"
        engineering_str = f"{display_value:.{precision}f} {selected_unit}"
        
        return {
            'value': display_value,
            'exponent': adjusted_exponent,
            'unit': selected_unit,
            'scientific': scientific_str,
            'engineering': engineering_str
        }
    
    def set_wavelength(self, wavelength):
        """
        设置测量波长
        :param wavelength: 波长值（单位：纳米）
        """
        try:
            # 设置波长（内部以纳米为单位）
            self.tlPM.setWavelength(c_double(wavelength))
            
            # 验证设置
            current_wl = c_double()
            self.tlPM.getWavelength(c_int16(0), byref(current_wl))
            self.wavelength = current_wl.value  # 更新内部存储值
            
            print(f"波长已设置为: {current_wl.value} nm")
            return True
            
        except Exception as e:
            print(f"设置波长失败: {str(e)}")
            raise
    
    def measure_power(self, samples=5, interval=0.001):
        """
        测量功率并返回处理后结果，包含量程信息和优化精度
        使用简单方法：去除偏离最大的两个异常值后求平均
        :param samples: 采样次数（默认5次）
        :param interval: 采样间隔（秒，默认0.001秒）
        :return: 包含功率值、量程及相关信息的字典
        """
        if samples < 2:
            raise ValueError("采样次数不能少于2次")
        
        try:
            # 获取当前波长（用于结果返回）
            current_wl = c_double()
            self.tlPM.getWavelength(c_int16(0), byref(current_wl)) 
            wavelength_m = current_wl.value
            wavelength_nm = wavelength_m * 1e9
            
            # 获取初始量程
            initial_range = self._update_current_range()
            
            # 多次采样
            measurements = np.zeros(samples, dtype=np.float64)
            
            for i in range(samples):
                power = c_double()
                self.tlPM.measPower(byref(power))
                power_val = power.value
                measurements[i] = power_val
                
                # 动态更新量程
                if i == samples - 1:
                    final_range = self._update_current_range()
                else:
                    final_range = initial_range
                
                # 使用科学计数法显示小数值
                if abs(power_val) < 1e-6:  # 小于1微瓦时使用科学计数法
                    print(f"第{i+1}次采样: {power_val:.3e} W")
                else:
                    print(f"第{i+1}次采样: {power_val:.9f} W")
                
                if i < samples - 1:
                    time.sleep(interval)
            
            # 简单数据处理：去除偏离最大的两个异常值后求平均
            if samples == 5:
                # 计算每个数据点与中位数的绝对偏差
                median_val = np.median(measurements)
                deviations = np.abs(measurements - median_val)
                
                # 找出偏离最大的两个索引
                max_dev_indices = np.argsort(deviations)[-2:]
                
                # 去除这两个异常值
                valid_measurements = np.delete(measurements, max_dev_indices)
                
                # 对剩下的三个值求平均
                final_avg = np.mean(valid_measurements)
                
                print(f"去除异常值索引: {max_dev_indices}")
                print(f"有效数据: {valid_measurements}")
            else:
                # 对于不是5个样本的情况，使用简单平均
                final_avg = np.mean(measurements)
                valid_measurements = measurements
            
            # 计算统计信息
            stats = {
                'mean': float(final_avg),
                'median': float(np.median(measurements)),
                'std': float(np.std(measurements, dtype=np.float64)) if len(measurements) > 1 else 0.0,
                'min': float(np.min(measurements)),
                'max': float(np.max(measurements)),
                'range': float(np.ptp(measurements)) if len(measurements) > 1 else 0.0,
                'valid_samples': len(valid_measurements),
                'removed_samples': len(measurements) - len(valid_measurements)
            }
            
            # 确定最佳显示格式
            display_info = self._get_scientific_display_info(final_avg, final_range)
            
            # 返回结果字典
            result = {
                "power": final_avg,  # 最终功率值（单位：W）
                "display_value": display_info['value'],  # 显示值
                "display_exponent": display_info['exponent'],  # 10的指数
                "display_unit": display_info['unit'],  # 显示单位
                "scientific_notation": display_info['scientific'],  # 科学计数法字符串
                "engineering_notation": display_info['engineering'],  # 工程单位字符串
                "power_range": final_range,  # 最终功率量程（W）
                "wavelength_m": wavelength_m,  # 波长（米）
                "wavelength_nm": wavelength_nm,  # 波长（纳米）
                "raw_data": measurements.tolist(),  # 原始采样数据
                "valid_data": valid_measurements.tolist(),  # 有效数据（去除异常值后）
                "statistics": stats,  # 统计信息
                "timestamp": datetime.now().isoformat(),  # 时间戳
                "auto_range_enabled": True,  # 自动量程状态
            }
            
            # 打印最终结果（使用科学计数法）
            print(f"最终功率: {result['engineering_notation']} ({result['scientific_notation']})")
            print(f"量程: {final_range:.3e} W")
            print(f"使用有效数据点数: {len(valid_measurements)}/{samples}")
            
            return result
            
        except Exception as e:
            print(f"功率测量失败: {str(e)}")
            # 尝试重新连接后再次测量
            print("尝试重新连接设备...")
            self.close()
            self._find_and_connect_device()
            return self.measure_power(samples, interval)
                
    
    def measure_power_fast(self):
        """
        快速单次功率测量，包含量程信息
        :return: 包含功率值和量程的字典
        """
        try:
            power = c_double()
            self.tlPM.measPower(byref(power))
            power_val = power.value
            
            # 获取当前量程
            current_range = self._update_current_range()
            
            # 根据量程确定最佳显示格式
            display_info = self._get_scientific_display_info(power_val, current_range)
            
            return {
                "power": power_val,
                "display_value": display_info['value'],
                "display_unit": display_info['unit'],
                "scientific_notation": display_info['scientific'],
                "engineering_notation": display_info['engineering'],
                "power_range": current_range,
                "auto_range_enabled": True
            }
            
        except Exception as e:
            print(f"快速功率测量失败: {str(e)}")
            raise
    
    def powertest(self):
        """
        快速测量当前功率（用于GUI实时监测）
        单次测量，不进行多次采样和过滤
        :return: 当前功率值（单位：W）
        """
        try:
            power = c_double()
            self.tlPM.measPower(byref(power))
            return power.value
        except Exception as e:
            print(f"快速功率测量失败: {str(e)}")
    
    def set_power_auto_range(self, enabled=True):
        """
        设置功率自动量程
        :param enabled: True启用自动量程，False禁用
        """
        try:
            mode = c_int16(1) if enabled else c_int16(0)  # TLPM_AUTORANGE_POWER_ON = 1, OFF = 0
            self.tlPM.setPowerAutoRange(mode)
            print(f"功率自动量程已{'启用' if enabled else '禁用'}")
            return True
        except Exception as e:
            print(f"设置自动量程失败: {str(e)}")
            return False
    
    def close(self):
        """关闭设备连接"""
        if hasattr(self, 'tlPM') and self.tlPM is not None:
            try:
                self.tlPM.close()
                print("功率计连接已关闭")
            except Exception as e:
                print(f"关闭连接时出错: {str(e)}")
        self.tlPM = None
    
    def __del__(self):
        """对象销毁时自动关闭连接"""
        self.close()


# 单例模式：保持设备长期连接
_power_meter_instance = None

def get_power_meter(wavelength=PowerMeter.DEFAULT_WAVELENGTH):
    """获取功率计单例实例（确保全局唯一连接）"""
    global _power_meter_instance
    if _power_meter_instance is None:
        print(f"初始化功率计（默认波长: {wavelength} nm）")
        _power_meter_instance = PowerMeter(wavelength=wavelength)
    return _power_meter_instance


def measure_power(samples=5, interval=0.02, wavelength=None):
    """便捷测量函数：获取功率值（支持临时设置波长）"""
    pm = get_power_meter()
    
    # 若指定了新波长则更新
    if wavelength is not None:
        pm.set_wavelength(wavelength)
    
    return pm.measure_power(samples, interval)

def powertest():
    """快速功率测试"""
    pm = get_power_meter()
    return pm.powertest()

def get_current_range():
    """获取当前功率量程"""
    pm = get_power_meter()
    return pm.get_current_range()

def measure_power_fast():
    """快速功率测量"""
    pm = get_power_meter()
    return pm.measure_power_fast()