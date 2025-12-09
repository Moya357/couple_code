# high_power_keep.py
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List, Optional

class HighPowerKeepMode:
    """
    改进的高功率保持模式
    在全局收敛后，以最佳个体为中心进行小范围搜索，以补偿温度漂移
    """
    
    def __init__(self, config: dict, selected_variables_A: List[str], 
             selected_variables_B: List[str], 
             search_range_A: Dict, search_range_B: Dict):
        """
        初始化高功率保持模式
        
        参数:
            config: 配置字典
            selected_variables_A: A端选择的变量
            selected_variables_B: B端选择的变量
            search_range_A: A端原始搜索范围
            search_range_B: B端原始搜索范围
        """
        self.config = config
        self.selected_variables_A = selected_variables_A
        self.selected_variables_B = selected_variables_B
        self.search_range_A = search_range_A
        self.search_range_B = search_range_B
        
        # 高功率保持模式参数
        self.high_power_population_size = config.get('high_power_population_size', 20)
        self.high_power_mutation_rate = config.get('high_power_mutation_rate', 0.05)
        self.high_power_crossover_rate = config.get('high_power_crossover_rate', 0.3)
        
        # 新增：小范围搜索参数
        self.high_power_search_range_percent = config.get('high_power_search_range_percent', 0.05)  # 5%的搜索范围
        self.high_power_perturbation_strength = config.get('high_power_perturbation_strength', 0.01)  # 克隆扰动强度
        
        # 状态跟踪
        self.center_individual_A = None  # 当前搜索中心（A端）
        self.center_individual_B = None  # 当前搜索中心（B端）
        self.best_fitness = -np.inf
        self.generation = 0
        self.best_individuals_history = []  # 记录最佳个体历史，用于回退
        # 新增：高功率收敛阈值
        self.high_power_convergence_threshold = config.get('high_power_convergence_threshold', 0.01)  # 1%的阈值
        
        # 新增：参数调整相关
        self.last_best_fitness = -np.inf
        self.convergence_counter = 0
        self.max_convergence_count = 3  # 连续收敛次数阈值
        
        # 新增：参数调整幅度
        self.param_adjustment_rate = config.get('param_adjustment_rate', 0.5)  # 参数调整幅度
        self.min_mutation_rate = config.get('min_mutation_rate', 0.01)  # 最小变异率
        self.max_mutation_rate = config.get('max_mutation_rate', 0.2)   # 最大变异率
        self.min_crossover_rate = config.get('min_crossover_rate', 0.1) # 最小交叉率
        self.max_crossover_rate = config.get('max_crossover_rate', 0.8) # 最大交叉率
        
        print(f"高功率收敛阈值: {self.high_power_convergence_threshold*100}%")
        # 事件记录
        self.events = []
        
        print(f"高功率保持模式初始化完成")
        print(f"搜索范围: ±{self.high_power_search_range_percent*100}%")
        print(f"克隆扰动强度: {self.high_power_perturbation_strength}")
    def update_search_center(self, best_individual_A: np.ndarray, best_individual_B: np.ndarray, 
                       best_fitness: float):
        """
        更新搜索中心，并动态调整参数
        """
        # 计算适应度变化
        fitness_change = 0
        if self.last_best_fitness > 0:
            fitness_change = abs(best_fitness - self.last_best_fitness) / self.last_best_fitness
        
        # 记录当前适应度
        current_fitness = self.last_best_fitness
        self.last_best_fitness = best_fitness
        
        # 更新最佳适应度
        if best_fitness > self.best_fitness:
            old_fitness = self.best_fitness
            self.best_fitness = best_fitness
            self.center_individual_A = best_individual_A.copy()
            self.center_individual_B = best_individual_B.copy()
            
            print(f"高功率保持模式：更新搜索中心，功率从{old_fitness:.6f}mW提升到{best_fitness:.6f}mW")
            
            # 记录事件
            improvement_event = {
                'event_type': 'search_center_improved',
                'timestamp': datetime.now().isoformat(),
                'old_fitness': old_fitness,
                'new_fitness': best_fitness,
                'improvement_percent': (best_fitness - old_fitness) / old_fitness if old_fitness > 0 else 0
            }
            self.events.append(improvement_event)
        
        # 动态调整参数
        self._adjust_parameters_dynamically(fitness_change, best_fitness)
        
        # 保存到历史记录
        self.best_individuals_history.append((best_individual_A.copy(), 
                                            best_individual_B.copy(), 
                                            best_fitness))
        if len(self.best_individuals_history) > 10:
            self.best_individuals_history.pop(0)

    def _adjust_parameters_dynamically(self, fitness_change: float, current_fitness: float):
        """
        动态调整参数
        
        参数:
            fitness_change: 适应度变化率
            current_fitness: 当前适应度
        """
        if fitness_change == 0:
            return
        
        # 检查是否收敛（变化小于阈值）
        if fitness_change < self.high_power_convergence_threshold:
            self.convergence_counter += 1
            print(f"高功率模式：检测到收敛，连续收敛次数: {self.convergence_counter}")
            
            # 如果连续收敛次数超过阈值，减小参数以保护优秀基因
            if self.convergence_counter >= self.max_convergence_count:
                # 减小变异率和交叉率
                self.high_power_mutation_rate = max(
                    self.min_mutation_rate,
                    self.high_power_mutation_rate * (1 - self.param_adjustment_rate)
                )
                self.high_power_crossover_rate = max(
                    self.min_crossover_rate,
                    self.high_power_crossover_rate * (1 - self.param_adjustment_rate)
                )
                
                print(f"高功率模式：减小参数以保护优秀基因")
                print(f"  变异率: {self.high_power_mutation_rate:.3f}")
                print(f"  交叉率: {self.high_power_crossover_rate:.3f}")
                
                # 记录参数调整事件
                param_event = {
                    'event_type': 'parameters_decreased',
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'convergence_detected',
                    'convergence_counter': self.convergence_counter,
                    'new_mutation_rate': self.high_power_mutation_rate,
                    'new_crossover_rate': self.high_power_crossover_rate
                }
                self.events.append(param_event)
        else:
            # 适应度变化较大，重置收敛计数器并增大参数以增加探索
            self.convergence_counter = 0
            
            # 增大变异率和交叉率
            self.high_power_mutation_rate = min(
                self.max_mutation_rate,
                self.high_power_mutation_rate * (1 + self.param_adjustment_rate)
            )
            self.high_power_crossover_rate = min(
                self.max_crossover_rate,
                self.high_power_crossover_rate * (1 + self.param_adjustment_rate)
            )
            
            print(f"高功率模式：增大参数以增加探索")
            print(f"  变异率: {self.high_power_mutation_rate:.3f}")
            print(f"  交叉率: {self.high_power_crossover_rate:.3f}")
            
            # 记录参数调整事件
            param_event = {
                'event_type': 'parameters_increased',
                'timestamp': datetime.now().isoformat(),
                'reason': 'significant_fitness_change',
                'fitness_change': fitness_change,
                'new_mutation_rate': self.high_power_mutation_rate,
                'new_crossover_rate': self.high_power_crossover_rate
            }
            self.events.append(param_event)

    # 添加获取当前参数的方法
    def get_current_parameters(self) -> dict:
        """
        获取当前参数
        
        返回:
            参数字典
        """
        return {
            'high_power_mutation_rate': self.high_power_mutation_rate,
            'high_power_crossover_rate': self.high_power_crossover_rate,
            'high_power_search_range_percent': self.high_power_search_range_percent,
            'high_power_perturbation_strength': self.high_power_perturbation_strength,
            'convergence_counter': self.convergence_counter,
            'high_power_convergence_threshold': self.high_power_convergence_threshold
        }

    def update_parameters_from_gui(self, new_params: dict):
        """
        从GUI更新参数
        
        参数:
            new_params: 新参数字典
        """
        update_count = 0
        
        if 'high_power_population_size' in new_params:
            new_size = int(new_params['high_power_population_size'])
            if new_size >= 5:
                self.high_power_population_size = new_size
                update_count += 1
                print(f"高功率种群大小更新为: {new_size}")
        
        if 'high_power_mutation_rate' in new_params:
            new_rate = float(new_params['high_power_mutation_rate'])
            if 0 <= new_rate <= 1:
                self.high_power_mutation_rate = new_rate
                update_count += 1
                print(f"高功率变异率更新为: {new_rate}")
        
        if 'high_power_crossover_rate' in new_params:
            new_rate = float(new_params['high_power_crossover_rate'])
            if 0 <= new_rate <= 1:
                self.high_power_crossover_rate = new_rate
                update_count += 1
                print(f"高功率交叉率更新为: {new_rate}")
        
        # 新增：更新小范围搜索参数
        if 'high_power_search_range_percent' in new_params:
            new_range = float(new_params['high_power_search_range_percent'])
            if 0.001 <= new_range <= 0.2:  # 限制在0.1%到20%之间
                self.high_power_search_range_percent = new_range
                update_count += 1
                print(f"高功率搜索范围更新为: ±{new_range*100}%")
        
        if 'high_power_perturbation_strength' in new_params:
            new_strength = float(new_params['high_power_perturbation_strength'])
            if 0 <= new_strength <= 0.1:  # 限制在0-10%之间
                self.high_power_perturbation_strength = new_strength
                update_count += 1
                print(f"高功率克隆扰动强度更新为: {new_strength}")
        # 新增：高功率收敛阈值
        if 'high_power_convergence_threshold' in new_params:
            new_threshold = float(new_params['high_power_convergence_threshold'])
            if 0.001 <= new_threshold <= 0.1:  # 0.1%到10%
                self.high_power_convergence_threshold = new_threshold
                update_count += 1
                print(f"高功率收敛阈值更新为: {new_threshold*100}%")
        
        # 新增：参数调整相关
        if 'param_adjustment_rate' in new_params:
            new_rate = float(new_params['param_adjustment_rate'])
            if 0.1 <= new_rate <= 1.0:
                self.param_adjustment_rate = new_rate
                update_count += 1
                print(f"参数调整幅度更新为: {new_rate}")
        
        if 'min_mutation_rate' in new_params:
            new_rate = float(new_params['min_mutation_rate'])
            if 0 <= new_rate <= 0.1:
                self.min_mutation_rate = new_rate
                update_count += 1
                print(f"最小变异率更新为: {new_rate}")
        
        if 'max_mutation_rate' in new_params:
            new_rate = float(new_params['max_mutation_rate'])
            if 0.1 <= new_rate <= 1.0:
                self.max_mutation_rate = new_rate
                update_count += 1
                print(f"最大变异率更新为: {new_rate}")
        
        if 'min_crossover_rate' in new_params:
            new_rate = float(new_params['min_crossover_rate'])
            if 0 <= new_rate <= 0.5:
                self.min_crossover_rate = new_rate
                update_count += 1
                print(f"最小交叉率更新为: {new_rate}")
        
        if 'max_crossover_rate' in new_params:
            new_rate = float(new_params['max_crossover_rate'])
            if 0.5 <= new_rate <= 1.0:
                self.max_crossover_rate = new_rate
                update_count += 1
                print(f"最大交叉率更新为: {new_rate}")
        
        if update_count > 0:
            update_event = {
                'event_type': 'high_power_parameters_updated',
                'timestamp': datetime.now().isoformat(),
                'updated_parameters': new_params,
                'update_count': update_count
            }
            self.events.append(update_event)
        
        return update_count
    
    def initialize(self, best_individual_A: np.ndarray, best_individual_B: np.ndarray, 
                  best_fitness: float, best_individuals_history: list = None):
        """
        初始化高功率保持模式
        
        参数:
            best_individual_A: A端最佳个体
            best_individual_B: B端最佳个体
            best_fitness: 最佳适应度
            best_individuals_history: 最佳个体历史记录（最后几代）
        """
        # 如果提供了历史记录，从最后三代中选取功率最高的个体
        if best_individuals_history and len(best_individuals_history) >= 3:
            print("从最后三代中选取功率最高的个体作为搜索中心")
            
            # 找出最后三代中功率最高的个体
            best_fitness_in_history = -np.inf
            best_idx = -1
            
            for i, (ind_A, ind_B, fitness) in enumerate(best_individuals_history[-3:]):
                if fitness > best_fitness_in_history:
                    best_fitness_in_history = fitness
                    best_idx = i
                    self.center_individual_A = ind_A.copy()
                    self.center_individual_B = ind_B.copy()
            
            print(f"选取第{len(best_individuals_history)-3+best_idx+1}代个体作为搜索中心")
        else:
            # 使用当前最佳个体作为搜索中心
            self.center_individual_A = best_individual_A.copy()
            self.center_individual_B = best_individual_B.copy()
        
        self.best_fitness = best_fitness
        
        # 保存到历史记录（用于回退）
        self.best_individuals_history = [(self.center_individual_A.copy(), 
                                         self.center_individual_B.copy(), 
                                         self.best_fitness)]
        
        print(f"高功率保持模式初始化完成")
        print(f"搜索中心设置完成，中心功率: {self.best_fitness:.6f}mW")
        print(f"搜索范围: ±{self.high_power_search_range_percent*100}%")
        
        return True
    
    def create_initial_population(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.center_individual_A is None or self.center_individual_B is None:
            raise ValueError("高功率保持模式未初始化，请先调用initialize方法")
        
        population_A = np.zeros((self.high_power_population_size, len(self.selected_variables_A)))
        population_B = np.zeros((self.high_power_population_size, len(self.selected_variables_B)))
        
        # 获取小范围搜索区间
        search_range_A, search_range_B = self.get_search_range_around_center()
        
        # 克隆最佳个体并添加扰动（使用小范围搜索区间）
        for i in range(self.high_power_population_size):
            population_A[i] = self._create_near_center_individual_with_range(
                self.center_individual_A, 
                self.selected_variables_A, 
                search_range_A  # ✅ 使用小范围搜索区间
            )
            population_B[i] = self._create_near_center_individual_with_range(
                self.center_individual_B, 
                self.selected_variables_B, 
                search_range_B  # ✅ 使用小范围搜索区间
            )
        
        print(f"高功率保持模式：创建初始种群完成")
        print(f"  种群大小: {self.high_power_population_size}")
        print(f"  搜索范围: ±{self.high_power_search_range_percent*100}%")
        print(f"  克隆扰动强度: {self.high_power_perturbation_strength}")
        
        return population_A, population_B
    
    def _clone_with_perturbation(self, individual: np.ndarray, selected_variables: List[str], 
                                search_range: Dict) -> np.ndarray:
        """
        克隆个体并添加随机扰动
        
        参数:
            individual: 要克隆的个体
            selected_variables: 选择的变量
            search_range: 搜索范围
            
        返回:
            perturbed_individual: 添加扰动后的个体
        """
        perturbed = individual.copy()
        
        for i, var in enumerate(selected_variables):
            lower, upper = search_range[var]
            range_size = upper - lower
            
            # 计算扰动幅度
            perturbation = np.random.normal(0, range_size * self.high_power_perturbation_strength)
            perturbed[i] += perturbation
            
            # 确保在搜索范围内
            perturbed[i] = np.clip(perturbed[i], lower, upper)
        
        return perturbed
    
    def update_search_center(self, best_individual_A: np.ndarray, best_individual_B: np.ndarray, 
                           best_fitness: float):
        """
        更新搜索中心（每代结束后调用）
        
        参数:
            best_individual_A: 当前代A端最佳个体
            best_individual_B: 当前代B端最佳个体
            best_fitness: 当前代最佳适应度
        """
        # 更新最佳适应度
        if best_fitness > self.best_fitness:
            old_fitness = self.best_fitness
            self.best_fitness = best_fitness
            self.center_individual_A = best_individual_A.copy()
            self.center_individual_B = best_individual_B.copy()
            
            print(f"高功率保持模式：更新搜索中心，功率从{old_fitness:.6f}mW提升到{best_fitness:.6f}mW")
            
            # 记录事件
            improvement_event = {
                'event_type': 'search_center_improved',
                'timestamp': datetime.now().isoformat(),
                'old_fitness': old_fitness,
                'new_fitness': best_fitness,
                'improvement_percent': (best_fitness - old_fitness) / old_fitness if old_fitness > 0 else 0
            }
            self.events.append(improvement_event)
        elif best_fitness < self.best_fitness * 0.95:  # 如果功率下降超过5%
            print(f"高功率保持模式：功率下降超过5%，考虑是否发生漂移")
            
            # 记录漂移事件
            drift_event = {
                'event_type': 'possible_drift_detected',
                'timestamp': datetime.now().isoformat(),
                'current_fitness': best_fitness,
                'best_fitness': self.best_fitness,
                'drop_percent': (self.best_fitness - best_fitness) / self.best_fitness
            }
            self.events.append(drift_event)
        
        # 保存到历史记录（最多保存10个）
        self.best_individuals_history.append((best_individual_A.copy(), 
                                            best_individual_B.copy(), 
                                            best_fitness))
        if len(self.best_individuals_history) > 10:
            self.best_individuals_history.pop(0)
    
    def get_search_range_around_center(self) -> Tuple[Dict, Dict]:
        """
        获取以当前搜索中心为中心的小范围搜索区间
        
        返回:
            search_range_A: A端搜索范围
            search_range_B: B端搜索范围
        """
        search_range_A = {}
        search_range_B = {}
        
        # 为A端创建小范围搜索区间
        if self.center_individual_A is not None:
            for i, var in enumerate(self.selected_variables_A):
                center_value = self.center_individual_A[i]
                lower, upper = self.search_range_A[var]
                range_size = upper - lower
                search_range = range_size * self.high_power_search_range_percent
                
                # 计算局部搜索范围
                new_lower = max(center_value - search_range/2, lower)
                new_upper = min(center_value + search_range/2, upper)
                search_range_A[var] = (new_lower, new_upper)
        
        # 为B端创建小范围搜索区间
        if self.center_individual_B is not None:
            for i, var in enumerate(self.selected_variables_B):
                center_value = self.center_individual_B[i]
                lower, upper = self.search_range_B[var]
                range_size = upper - lower
                search_range = range_size * self.high_power_search_range_percent
                
                # 计算局部搜索范围
                new_lower = max(center_value - search_range/2, lower)
                new_upper = min(center_value + search_range/2, upper)
                search_range_B[var] = (new_lower, new_upper)
        
        return search_range_A, search_range_B
    
    def create_new_population(self, population_A: np.ndarray, population_B: np.ndarray, 
                        fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建新一代种群（高功率保持模式专用）
        
        参数:
            population_A: 当前代A端种群
            population_B: 当前代B端种群
            fitness: 当前代适应度
            
        返回:
            new_population_A: 新一代A端种群
            new_population_B: 新一代B端种群
        """
        new_population_A = np.zeros_like(population_A)
        new_population_B = np.zeros_like(population_B)
        
        # 1. 精英保留（保留最佳个体）
        elite_count = 1  # 高功率模式下只保留一个最佳个体
        if elite_count > 0:
            best_idx = np.argmax(fitness)
            new_population_A[0] = population_A[best_idx]
            new_population_B[0] = population_B[best_idx]
        
        # 2. 获取小范围搜索区间
        search_range_A, search_range_B = self.get_search_range_around_center()
        
        # 3. 基于当前搜索中心创建新个体（使用小范围搜索区间）
        for i in range(elite_count, len(new_population_A)):
            # 以当前搜索中心为基础，在小范围内添加随机扰动
            new_population_A[i] = self._create_near_center_individual_with_range(
                self.center_individual_A, 
                self.selected_variables_A, 
                search_range_A
            )
            new_population_B[i] = self._create_near_center_individual_with_range(
                self.center_individual_B, 
                self.selected_variables_B, 
                search_range_B
            )
        
        return new_population_A, new_population_B

    def _create_near_center_individual_with_range(self, center_individual: np.ndarray, 
                                            selected_variables: List[str], 
                                            local_search_range: Dict) -> np.ndarray:
        """
        在小范围搜索区间内创建接近搜索中心的个体
        """
        individual = center_individual.copy()
        
        for i, var in enumerate(selected_variables):
            if var in local_search_range:
                lower, upper = local_search_range[var]
            else:
                # 计算动态的小范围搜索区间
                # 确定原始搜索范围
                if var in self.search_range_A:
                    original_lower, original_upper = self.search_range_A[var]
                elif var in self.search_range_B:
                    original_lower, original_upper = self.search_range_B[var]
                else:
                    raise KeyError(f"变量 {var} 不在任何搜索范围中")
                
                center_value = center_individual[i]
                range_size = original_upper - original_lower
                search_range = range_size * self.high_power_search_range_percent
                
                # 计算局部搜索范围
                lower = max(center_value - search_range/2, original_lower)
                upper = min(center_value + search_range/2, original_upper)
            
            # 在小范围内随机扰动
            perturbation_range = upper - lower
            perturbation = np.random.uniform(-perturbation_range/2, perturbation_range/2)
            individual[i] += perturbation
            
            # 确保在小范围搜索区间内
            individual[i] = np.clip(individual[i], lower, upper)
        
        return individual
    
    def _create_near_center_individual(self, center_individual: np.ndarray, 
                                 selected_variables: List[str], 
                                 search_range: Dict) -> np.ndarray:
        """
        创建接近搜索中心的个体
        
        参数:
            center_individual: 中心个体
            selected_variables: 选择的变量
            search_range: 搜索范围
            
        返回:
            individual: 新个体
        """
        individual = center_individual.copy()
        
        for i, var in enumerate(selected_variables):
            if var in search_range:
                lower, upper = search_range[var]
            else:
                # 如果变量不在search_range中，使用默认范围
                # 这应该是从self.search_range_A或self.search_range_B中获取
                # 需要根据具体情况确定
                continue
                
            # 在小范围内随机扰动
            perturbation_range = (upper - lower) * self.high_power_search_range_percent
            perturbation = np.random.uniform(-perturbation_range/2, perturbation_range/2)
            individual[i] += perturbation
            
            # 确保在搜索范围内
            individual[i] = np.clip(individual[i], lower, upper)
        
        return individual
    
    def get_status(self) -> dict:
        """
        获取高功率保持模式状态
        
        返回:
            status: 状态字典
        """
        return {
            'center_individual_A': self.center_individual_A.tolist() if self.center_individual_A is not None else None,
            'center_individual_B': self.center_individual_B.tolist() if self.center_individual_B is not None else None,
            'best_fitness': self.best_fitness,
            'high_power_population_size': self.high_power_population_size,
            'high_power_search_range_percent': self.high_power_search_range_percent,
            'high_power_perturbation_strength': self.high_power_perturbation_strength,
            'generation': self.generation,
            'events_count': len(self.events)
        }