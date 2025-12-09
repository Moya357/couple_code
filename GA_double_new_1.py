import numpy as np
import matplotlib.pyplot as plt
import json
import time
import os
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Callable, Any
import copy
from hardware_adapter import HardwareAdapter
from high_power_keep import HighPowerKeepMode  # 导入新的高功率保持模式模块

# 设置中文字体，解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 优化阶段枚举
# =============================================================================

class OptimizationPhase(Enum):
    """优化阶段枚举"""
    BOTH_ACTIVE = "both_active"      # 两端同时优化（正常模式）
    BOTH_FIXED = "both_fixed"        # 高功率保持模式

# =============================================================================
# 完整的双端遗传算法优化器
# =============================================================================

class DualEndGeneticAlgorithmOptimizer:
    """双端光纤耦合对准优化器 - 管理A、B两端的协同优化"""
    
    def __init__(self, config: dict, hardware_adapter: HardwareAdapter):
        """
        初始化双端遗传算法优化器
        
        参数:
            config: 配置字典，包含算法参数
            hardware_adapter: 硬件适配器实例
        """
        self.config = config
        self.hardware_adapter = hardware_adapter
        
        # 从GUI获取选择的变量
        self.selected_variables_A = config.get('selected_variables_A', ['x', 'y', 'z', 'rx', 'ry'])
        self.selected_variables_B = config.get('selected_variables_B', ['x', 'y', 'z', 'rx', 'ry'])
        
        # 从GUI获取算法参数
        self.population_size = config.get('population_size', 30)
        self.normal_population_size = self.population_size  # 保存正常种群大小
        self.generations = config.get('generations', 200)
        self.gene_mutation_rate = config.get('gene_mutation_rate', 0.15)  # 基因变异率
        self.gene_crossover_rate = config.get('gene_crossover_rate', 0.8)  # 基因交叉率
        self.chromosome_crossover_rate = config.get('chromosome_crossover_rate', 0.2)  # 染色体交叉率
        self.elite_size = config.get('elite_size', 4)
        self.tournament_size = config.get('tournament_size', 3)
        
        # 从GUI获取自适应参数
        self.adaptive_mutation_rate = config.get('adaptive_mutation_rate', True)
        self.adaptive_crossover_rate = config.get('adaptive_crossover_rate', True)
        
        # 从GUI获取收敛检测参数
        self.convergence_threshold_percent = config.get('convergence_threshold', 0.05)
        self.convergence_patience = config.get('convergence_patience', 8)
        self.enhanced_exploration_max = config.get('enhanced_exploration_max', 3)  # 修改为3次
        self.enhanced_mutation_rate = config.get('enhanced_mutation_rate', 0.7)
        
        # 从GUI获取高功率保持模式参数
        self.high_power_population_size = config.get('high_power_population_size', 20)  # 高功率模式种群大小
        self.high_power_mutation_rate = config.get('high_power_mutation_rate', 0.05)  # 高功率模式变异率
        self.high_power_crossover_rate = config.get('high_power_crossover_rate', 0.3)  # 高功率模式交叉率
        self.fitness_variance_threshold = config.get('fitness_variance_threshold', 0.005)
        
        # 新增：高功率保持模式小范围搜索参数
        self.high_power_search_range_percent = config.get('high_power_search_range_percent', 0.05)  # 5%的搜索范围
        self.high_power_perturbation_strength = config.get('high_power_perturbation_strength', 0.01)  # 克隆扰动强度
        
        # 位置锁定参数
        self.lock_mode_threshold = config.get('lock_mode_threshold', 0.001)  # 修改为0.1% = 0.001
        self.lock_mode_activated = False
        self.lock_position_A = None
        self.lock_position_B = None
        self.lock_fitness = 0.0
        self.lock_callback = None
        self.lock_population_A = None  # 保存锁定时的种群
        self.lock_population_B = None  # 保存锁定时的种群
        
        # 从GUI获取精英保护参数
        self.elite_protection = config.get('elite_protection', True)
        self.elite_clone_rate = config.get('elite_clone_rate', 0.25)
        
        # 从GUI获取其他参数
        self.light_threshold = config.get('light_threshold', 0.2)
        
        # 收敛状态跟踪
        self.convergence_counter = 0
        self.local_convergence_count = 0  # 局部收敛计数器
        self.best_fitness_memory = None
        self.best_individual_A_memory = None
        self.best_individual_B_memory = None
        self.is_enhanced_exploration = False
        self.enhanced_exploration_counter = 0
        self.original_mutation_rate = self.gene_mutation_rate
        self.enhanced_exploration_history = []
        
        # 收敛状态
        self.converged = False
        self.final_convergence = False
        self.high_power_keep_mode = False
        self.high_power_mode = None  # 新增：高功率保持模式实例
        
        # 从GUI获取搜索范围
        self.search_range_A = config.get('search_range_A', {
            'x': (0, 30), 'y': (0, 30), 'z': (0, 30), 'rx': (0.0, 0.03), 'ry': (0.0, 0.03)
        })
        self.search_range_B = config.get('search_range_B', {
            'x': (0, 30), 'y': (0, 30), 'z': (0, 30), 'rx': (0.0, 0.03), 'ry': (0.0, 0.03)
        })
        
        # 优化状态
        self.is_running = False
        self.optimization_phase = OptimizationPhase.BOTH_ACTIVE
        self.light_detected = False
        
        # 种群和最佳解
        self.population_A = None
        self.population_B = None
        self.best_individual_A = None
        self.best_individual_B = None
        self.best_fitness = -np.inf
        
        # 历史记录
        self.history = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'best_individual_A': [],
            'best_individual_B': [],
            'optimization_phase': [],
            'evaluation_count': 0,
            'search_history': [],
            'population_diversity_A': [],
            'population_diversity_B': [],
            'convergence_status': [],
            'mutation_rate_history': [],
            'enhanced_exploration_events': [],
            'lock_events': [],
            'selected_variables_A': self.selected_variables_A,
            'selected_variables_B': self.selected_variables_B,
        }
        
        # 回调函数
        self.progress_callback = None
        self.finished_callback = None
        self.convergence_callback = None
        self.lock_callback = None
        self.request_parameters_callback = None
        
        print(f"双端优化器初始化完成")
        print(f"A端优化变量: {self.selected_variables_A}")
        print(f"B端优化变量: {self.selected_variables_B}")
        print(f"位置锁定阈值: {self.lock_mode_threshold*100}%")
        print(f"基因变异率: {self.gene_mutation_rate}, 基因交叉率: {self.gene_crossover_rate}")
        print(f"染色体交叉率: {self.chromosome_crossover_rate}")
        print(f"高功率保持搜索范围: ±{self.high_power_search_range_percent*100}%")
        print(f"高功率保持克隆扰动强度: {self.high_power_perturbation_strength}")


    # =============================================================================
    # 动态参数更新功能
    # =============================================================================
    
    def update_parameters_from_gui(self, new_params: dict):
        """
        从GUI动态更新优化参数
        
        参数:
            new_params: 包含更新参数的字典
        """
        if not new_params:
            return
        
        update_count = 0
        
        # 1. 遗传算法参数
        if 'gene_mutation_rate' in new_params:
            new_rate = float(new_params['gene_mutation_rate'])
            if 0 <= new_rate <= 1:
                self.gene_mutation_rate = new_rate
                update_count += 1
                print(f"  基因变异率更新为: {new_rate}")
        
        if 'gene_crossover_rate' in new_params:
            new_rate = float(new_params['gene_crossover_rate'])
            if 0 <= new_rate <= 1:
                self.gene_crossover_rate = new_rate
                update_count += 1
                print(f"  基因交叉率更新为: {new_rate}")
                
        if 'chromosome_crossover_rate' in new_params:
            new_rate = float(new_params['chromosome_crossover_rate'])
            if 0 <= new_rate <= 1:
                self.chromosome_crossover_rate = new_rate
                update_count += 1
                print(f"  染色体交叉率更新为: {new_rate}")
        
        if 'population_size' in new_params:
            new_size = int(new_params['population_size'])
            if new_size >= 5 and new_size != self.population_size:
                # 注意：调整种群大小需要在下一代的初始化时生效
                self.population_size = new_size
                self.normal_population_size = new_size
                update_count += 1
                print(f"  种群大小更新为: {new_size} (下一代生效)")
        
        if 'elite_size' in new_params:
            new_size = int(new_params['elite_size'])
            if new_size >= 1:
                self.elite_size = new_size
                update_count += 1
                print(f"  精英数量更新为: {new_size}")
        
        if 'tournament_size' in new_params:
            new_size = int(new_params['tournament_size'])
            if new_size >= 2:
                self.tournament_size = new_size
                update_count += 1
                print(f"  锦标赛大小更新为: {new_size}")
        
        # 2. 收敛检测参数
        if 'convergence_threshold' in new_params:
            new_threshold = float(new_params['convergence_threshold'])
            if 0.001 <= new_threshold <= 0.5:  # 限制在0.1%到50%之间
                self.convergence_threshold_percent = new_threshold
                update_count += 1
                print(f"  收敛阈值更新为: {new_threshold*100}%")
        
        if 'convergence_patience' in new_params:
            new_patience = int(new_params['convergence_patience'])
            if new_patience >= 3:
                self.convergence_patience = new_patience
                update_count += 1
                print(f"  收敛耐心代数更新为: {new_patience}")
        
        if 'enhanced_exploration_max' in new_params:
            new_max = int(new_params['enhanced_exploration_max'])
            if new_max >= 1:
                self.enhanced_exploration_max = new_max
                update_count += 1
                print(f"  增强探索最大次数更新为: {new_max}")
        
        if 'enhanced_mutation_rate' in new_params:
            new_rate = float(new_params['enhanced_mutation_rate'])
            if 0 <= new_rate <= 1:
                self.enhanced_mutation_rate = new_rate
                update_count += 1
                print(f"  增强探索变异率更新为: {new_rate}")
        
        # 3. 高功率保持模式参数
        if 'high_power_population_size' in new_params:
            new_size = int(new_params['high_power_population_size'])
            if new_size >= 5:
                self.high_power_population_size = new_size
                update_count += 1
                print(f"  高功率种群大小更新为: {new_size}")
                
                # 如果高功率模式已激活，更新其参数
                if self.high_power_mode and hasattr(self.high_power_mode, 'update_parameters_from_gui'):
                    self.high_power_mode.update_parameters_from_gui({
                        'high_power_population_size': new_size
                    })
        
        if 'high_power_mutation_rate' in new_params:
            new_rate = float(new_params['high_power_mutation_rate'])
            if 0 <= new_rate <= 1:
                self.high_power_mutation_rate = new_rate
                update_count += 1
                print(f"  高功率变异率更新为: {new_rate}")
        
        if 'high_power_crossover_rate' in new_params:
            new_rate = float(new_params['high_power_crossover_rate'])
            if 0 <= new_rate <= 1:
                self.high_power_crossover_rate = new_rate
                update_count += 1
                print(f"  高功率交叉率更新为: {new_rate}")
        
        # 4. 高功率保持模式小范围搜索参数（新增）
        if 'high_power_search_range_percent' in new_params:
            new_range = float(new_params['high_power_search_range_percent'])
            if 0.001 <= new_range <= 0.2:  # 限制在0.1%到20%之间
                self.high_power_search_range_percent = new_range
                update_count += 1
                print(f"  高功率搜索范围更新为: ±{new_range*100}%")
                
                # 如果高功率模式已激活，更新其参数
                if self.high_power_mode and hasattr(self.high_power_mode, 'update_parameters_from_gui'):
                    self.high_power_mode.update_parameters_from_gui({
                        'high_power_search_range_percent': new_range
                    })
        
        if 'high_power_perturbation_strength' in new_params:
            new_strength = float(new_params['high_power_perturbation_strength'])
            if 0 <= new_strength <= 0.1:  # 限制在0-10%之间
                self.high_power_perturbation_strength = new_strength
                update_count += 1
                print(f"  高功率克隆扰动强度更新为: {new_strength}")
                
                # 如果高功率模式已激活，更新其参数
                if self.high_power_mode and hasattr(self.high_power_mode, 'update_parameters_from_gui'):
                    self.high_power_mode.update_parameters_from_gui({
                        'high_power_perturbation_strength': new_strength
                    })
        
        # 5. 位置锁定参数
        if 'lock_mode_threshold' in new_params:
            new_threshold = float(new_params['lock_mode_threshold'])
            if 0.0001 <= new_threshold <= 0.1:  # 限制在0.01%到10%之间
                self.lock_mode_threshold = new_threshold
                update_count += 1
                print(f"  位置锁定阈值更新为: {new_threshold*100}%")
        
        # 6. 精英保护参数
        if 'elite_protection' in new_params:
            self.elite_protection = bool(new_params['elite_protection'])
            update_count += 1
            print(f"  精英保护更新为: {self.elite_protection}")
        
        if 'elite_clone_rate' in new_params:
            new_rate = float(new_params['elite_clone_rate'])
            if 0 <= new_rate <= 1:
                self.elite_clone_rate = new_rate
                update_count += 1
                print(f"  精英克隆率更新为: {new_rate}")
        
        # 7. 自适应参数
        if 'adaptive_mutation_rate' in new_params:
            self.adaptive_mutation_rate = bool(new_params['adaptive_mutation_rate'])
            update_count += 1
            print(f"  自适应变异率更新为: {self.adaptive_mutation_rate}")
        
        if 'adaptive_crossover_rate' in new_params:
            self.adaptive_crossover_rate = bool(new_params['adaptive_crossover_rate'])
            update_count += 1
            print(f"  自适应交叉率更新为: {self.adaptive_crossover_rate}")
        
        # 8. 光阈值参数
        if 'light_threshold' in new_params:
            new_threshold = float(new_params['light_threshold'])
            if new_threshold >= 0:
                self.light_threshold = new_threshold
                update_count += 1
                print(f"  光检测阈值更新为: {new_threshold} mW")
        
        # 记录参数更新事件
        if update_count > 0:
            update_event = {
                'event_type': 'parameters_updated_from_gui',
                'timestamp': datetime.now().isoformat(),
                'updated_parameters': new_params,
                'update_count': update_count,
                'current_generation': len(self.history['generations']) if hasattr(self, 'history') else 0
            }
            self.history['enhanced_exploration_events'].append(update_event)
            
            # 通知GUI参数已更新
            if self.progress_callback:
                self.progress_callback({
                    'type': 'parameters_updated',
                    'updated_parameters': new_params,
                    'update_count': update_count,
                    'timestamp': datetime.now().isoformat(),
                    'message': f"成功更新 {update_count} 个参数"
                })
        
        print(f"参数更新完成，更新了 {update_count} 个参数")
    # 在 DualEndGeneticAlgorithmOptimizer 类中添加

    def update_high_power_parameters(self, new_params: dict) -> Tuple[bool, str]:
        """
        更新高功率保持模式参数 - GUI兼容方法
        
        参数:
            new_params: 新参数的字典
            
        返回:
            (是否成功, 消息)
        """
        try:
            update_count = 0
            validation_errors = []
            
            # 验证并更新参数
            if 'high_power_search_range_percent' in new_params:
                new_range = float(new_params['high_power_search_range_percent'])
                if 0.001 <= new_range <= 0.2:
                    self.high_power_search_range_percent = new_range
                    update_count += 1
                    print(f"高功率搜索范围更新为: ±{new_range*100}%")
                else:
                    validation_errors.append(f"高功率搜索范围必须在0.1%-20%之间，当前值: {new_range*100}%")
            
            if 'high_power_perturbation_strength' in new_params:
                new_strength = float(new_params['high_power_perturbation_strength'])
                if 0 <= new_strength <= 0.1:
                    self.high_power_perturbation_strength = new_strength
                    update_count += 1
                    print(f"高功率克隆扰动强度更新为: {new_strength}")
                else:
                    validation_errors.append(f"克隆扰动强度必须在0-0.1之间，当前值: {new_strength}")
            
            if 'high_power_population_size' in new_params:
                new_size = int(new_params['high_power_population_size'])
                if new_size >= 5:
                    self.high_power_population_size = new_size
                    update_count += 1
                    print(f"高功率种群大小更新为: {new_size}")
                else:
                    validation_errors.append(f"高功率种群大小必须≥5，当前值: {new_size}")
            
            if 'high_power_mutation_rate' in new_params:
                new_rate = float(new_params['high_power_mutation_rate'])
                if 0 <= new_rate <= 1:
                    self.high_power_mutation_rate = new_rate
                    update_count += 1
                    print(f"高功率变异率更新为: {new_rate}")
                else:
                    validation_errors.append(f"高功率变异率必须在0-1之间，当前值: {new_rate}")
            
            if 'high_power_crossover_rate' in new_params:
                new_rate = float(new_params['high_power_crossover_rate'])
                if 0 <= new_rate <= 1:
                    self.high_power_crossover_rate = new_rate
                    update_count += 1
                    print(f"高功率交叉率更新为: {new_rate}")
                else:
                    validation_errors.append(f"高功率交叉率必须在0-1之间，当前值: {new_rate}")
            
            # 如果高功率模式已激活，同时更新高功率模式实例
            if self.high_power_mode and hasattr(self.high_power_mode, 'update_parameters_from_gui'):
                # 只传递高功率相关的参数
                high_power_params = {k: v for k, v in new_params.items() 
                                if k.startswith('high_power_')}
                if high_power_params:
                    self.high_power_mode.update_parameters_from_gui(high_power_params)
            
            # 更新优化器状态（如果当前处于高功率保持模式）
            if self.high_power_keep_mode:
                self.population_size = self.high_power_population_size
                self.gene_mutation_rate = self.high_power_mutation_rate
                self.gene_crossover_rate = self.high_power_crossover_rate
            
            if validation_errors:
                error_msg = "\n".join(validation_errors)
                return False, f"参数验证失败: {error_msg}"
            
            if update_count > 0:
                return True, f"成功更新 {update_count} 个高功率保持模式参数"
            else:
                return True, "没有参数需要更新"
                
        except ValueError as e:
            return False, f"参数格式错误: {str(e)}"
        except Exception as e:
            return False, f"更新参数时发生错误: {str(e)}"
    def enhanced_convergence_check(self, current_best_fitness: float, 
                             population_A: np.ndarray, population_B: np.ndarray,
                             current_fitness: np.ndarray, generation: int) -> Tuple[bool, bool]:
        """
        增强的收敛检测
        返回: (是否检测到收敛, 是否全局收敛)
        """
        if len(self.history['best_fitness']) < 3:
            return False, False
        
        recent_fitness = self.history['best_fitness'][-3:]
        max_recent = max(recent_fitness)
        min_recent = min(recent_fitness)
        
        if max_recent > 0:
            change_percent = (max_recent - min_recent) / max_recent
        else:
            change_percent = 1.0
        
        convergence_detected = change_percent < self.convergence_threshold_percent
        
        convergence_record = {
            'generation': generation,
            'recent_fitness': recent_fitness,
            'change_percent': change_percent,
            'convergence_detected': convergence_detected,
            'enhanced_exploration_count': self.enhanced_exploration_counter,
            'is_enhanced_exploration': self.is_enhanced_exploration,
            'local_convergence_count': self.local_convergence_count,
            'timestamp': datetime.now().isoformat()
        }
        self.history['convergence_status'].append(convergence_record)
        
        if convergence_detected and not self.is_enhanced_exploration and not self.final_convergence:
            # 检测到局部收敛
            self.local_convergence_count += 1
            print(f"第{generation}代: 检测到局部收敛 (第{self.local_convergence_count}次)")
            
            if self.local_convergence_count <= self.enhanced_exploration_max:
                # 开始增强探索
                self.start_enhanced_exploration(current_best_fitness, generation)
                return True, False
            else:
                # 局部收敛次数超过阈值，检测全局收敛
                print(f"第{generation}代: 局部收敛次数达到{self.local_convergence_count}次，进入全局收敛状态")
                
                # 返回全局收敛信号，由run方法处理后续逻辑
                return True, True
        
        elif self.is_enhanced_exploration:
            # 处理增强探索阶段
            return self.handle_enhanced_exploration(current_best_fitness, generation)
        
        return False, False

    def start_enhanced_exploration(self, current_best_fitness: float, generation: int):
        """开始增强探索阶段"""
        self.is_enhanced_exploration = True
        self.enhanced_exploration_counter += 1
        self.original_mutation_rate = self.gene_mutation_rate
        self.gene_mutation_rate = self.enhanced_mutation_rate
        self.best_fitness_memory = current_best_fitness
        
        exploration_event = {
            'event_type': 'start_enhanced_exploration',
            'generation': generation,
            'exploration_count': self.enhanced_exploration_counter,
            'local_convergence_count': self.local_convergence_count,
            'original_mutation_rate': self.original_mutation_rate,
            'enhanced_mutation_rate': self.gene_mutation_rate,
            'best_fitness': current_best_fitness,
            'timestamp': datetime.now().isoformat()
        }
        self.history['enhanced_exploration_events'].append(exploration_event)
        
        print(f"第{generation}代: 开始第{self.enhanced_exploration_counter}次增强探索")
        print(f"  变异率从{self.original_mutation_rate}提高到{self.gene_mutation_rate}")

    def handle_enhanced_exploration(self, current_best_fitness: float, generation: int) -> Tuple[bool, bool]:
        """处理增强探索阶段"""
        improvement_percent = 0
        if self.best_fitness_memory > 0:
            improvement_percent = (current_best_fitness - self.best_fitness_memory) / self.best_fitness_memory
        
        improvement_found = improvement_percent > 0.05  # 改进大于5%
        
        if improvement_found:
            self.end_enhanced_exploration(generation, True, current_best_fitness, improvement_percent)
            print(f"第{generation}代: 增强探索找到更好解，改进{improvement_percent*100:.2f}%，继续优化")
            return False, False
        else:
            if self.enhanced_exploration_counter >= self.enhanced_exploration_max:
                self.end_enhanced_exploration(generation, False, current_best_fitness, improvement_percent)
                print(f"第{generation}代: 经过{self.enhanced_exploration_counter}次增强探索未找到足够好的解，进入全局收敛")
                return True, True
            else:
                return False, False

    def end_enhanced_exploration(self, generation: int, improvement_found: bool, 
                                current_best_fitness: float, improvement_percent: float):
        """结束增强探索阶段"""
        self.is_enhanced_exploration = False
        self.gene_mutation_rate = self.original_mutation_rate
        
        exploration_event = {
            'event_type': 'end_enhanced_exploration',
            'generation': generation,
            'exploration_count': self.enhanced_exploration_counter,
            'improvement_found': improvement_found,
            'improvement_percent': improvement_percent,
            'final_mutation_rate': self.gene_mutation_rate,
            'best_fitness': current_best_fitness,
            'timestamp': datetime.now().isoformat()
        }
        self.history['enhanced_exploration_events'].append(exploration_event)
        
        if improvement_found:
            print(f"第{generation}代: 增强探索找到更好解，恢复变异率为{self.gene_mutation_rate}")
        else:
            print(f"第{generation}代: 增强探索结束，恢复变异率为{self.gene_mutation_rate}")

    def enter_enhanced_high_power_mode(self, best_individual_A: np.ndarray, best_individual_B: np.ndarray, 
                                      best_fitness: float, best_individuals_history: list = None):
        """
        进入增强的高功率保持模式
        
        参数:
            best_individual_A: A端最佳个体
            best_individual_B: B端最佳个体
            best_fitness: 最佳适应度
            best_individuals_history: 最佳个体历史记录
        """
        self.high_power_keep_mode = True
        self.optimization_phase = OptimizationPhase.BOTH_FIXED
        
        # 创建高功率保持模式实例
        high_power_config = {
            'high_power_population_size': self.high_power_population_size,
            'high_power_mutation_rate': self.high_power_mutation_rate,
            'high_power_crossover_rate': self.high_power_crossover_rate,
            'high_power_search_range_percent': self.high_power_search_range_percent,
            'high_power_perturbation_strength': self.high_power_perturbation_strength
        }
        
        self.high_power_mode = HighPowerKeepMode(
            high_power_config,
            self.selected_variables_A,
            self.selected_variables_B,
            self.search_range_A,
            self.search_range_B
        )
        
        # 初始化高功率保持模式
        self.high_power_mode.initialize(
            best_individual_A,
            best_individual_B,
            best_fitness,
            best_individuals_history
        )
        
        # 减小种群大小至高功率保持模式的大小
        self.population_size = self.high_power_population_size
        
        # 降低变异率和交叉率
        self.gene_mutation_rate = self.high_power_mutation_rate
        self.gene_crossover_rate = self.high_power_crossover_rate
        # 新增：降低染色体交叉率
        self.chromosome_crossover_rate = 0.1  # 高功率模式下降低染色体交叉率
        # 记录高功率保持模式开始事件
        current_generation = len(self.history['generations'])
        high_power_event = {
            'event_type': 'enter_enhanced_high_power_mode',
            'generation': current_generation,
            'best_fitness': best_fitness,
            'high_power_search_range_percent': self.high_power_search_range_percent,
            'high_power_perturbation_strength': self.high_power_perturbation_strength,
            'population_size': self.population_size,
            'mutation_rate': self.gene_mutation_rate,
            'crossover_rate': self.gene_crossover_rate,
            'timestamp': datetime.now().isoformat()
        }
        self.history['enhanced_exploration_events'].append(high_power_event)
        
        # 通知GUI进入高功率保持模式
        if self.progress_callback:
            self.progress_callback({
                'type': 'enhanced_high_power_mode',
                'converged': self.final_convergence,
                'high_power_keep_mode': self.high_power_keep_mode,
                'population_size': self.population_size,
                'mutation_rate': self.gene_mutation_rate,
                'crossover_rate': self.gene_crossover_rate,
                'high_power_search_range_percent': self.high_power_search_range_percent,
                'high_power_perturbation_strength': self.high_power_perturbation_strength,
                'best_fitness': best_fitness,
                'timestamp': datetime.now().isoformat(),
                'message': f"系统已全局收敛，进入增强高功率保持模式（搜索范围: ±{self.high_power_search_range_percent*100}%）"
            })
        
        print(f"进入增强高功率保持模式，最佳功率: {best_fitness:.6f}mW")
        print(f"搜索范围: ±{self.high_power_search_range_percent*100}%")
        print(f"克隆扰动强度: {self.high_power_perturbation_strength}")
        print(f"种群大小: {self.population_size}")
        print(f"变异率: {self.gene_mutation_rate}")
        print(f"交叉率: {self.gene_crossover_rate}")

    # =============================================================================
    # 位置锁定模式功能
    # =============================================================================
    
    def activate_lock_mode(self):
        """激活位置锁定模式"""
        self.lock_mode_activated = True
        self.lock_position_A = None
        self.lock_position_B = None
        self.lock_fitness = 0.0
        print("位置锁定模式已激活，将在满足条件时停止优化并保持当前位置")
        print(f"锁定阈值: {self.lock_mode_threshold*100}%")
        
        # 记录锁定模式激活事件
        lock_event = {
            'event_type': 'lock_mode_activated',
            'timestamp': datetime.now().isoformat(),
            'lock_mode_threshold': self.lock_mode_threshold
        }
        self.history['lock_events'].append(lock_event)
        
        # 通知GUI位置锁定模式已激活
        if self.progress_callback:
            self.progress_callback({
                'type': 'lock_mode_activated',
                'lock_mode_activated': self.lock_mode_activated,
                'lock_mode_threshold': self.lock_mode_threshold,
                'timestamp': datetime.now().isoformat(),
                'message': f"位置锁定模式已激活，锁定阈值: {self.lock_mode_threshold*100}%"
            })
    def start_high_power_keep_mode_from_gui(self, center_individual_A: np.ndarray = None, 
                                        center_individual_B: np.ndarray = None,
                                        current_fitness: float = None):
        """
        从GUI启动高功率保持模式
        以当前坐标为中心点，使用较小的搜索范围和扰动生成种群
        
        参数:
            center_individual_A: A端中心个体（如为None则使用当前最佳个体）
            center_individual_B: B端中心个体（如为None则使用当前最佳个体）
            current_fitness: 当前适应度（如为None则使用当前最佳适应度）
        """
        # 确定中心点和适应度
        if center_individual_A is None and self.best_individual_A is not None:
            center_individual_A = self.best_individual_A.copy()
        
        if center_individual_B is None and self.best_individual_B is not None:
            center_individual_B = self.best_individual_B.copy()
        
        if current_fitness is None and self.best_fitness > 0:
            current_fitness = self.best_fitness
        
        if center_individual_A is None or center_individual_B is None:
            print("警告：无法获取中心个体，无法启动高功率保持模式")
            return False
        
        print("从GUI启动高功率保持模式")
        
        # 停止当前优化
        self.is_running = False
        
        # 设置高功率保持模式参数
        self.high_power_keep_mode = True
        self.optimization_phase = OptimizationPhase.BOTH_FIXED
        
        # 重置种群大小为高功率模式大小
        self.population_size = self.high_power_population_size
        
        # 设置高功率模式特有的参数
        self.gene_mutation_rate = self.high_power_mutation_rate
        self.gene_crossover_rate = self.high_power_crossover_rate
        self.chromosome_crossover_rate = 0.1  # 高功率模式下降低染色体交叉率
        
        # 创建以中心点为基础的小范围种群
        self.population_A = self._create_population_around_center(
            center_individual_A, 
            self.selected_variables_A, 
            self.search_range_A,
            self.high_power_search_range_percent
        )
        
        self.population_B = self._create_population_around_center(
            center_individual_B, 
            self.selected_variables_B, 
            self.search_range_B,
            self.high_power_search_range_percent
        )
        
        # 记录高功率保持模式启动事件
        high_power_event = {
            'event_type': 'high_power_mode_started_from_gui',
            'timestamp': datetime.now().isoformat(),
            'center_individual_A': center_individual_A.tolist(),
            'center_individual_B': center_individual_B.tolist(),
            'current_fitness': current_fitness,
            'high_power_search_range_percent': self.high_power_search_range_percent,
            'high_power_perturbation_strength': self.high_power_perturbation_strength,
            'population_size': self.population_size,
            'gene_mutation_rate': self.gene_mutation_rate,
            'gene_crossover_rate': self.gene_crossover_rate,
            'chromosome_crossover_rate': self.chromosome_crossover_rate
        }
        self.history['enhanced_exploration_events'].append(high_power_event)
        
        # 通知GUI
        if self.progress_callback:
            self.progress_callback({
                'type': 'high_power_mode_started',
                'center_position_A': {f'A_{var}': center_individual_A[i] for i, var in enumerate(self.selected_variables_A)},
                'center_position_B': {f'B_{var}': center_individual_B[i] for i, var in enumerate(self.selected_variables_B)},
                'current_fitness': current_fitness,
                'high_power_search_range_percent': self.high_power_search_range_percent,
                'high_power_perturbation_strength': self.high_power_perturbation_strength,
                'timestamp': datetime.now().isoformat(),
                'message': f"从GUI启动高功率保持模式，搜索范围: ±{self.high_power_search_range_percent*100}%"
            })
        
        print(f"高功率保持模式已启动，搜索范围: ±{self.high_power_search_range_percent*100}%")
        print(f"中心功率: {current_fitness:.6f}mW")
        
        return True

    def _create_population_around_center(self, center_individual: np.ndarray, 
                                    selected_variables: List[str], 
                                    search_range: Dict,
                                    search_range_percent: float) -> np.ndarray:
        """
        创建以中心点为基础的小范围种群
        
        参数:
            center_individual: 中心个体
            selected_variables: 选择的变量
            search_range: 原始搜索范围
            search_range_percent: 搜索范围百分比
            
        返回:
            population: 新种群
        """
        population = np.zeros((self.population_size, len(selected_variables)))
        
        for i in range(self.population_size):
            if i == 0:
                # 第一个个体就是中心点（不加扰动）
                population[i] = center_individual.copy()
            else:
                # 其他个体添加小范围扰动
                perturbed_individual = center_individual.copy()
                
                for j, var in enumerate(selected_variables):
                    lower, upper = search_range[var]
                    range_size = upper - lower
                    
                    # 计算扰动幅度
                    perturbation_range = range_size * self.high_power_perturbation_strength
                    perturbation = np.random.normal(0, perturbation_range)
                    perturbed_individual[j] += perturbation
                    
                    # 确保在搜索范围内
                    perturbed_individual[j] = np.clip(perturbed_individual[j], lower, upper)
                
                population[i] = perturbed_individual
        
        return population
    def check_lock_mode_condition(self, current_fitness: float, current_individual_A: np.ndarray, 
                            current_individual_B: np.ndarray) -> bool:
        """
        检查位置锁定条件 - 当前个体适应度与最佳适应度的差值小于锁定阈值
        """
        if not self.lock_mode_activated:
            return False
        
        if self.best_fitness_memory is None or self.best_fitness_memory <= 0:
            return False
        
        # 计算与最佳适应度的偏差
        fitness_deviation = abs(current_fitness - self.best_fitness_memory) / self.best_fitness_memory
        
        # 检查偏差是否在阈值内
        if fitness_deviation <= self.lock_mode_threshold:
            print(f"位置锁定条件满足：当前功率{current_fitness:.6f}mW，最佳功率{self.best_fitness_memory:.6f}mW，偏差{fitness_deviation*100:.2f}%")
            
            # 保存当前满足条件的个体位置
            self.lock_position_A = current_individual_A.copy()
            self.lock_position_B = current_individual_B.copy()
            self.lock_fitness = current_fitness
            
            # 保存当前种群
            self.lock_population_A = self.population_A.copy()
            self.lock_population_B = self.population_B.copy()
            
            # 记录锁定事件
            lock_event = {
                'event_type': 'position_locked',
                'timestamp': datetime.now().isoformat(),
                'fitness': current_fitness,
                'fitness_deviation_percent': fitness_deviation * 100,
                'position_A': {f'A_{var}': current_individual_A[i] for i, var in enumerate(self.selected_variables_A)},
                'position_B': {f'B_{var}': current_individual_B[i] for i, var in enumerate(self.selected_variables_B)}
            }
            self.history['lock_events'].append(lock_event)
            
            # 调用锁定回调
            if self.lock_callback:
                full_position_dict = self.get_full_position_dict(current_individual_A, current_individual_B)
                self.lock_callback(full_position_dict, current_fitness)
            
            # 通知GUI位置已锁定
            if self.progress_callback:
                self.progress_callback({
                    'type': 'position_locked',
                    'lock_position_A': self.lock_position_A.tolist() if self.lock_position_A is not None else None,
                    'lock_position_B': self.lock_position_B.tolist() if self.lock_position_B is not None else None,
                    'lock_fitness': self.lock_fitness,
                    'timestamp': datetime.now().isoformat(),
                    'message': f"位置已锁定，锁定功率: {self.lock_fitness:.6f}mW"
                })
            
            # 停止优化
            self.is_running = False
            print("位置锁定条件满足，停止优化")
            
            return True
        
        return False

    # =============================================================================
    # 核心遗传算法功能
    # =============================================================================

    def get_full_position_dict(self, individual_A: np.ndarray, individual_B: np.ndarray) -> Dict:
        """
        根据A、B两端的个体构建完整的位置字典
        """
        position_dict = {}
        
        # 构建A端位置
        idx_A = 0
        for var in ['x', 'y', 'z', 'rx', 'ry']:
            if var in self.selected_variables_A:
                position_dict[f'A_{var}'] = individual_A[idx_A]
                idx_A += 1
            else:
                lower, upper = self.search_range_A[var]
                position_dict[f'A_{var}'] = (lower + upper) / 2
        
        # 构建B端位置  
        idx_B = 0
        for var in ['x', 'y', 'z', 'rx', 'ry']:
            if var in self.selected_variables_B:
                position_dict[f'B_{var}'] = individual_B[idx_B]
                idx_B += 1
            else:
                lower, upper = self.search_range_B[var]
                position_dict[f'B_{var}'] = (lower + upper) / 2
                
        return position_dict

    def initialize_populations(self):
        """初始化A、B两端的种群"""
        # 正常模式：随机初始化
        self.population_A = self._initialize_single_population(self.selected_variables_A, self.search_range_A)
        self.population_B = self._initialize_single_population(self.selected_variables_B, self.search_range_B)
        
        print(f"A端种群初始化完成，维度: {self.population_A.shape}")
        print(f"B端种群初始化完成，维度: {self.population_B.shape}")

    def _initialize_single_population(self, selected_variables: List[str], search_range: Dict) -> np.ndarray:
        """初始化单个种群"""
        population = np.zeros((self.population_size, len(selected_variables)))
        
        for i, var in enumerate(selected_variables):
            lower, upper = search_range[var]
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
            
        return population

    def get_power_value(self, power_result):
        """
        从功率计返回结果中提取功率值
        支持新旧两种格式
        """
        if power_result is None:
            return 0.0
            
        if isinstance(power_result, dict):
            # 新格式：字典包含功率值和其他信息
            power_value = power_result.get("power", 0.0)
            
            # 可选：记录详细的功率信息用于调试
            if hasattr(self, 'debug_mode') and self.debug_mode:
                engineering_notation = power_result.get("engineering_notation", "N/A")
                scientific_notation = power_result.get("scientific_notation", "N/A")
                print(f"功率详情: {engineering_notation} ({scientific_notation})")
                
            return power_value
        else:
            # 旧格式：直接返回功率数值
            return float(power_result)

    def evaluate_dual_fitness(self, individual_A: np.ndarray, individual_B: np.ndarray) -> float:
        """
        评估A、B两端组合的适应度
        """
        position_dict = self.get_full_position_dict(individual_A, individual_B)
        
        try:
            # 使用硬件适配器测量功率
            power_result = self.hardware_adapter.measure_power_average(position_dict)
            
            # 从功率结果中提取功率值
            power = self.get_power_value(power_result)
            
            # 检测通光
            if not self.light_detected and power >= self.light_threshold:
                self.light_detected = True
                print(f"🎉 检测到通光! 功率: {power:.6f} mW")
            
            # 检查位置锁定条件
            if self.lock_mode_activated:
                if self.check_lock_mode_condition(power, individual_A, individual_B):
                    # 位置锁定条件满足，停止当前评估
                    return power
            
            # 记录评估历史
            self.history['evaluation_count'] += 1
            evaluation_record = {
                'position_A': {f'A_{var}': individual_A[i] for i, var in enumerate(self.selected_variables_A)},
                'position_B': {f'B_{var}': individual_B[i] for i, var in enumerate(self.selected_variables_B)},
                'power': power,
                'power_result': power_result if isinstance(power_result, dict) else {'power': power},
                'timestamp': datetime.now().isoformat(),
                'evaluation_index': self.history['evaluation_count'],
                'optimization_phase': self.optimization_phase.value,
                'light_detected': self.light_detected
            }
            self.history['search_history'].append(evaluation_record)
            
            # 发送评估数据到GUI
            if self.progress_callback:
                self.progress_callback({
                    'type': 'evaluation',
                    'evaluation_data': {
                        'evaluation_count': self.history['evaluation_count'],
                        'power': power,
                        'position_A': evaluation_record['position_A'],
                        'position_B': evaluation_record['position_B'],
                        'individual_A': individual_A.tolist(),
                        'individual_B': individual_B.tolist(),
                        'timestamp': datetime.now().isoformat(),
                        'optimization_phase': self.optimization_phase.value,
                        'light_detected': self.light_detected
                    }
                })
            
            return power
            
        except Exception as e:
            print(f"评估失败: {e}")
            return 0.0

    def evaluate_population_pair(self, population_A: np.ndarray, population_B: np.ndarray) -> np.ndarray:
        """
        评估种群对的适应度
        """
        fitness = np.zeros(len(population_A))
        
        for i in range(len(population_A)):
            if not self.is_running:
                break
                
            individual_A = population_A[i]
            individual_B = population_B[i]
            fitness[i] = self.evaluate_dual_fitness(individual_A, individual_B)
            
        return fitness

    def create_new_population_enhanced(self, population_A: np.ndarray, population_B: np.ndarray, 
                                 fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        增强的种群生成机制
        包含染色体交叉操作（A端和B端染色体交叉）
        """
        new_population_A = np.zeros_like(population_A)
        new_population_B = np.zeros_like(population_B)
        
        # 1. 精英保留
        elite_count = min(self.elite_size, len(population_A))
        if elite_count > 0:
            elite_indices = np.argsort(fitness)[-elite_count:]
            new_population_A[:elite_count] = population_A[elite_indices]
            new_population_B[:elite_count] = population_B[elite_indices]
        
        # 2. 锦标赛选择和遗传操作
        current_idx = elite_count
        while current_idx < len(new_population_A):
            # 选择父母
            parent1_idx = self._tournament_selection(fitness, self.tournament_size)
            parent2_idx = self._tournament_selection(fitness, self.tournament_size)
            
            parent1_A = population_A[parent1_idx]
            parent1_B = population_B[parent1_idx]
            parent2_A = population_A[parent2_idx]
            parent2_B = population_B[parent2_idx]
            
            # 决定是否进行染色体交叉
            if np.random.random() < self.chromosome_crossover_rate:
                # 染色体交叉：交换A端和B端染色体，并进行基因操作
                # 个体1的A端与个体2的B端结合形成新个体1
                child1_A = parent1_A.copy()
                child1_B = parent2_B.copy()
                
                # 个体1的B端与个体2的A端结合形成新个体2
                child2_A = parent2_A.copy()
                child2_B = parent1_B.copy()
                
                # 对染色体交叉后的个体进行基因交叉（如果进行染色体交叉）
                if np.random.random() < self.gene_crossover_rate:
                    child1_A, child2_A = self._gene_crossover(child1_A, child2_A)
                    child1_B, child2_B = self._gene_crossover(child1_B, child2_B)
                
                # 对染色体交叉后的个体进行基因变异
                child1_A = self._mutate_genes(child1_A, self.selected_variables_A, self.search_range_A)
                child1_B = self._mutate_genes(child1_B, self.selected_variables_B, self.search_range_B)
                child2_A = self._mutate_genes(child2_A, self.selected_variables_A, self.search_range_A)
                child2_B = self._mutate_genes(child2_B, self.selected_variables_B, self.search_range_B)
            else:
                # 正常基因交叉和变异
                if np.random.random() < self.gene_crossover_rate:
                    child1_A, child2_A = self._gene_crossover(parent1_A, parent2_A)
                    child1_B, child2_B = self._gene_crossover(parent1_B, parent2_B)
                else:
                    child1_A, child2_A = parent1_A.copy(), parent2_A.copy()
                    child1_B, child2_B = parent1_B.copy(), parent2_B.copy()
                
                # 基因变异
                child1_A = self._mutate_genes(child1_A, self.selected_variables_A, self.search_range_A)
                child2_A = self._mutate_genes(child2_A, self.selected_variables_A, self.search_range_A)
                child1_B = self._mutate_genes(child1_B, self.selected_variables_B, self.search_range_B)
                child2_B = self._mutate_genes(child2_B, self.selected_variables_B, self.search_range_B)
            
            # 添加到新种群
            if current_idx < len(new_population_A):
                new_population_A[current_idx] = child1_A
                new_population_B[current_idx] = child1_B
                current_idx += 1
            if current_idx < len(new_population_A):
                new_population_A[current_idx] = child2_A
                new_population_B[current_idx] = child2_B
                current_idx += 1
        
        return new_population_A, new_population_B

    def _tournament_selection(self, fitness: np.ndarray, tournament_size: int) -> int:
        """锦标赛选择"""
        candidates = np.random.choice(len(fitness), tournament_size, replace=False)
        return candidates[np.argmax(fitness[candidates])]

    def _gene_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """基因交叉：模拟二进制交叉"""
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:  # 50%概率进行交叉
                alpha = np.random.random()
                child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]
                
        return child1, child2

    def _mutate_genes(self, individual: np.ndarray, selected_variables: List[str], 
                     search_range: Dict) -> np.ndarray:
        """基因变异操作"""
        mutated = individual.copy()
        
        for i, var in enumerate(selected_variables):
            if np.random.random() < self.gene_mutation_rate:
                lower, upper = search_range[var]
                mutation_strength = (upper - lower) * 0.1
                mutation = np.random.normal(0, mutation_strength)
                mutated[i] += mutation
                mutated[i] = np.clip(mutated[i], lower, upper)
                
        return mutated

    def _apply_small_perturbation(self, individual: np.ndarray, selected_variables: List[str],
                                 search_range: Dict, perturbation_strength: float = 0.01) -> np.ndarray:
        """应用小范围扰动，用于染色体交叉后的微小变异"""
        perturbed = individual.copy()
        
        for i, var in enumerate(selected_variables):
            lower, upper = search_range[var]
            range_size = upper - lower
            perturbation = np.random.normal(0, range_size * perturbation_strength)
            perturbed[i] += perturbation
            perturbed[i] = np.clip(perturbed[i], lower, upper)
                
        return perturbed

    # 修改 run 方法中的收敛处理逻辑
    def run(self):
        """运行双端优化过程"""
        self.is_running = True
        start_time = time.time()
        
        # 初始化种群
        self.initialize_populations()
        
        try:
            for generation in range(1, self.generations + 1):
                if not self.is_running:
                    break
                
                # ========== 在每一代开始前从GUI获取最新参数 ==========
                if hasattr(self, 'request_parameters_callback') and self.request_parameters_callback:
                    try:
                        # 从GUI请求最新的参数
                        new_params = self.request_parameters_callback()
                        if new_params and isinstance(new_params, dict) and new_params:
                            print(f"第{generation}代开始前从GUI获取参数...")
                            self.update_parameters_from_gui(new_params)
                    except Exception as e:
                        print(f"从GUI获取参数失败: {e}")
                        # 继续优化，不中断
                # =======================================================
                
                print(f"\n=== 第{generation}代 ===")
                print(f"当前阶段: {self.optimization_phase.value}")
                print(f"基因变异率: {self.gene_mutation_rate}, 基因交叉率: {self.gene_crossover_rate}, 染色体交叉率: {self.chromosome_crossover_rate}")
                
                # 如果是高功率保持模式，显示高功率参数
                if self.high_power_keep_mode:
                    print(f"高功率保持模式参数 - 种群大小: {self.population_size}, 变异率: {self.gene_mutation_rate}, 交叉率: {self.gene_crossover_rate}")
                    if self.high_power_mode:
                        status = self.high_power_mode.get_status()
                        print(f"搜索范围: ±{self.high_power_search_range_percent*100}%, 中心功率: {status['best_fitness']:.6f}mW")
                
                # 评估种群
                fitness = self.evaluate_population_pair(self.population_A, self.population_B)
                
                # 更新最佳解
                current_best_idx = np.argmax(fitness)
                current_best_fitness = fitness[current_best_idx]
                
                if current_best_fitness > self.best_fitness:
                    self.best_fitness = current_best_fitness
                    self.best_individual_A = self.population_A[current_best_idx].copy()
                    self.best_individual_B = self.population_B[current_best_idx].copy()
                    self.best_individual_A_memory = self.best_individual_A.copy()
                    self.best_individual_B_memory = self.best_individual_B.copy()
                
                # 收敛检测
                if not self.final_convergence:
                    convergence_detected, global_convergence = self.enhanced_convergence_check(
                        current_best_fitness, 
                        self.population_A, 
                        self.population_B, 
                        fitness, 
                        generation
                    )
                    
                    if convergence_detected and global_convergence:
                        print(f"第{generation}代: 检测到全局收敛，进入位置锁定模式")
                        self.final_convergence = True
                        
                        # 保存当前最佳个体为锁定参考
                        self.best_fitness_memory = self.best_fitness
                        self.best_individual_A_memory = self.best_individual_A.copy() if self.best_individual_A is not None else None
                        self.best_individual_B_memory = self.best_individual_B.copy() if self.best_individual_B is not None else None
                        
                        # 激活位置锁定模式，但继续优化
                        self.activate_lock_mode()
                        
                        # 不进入高功率保持模式，继续循环等待锁定条件满足
                        # 注意：这里没有调用 enter_enhanced_high_power_mode，也没有 continue
                        
                        # 通知GUI已全局收敛
                        if self.progress_callback:
                            self.progress_callback({
                                'type': 'global_convergence_detected',
                                'generation': generation,
                                'best_fitness': self.best_fitness,
                                'timestamp': datetime.now().isoformat(),
                                'message': f"检测到全局收敛，进入位置锁定模式等待锁定条件满足"
                            })
                
                # 检查是否已经满足锁定条件（如果已激活锁定模式）
                if self.lock_mode_activated:
                    # 检查当前最佳个体是否满足锁定条件
                    best_idx = np.argmax(fitness)
                    current_individual_A = self.population_A[best_idx]
                    current_individual_B = self.population_B[best_idx]
                    
                    # 这里会调用 check_lock_mode_condition，如果满足条件会停止优化
                    if self.check_lock_mode_condition(current_best_fitness, current_individual_A, current_individual_B):
                        # 锁定条件满足，停止优化
                        break
                
                # 种群生成
                if self.high_power_keep_mode and self.high_power_mode:
                    # 高功率保持模式：使用高功率模式专用的种群生成机制
                    self.population_A, self.population_B = self.high_power_mode.create_new_population(
                        self.population_A, self.population_B, fitness
                    )
                    
                    # 更新搜索中心
                    best_idx = np.argmax(fitness)
                    best_individual_A = self.population_A[best_idx]
                    best_individual_B = self.population_B[best_idx]
                    self.high_power_mode.update_search_center(
                        best_individual_A,
                        best_individual_B,
                        fitness[best_idx]
                    )
                    
                    # 确保使用高功率模式的参数
                    self.gene_mutation_rate = self.high_power_mutation_rate
                    self.gene_crossover_rate = self.high_power_crossover_rate
                    print(f"高功率保持模式参数已生效: 变异率={self.gene_mutation_rate}, 交叉率={self.gene_crossover_rate}")
                else:
                    # 正常模式：使用增强的种群生成机制
                    self.population_A, self.population_B = self.create_new_population_enhanced(
                        self.population_A, self.population_B, fitness
                    )
                
                # 记录历史
                self._update_history(generation, fitness)
                
                # 进度回调到GUI
                if self.progress_callback:
                    # 获取最佳配对
                    if self.best_individual_A is not None and self.best_individual_B is not None:
                        best_individual_A = self.best_individual_A
                        best_individual_B = self.best_individual_B
                    else:
                        best_idx = np.argmax(fitness)
                        best_individual_A = self.population_A[best_idx]
                        best_individual_B = self.population_B[best_idx]
                    
                    best_position = self.get_full_position_dict(best_individual_A, best_individual_B)
                    
                    # 添加高功率保持模式状态信息
                    high_power_status = None
                    if self.high_power_mode:
                        high_power_status = self.high_power_mode.get_status()
                    
                    # 添加位置锁定状态信息
                    lock_status = {
                        'lock_mode_activated': self.lock_mode_activated,
                        'lock_fitness': self.lock_fitness,
                        'lock_position_available': self.lock_position_A is not None and self.lock_position_B is not None,
                        'best_fitness_memory': self.best_fitness_memory
                    }
                    
                    self.progress_callback({
                        'type': 'generation',
                        'generation_data': {
                            'iteration': generation,
                            'total_iterations': self.generations,
                            'current_power': current_best_fitness,
                            'best_power': self.best_fitness,
                            'position_A': {k: v for k, v in best_position.items() if k.startswith('A_')},
                            'position_B': {k: v for k, v in best_position.items() if k.startswith('B_')},
                            'optimization_phase': self.optimization_phase.value,
                            'light_detected': self.light_detected,
                            'converged': self.final_convergence,
                            'enhanced_exploration': self.is_enhanced_exploration,
                            'high_power_keep_mode': self.high_power_keep_mode,
                            'lock_status': lock_status,
                            'local_convergence_count': self.local_convergence_count,
                            'selected_variables_A': self.selected_variables_A,
                            'selected_variables_B': self.selected_variables_B,
                            'timestamp': datetime.now().isoformat(),
                            'high_power_status': high_power_status,
                            'high_power_search_range_percent': self.high_power_search_range_percent,
                            'high_power_perturbation_strength': self.high_power_perturbation_strength,
                            'population_size': self.population_size,
                            'gene_mutation_rate': self.gene_mutation_rate,
                            'gene_crossover_rate': self.gene_crossover_rate,
                            'chromosome_crossover_rate': self.chromosome_crossover_rate
                        }
                    })
            
            # 优化完成
            optimization_time = time.time() - start_time
            
            # 构建结果
            best_position = self.get_full_position_dict(
                self.best_individual_A if self.best_individual_A is not None else 
                np.zeros(len(self.selected_variables_A)),
                self.best_individual_B if self.best_individual_B is not None else 
                np.zeros(len(self.selected_variables_B))
            )
            
            result = {
                'success': True,
                'best_power': self.best_fitness,
                'best_position_A': {k: v for k, v in best_position.items() if k.startswith('A_')},
                'best_position_B': {k: v for k, v in best_position.items() if k.startswith('B_')},
                'total_evaluations': self.history['evaluation_count'],
                'total_generations': len(self.history['generations']),
                'optimization_time': optimization_time,
                'light_detected': self.light_detected,
                'final_phase': self.optimization_phase.value,
                'final_convergence': self.final_convergence,
                'enhanced_exploration_count': self.enhanced_exploration_counter,
                'local_convergence_count': self.local_convergence_count,
                'high_power_keep_mode': self.high_power_keep_mode,
                'lock_mode_activated': self.lock_mode_activated,
                'lock_position_A': {f'A_{var}': self.lock_position_A[i] for i, var in enumerate(self.selected_variables_A)} if self.lock_position_A is not None else None,
                'lock_position_B': {f'B_{var}': self.lock_position_B[i] for i, var in enumerate(self.selected_variables_B)} if self.lock_position_B is not None else None,
                'lock_fitness': self.lock_fitness,
                'selected_variables_A': self.selected_variables_A,
                'selected_variables_B': self.selected_variables_B,
                'high_power_search_range_percent': self.high_power_search_range_percent,
                'high_power_perturbation_strength': self.high_power_perturbation_strength,
                'final_population_size': self.population_size,
                'final_gene_mutation_rate': self.gene_mutation_rate,
                'final_gene_crossover_rate': self.gene_crossover_rate,
                'final_chromosome_crossover_rate': self.chromosome_crossover_rate,
                'history': self.history
            }
            
        except Exception as e:
            print(f"优化过程中出现异常: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'success': False,
                'error': str(e),
                'total_evaluations': self.history['evaluation_count'],
                'selected_variables_A': self.selected_variables_A,
                'selected_variables_B': self.selected_variables_B
            }
            
        # 触发完成回调到GUI
        if self.finished_callback:
            self.finished_callback(result)
            
        self.is_running = False
        return result

    def _update_history(self, generation: int, fitness: np.ndarray):
        """更新历史记录"""
        self.history['generations'].append(generation)
        self.history['best_fitness'].append(np.max(fitness))
        self.history['avg_fitness'].append(np.mean(fitness))
        self.history['optimization_phase'].append(self.optimization_phase.value)
        self.history['mutation_rate_history'].append(self.gene_mutation_rate)
        
        # 记录最佳个体
        best_idx = np.argmax(fitness)
        self.history['best_individual_A'].append(self.population_A[best_idx].copy().tolist())
        self.history['best_individual_B'].append(self.population_B[best_idx].copy().tolist())
        
        # 计算种群多样性
        self.history['population_diversity_A'].append(self._calculate_diversity(self.population_A))
        self.history['population_diversity_B'].append(self._calculate_diversity(self.population_B))

    def _calculate_diversity(self, population: np.ndarray) -> float:
        """计算种群多样性"""
        if len(population) <= 1:
            return 0.0
        return np.mean(np.std(population, axis=0))

    def stop(self):
        """停止优化"""
        self.is_running = False

    def set_callbacks(self, progress_callback=None, finished_callback=None, 
                     convergence_callback=None, lock_callback=None,
                     request_parameters_callback=None):
        """设置回调函数"""
        self.progress_callback = progress_callback
        self.finished_callback = finished_callback
        self.convergence_callback = convergence_callback
        self.lock_callback = lock_callback
        self.request_parameters_callback = request_parameters_callback

# =============================================================================
# 配置和辅助函数
# =============================================================================

def get_dual_end_config():
    """获取双端优化的默认配置"""
    config = {
        'population_size': 30,
        'generations': 200,
        'gene_mutation_rate': 0.15,  # 基因变异率
        'gene_crossover_rate': 0.8,  # 基因交叉率
        'chromosome_crossover_rate': 0.2,  # 染色体交叉率
        'elite_size': 4,
        'tournament_size': 3,
        'convergence_threshold': 0.05,
        'convergence_patience': 8,
        'enhanced_exploration_max': 3,  # 修改为3次
        'enhanced_mutation_rate': 0.7,
        'fitness_variance_threshold': 0.005,
        'adaptive_mutation_rate': True,
        'adaptive_crossover_rate': True,
        'elite_protection': True,
        
        # 高功率保持模式参数
        'high_power_population_size': 20,  # 高功率模式种群大小
        'high_power_mutation_rate': 0.05,  # 高功率模式变异率
        'high_power_crossover_rate': 0.3,  # 高功率模式交叉率
        
        # 新增：高功率保持模式小范围搜索参数
        'high_power_search_range_percent': 0.05,  # 5%的搜索范围
        'high_power_perturbation_strength': 0.01,  # 克隆扰动强度
        # 新增：高功率保持模式动态参数调整
        'high_power_convergence_threshold': 0.01,  # 1%的阈值
        'param_adjustment_rate': 0.5,  # 参数调整幅度
        'min_mutation_rate': 0.01,  # 最小变异率
        'max_mutation_rate': 0.2,   # 最大变异率
        'min_crossover_rate': 0.1,  # 最小交叉率
        'max_crossover_rate': 0.8,  # 最大交叉率
        # 其他参数
        'light_threshold': 0.0002,  #通光阈值
        
        # 位置锁定参数
        'lock_mode_threshold': 0.001,  # 0.1%的阈值
        
        # 搜索范围
        'search_range_A': {
            'x': (0, 30),
            'y': (0, 30),
            'z': (0, 30),
            'rx': (0, 0.03),
            'ry': (0, 0.03)
        },
        'search_range_B': {
            'x': (0, 30),
            'y': (0, 30),
            'z': (0, 30),
            'rx': (0, 0.03),
            'ry': (0, 0.03)
        },
        
        # 选择的变量
        'selected_variables_A': ['x', 'y', 'z', 'rx', 'ry'],
        'selected_variables_B': ['x', 'y', 'z', 'rx', 'ry']
    }
    
    return config

# =============================================================================
# GUI接口函数
# =============================================================================

def create_gui_interface():
    """
    创建与GUI界面的接口函数
    这些函数应该在GUI线程中调用
    """
    
    def start_optimization(config: dict, hardware_adapter, 
                          progress_callback=None,
                          finished_callback=None,
                          convergence_callback=None,
                          lock_callback=None,
                          request_parameters_callback=None,  # 新增：参数请求回调
                          existing_optimizer=None):
        """
        开始优化 - 从GUI调用
        
        参数:
            config: 配置字典
            hardware_adapter: 硬件适配器实例
            progress_callback: 进度回调
            finished_callback: 完成回调
            convergence_callback: 收敛回调
            lock_callback: 锁定回调
            request_parameters_callback: 参数请求回调（新增）
            existing_optimizer: 现有的优化器实例
            
        返回:
            optimizer: 优化器实例
        """
        if existing_optimizer is not None:
            # 使用现有优化器
            optimizer = existing_optimizer
        else:
            # 创建新的优化器
            optimizer = DualEndGeneticAlgorithmOptimizer(config, hardware_adapter)
            optimizer.set_callbacks(
                progress_callback=progress_callback,
                finished_callback=finished_callback,
                convergence_callback=convergence_callback,
                lock_callback=lock_callback,
                request_parameters_callback=request_parameters_callback
            )
        
        # 在新线程中运行优化
        import threading
        optimization_thread = threading.Thread(target=optimizer.run)
        optimization_thread.daemon = True
        optimization_thread.start()
        
        return optimizer
    
    def stop_optimization(optimizer):
        """停止优化"""
        if optimizer and hasattr(optimizer, 'stop'):
            optimizer.stop()
    
    def activate_lock_mode(optimizer):
        """激活位置锁定模式"""
        if optimizer and hasattr(optimizer, 'activate_lock_mode'):
            optimizer.activate_lock_mode()
    
    def get_optimization_status(optimizer):
        """获取优化状态"""
        if not optimizer:
            return {'is_running': False}
        
        status = {
            'is_running': optimizer.is_running if hasattr(optimizer, 'is_running') else False,
            'best_power': optimizer.best_fitness if hasattr(optimizer, 'best_fitness') else 0,
            'current_generation': len(optimizer.history['generations']) if hasattr(optimizer, 'history') else 0,
            'total_evaluations': optimizer.history.get('evaluation_count', 0) if hasattr(optimizer, 'history') else 0,
            'optimization_phase': optimizer.optimization_phase.value if hasattr(optimizer, 'optimization_phase') else 'unknown',
            'light_detected': optimizer.light_detected if hasattr(optimizer, 'light_detected') else False,
            'converged': optimizer.final_convergence if hasattr(optimizer, 'final_convergence') else False,
            'high_power_keep_mode': optimizer.high_power_keep_mode if hasattr(optimizer, 'high_power_keep_mode') else False,
            'lock_mode_activated': optimizer.lock_mode_activated if hasattr(optimizer, 'lock_mode_activated') else False,
            'local_convergence_count': optimizer.local_convergence_count if hasattr(optimizer, 'local_convergence_count') else 0,
            'lock_position_available': optimizer.lock_population_A is not None and optimizer.lock_population_B is not None if hasattr(optimizer, 'lock_population_A') else False,
            'lock_fitness': optimizer.lock_fitness if hasattr(optimizer, 'lock_fitness') else 0,
            'population_size': optimizer.population_size if hasattr(optimizer, 'population_size') else 0,
            'gene_mutation_rate': optimizer.gene_mutation_rate if hasattr(optimizer, 'gene_mutation_rate') else 0,
            'gene_crossover_rate': optimizer.gene_crossover_rate if hasattr(optimizer, 'gene_crossover_rate') else 0,
            'chromosome_crossover_rate': optimizer.chromosome_crossover_rate if hasattr(optimizer, 'chromosome_crossover_rate') else 0,
            'high_power_search_range_percent': optimizer.high_power_search_range_percent if hasattr(optimizer, 'high_power_search_range_percent') else 0.05,
            'high_power_perturbation_strength': optimizer.high_power_perturbation_strength if hasattr(optimizer, 'high_power_perturbation_strength') else 0.01
        }
        
        # 添加高功率保持模式状态
        if optimizer.high_power_mode and hasattr(optimizer.high_power_mode, 'get_status'):
            status['high_power_status'] = optimizer.high_power_mode.get_status()
        
        return status
    
    def get_optimization_history(optimizer):
        """获取优化历史数据"""
        if not optimizer or not hasattr(optimizer, 'history'):
            return {}
        
        return optimizer.history
    # 在 create_gui_interface 函数中添加：

    def start_high_power_keep_mode(optimizer, center_position_dict: dict = None):
        """
        启动高功率保持模式 - 从GUI调用
        
        参数:
            optimizer: 优化器实例
            center_position_dict: 中心位置字典（如为None则使用当前最佳位置）
            
        返回:
            success: 是否成功启动
            message: 消息
        """
        if not optimizer or not hasattr(optimizer, 'start_high_power_keep_mode_from_gui'):
            return False, "优化器不存在或不支持高功率保持模式"
        
        try:
            # 从位置字典提取中心个体
            center_individual_A = None
            center_individual_B = None
            
            if center_position_dict:
                # 从位置字典构建个体
                center_individual_A = np.zeros(len(optimizer.selected_variables_A))
                center_individual_B = np.zeros(len(optimizer.selected_variables_B))
                
                # 提取A端变量
                idx_A = 0
                for var in optimizer.selected_variables_A:
                    key = f'A_{var}'
                    if key in center_position_dict:
                        center_individual_A[idx_A] = center_position_dict[key]
                    idx_A += 1
                
                # 提取B端变量
                idx_B = 0
                for var in optimizer.selected_variables_B:
                    key = f'B_{var}'
                    if key in center_position_dict:
                        center_individual_B[idx_B] = center_position_dict[key]
                    idx_B += 1
            else:
                # 使用当前最佳个体
                center_individual_A = optimizer.best_individual_A
                center_individual_B = optimizer.best_individual_B
            
            if center_individual_A is None or center_individual_B is None:
                return False, "无法获取中心位置"
            
            # 启动高功率保持模式
            success = optimizer.start_high_power_keep_mode_from_gui(
                center_individual_A, 
                center_individual_B,
                optimizer.best_fitness
            )
            
            if success:
                return True, "高功率保持模式已启动"
            else:
                return False, "启动高功率保持模式失败"
                
        except Exception as e:
            return False, f"启动高功率保持模式时发生错误: {str(e)}"

    
    def save_optimization_results(result, filename=None):
        """保存优化结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dual_end_optimization_results_{timestamp}.json"
        
        # 转换为可序列化的格式
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, OptimizationPhase):
                return obj.value
            return obj
        
        serializable_result = convert_to_serializable(result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        print(f"优化结果已保存到: {filename}")
        return filename
    
    # 在返回的字典中添加新函数
    return {
        'start_optimization': start_optimization,
        'stop_optimization': stop_optimization,
        'activate_lock_mode': activate_lock_mode,
        'get_optimization_status': get_optimization_status,
        'get_optimization_history': get_optimization_history,
        'save_optimization_results': save_optimization_results,
        'get_dual_end_config': get_dual_end_config,
        'start_high_power_keep_mode': start_high_power_keep_mode  # 新增
        }

# 确保GUI接口函数可用
gui_interface = create_gui_interface()

# 导出主要函数供GUI使用
start_dual_end_optimization = gui_interface['start_optimization']
stop_dual_end_optimization = gui_interface['stop_optimization']
activate_dual_end_lock_mode = gui_interface['activate_lock_mode']
get_dual_end_optimization_status = gui_interface['get_optimization_status']
get_dual_end_optimization_history = gui_interface['get_optimization_history']
save_dual_end_optimization_results = gui_interface['save_optimization_results']
get_default_dual_end_config = gui_interface['get_dual_end_config']

# 新增：导出参数更新相关功能
def update_dual_end_optimization_parameters(optimizer, new_params: dict):
    """
    更新优化器参数 - 从GUI调用
    
    参数:
        optimizer: 优化器实例
        new_params: 新参数的字典
        
    返回:
        success: 是否成功更新
        message: 更新结果消息
    """
    if not optimizer or not hasattr(optimizer, 'update_parameters_from_gui'):
        return False, "优化器不存在或不支持参数更新"
    
    try:
        optimizer.update_parameters_from_gui(new_params)
        return True, f"成功更新 {len(new_params)} 个参数"
    except Exception as e:
        return False, f"参数更新失败: {str(e)}"

# 将新函数添加到导出列表
# 在文件末尾的导出部分，添加新函数
__all__ = [
    'start_dual_end_optimization',
    'stop_dual_end_optimization',
    'activate_dual_end_lock_mode',
    'get_dual_end_optimization_status',
    'get_dual_end_optimization_history',
    'save_dual_end_optimization_results',
    'get_default_dual_end_config',
    'update_dual_end_optimization_parameters',
    'update_high_power_parameters',  # 新增
    'DualEndGeneticAlgorithmOptimizer',
    'OptimizationPhase'
]

# 导出 update_high_power_parameters 函数
update_high_power_parameters = update_dual_end_optimization_parameters