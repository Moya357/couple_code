import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
from tkinter import scrolledtext
import threading
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import weakref
import copy
import numpy as np
from device_manager_double import GlobalDeviceManager
from hardware_adapter_double import HardwareAdapter
from GAtest import GeneticAlgorithmOptimizer
from GAtest import visualize_ga_results, save_ga_data
from GA_double_new_1 import DualEndGeneticAlgorithmOptimizer, get_dual_end_config
from GA_double_new import  visualize_dual_end_results, save_dual_end_ga_data, create_dual_end_report
# 假设这些常量和类在其他地方定义
LARGE_FONT = ('SimHei', 12)
BOLD_FONT = ('SimHei', 12, 'bold')

def get_optimized_config():
    return {
        'population_size': 30,
        'generations': 200,
        'mutation_rate': 0.65,
        'crossover_rate': 0.7,
        'elite_size': 8,
        'tournament_size': 10,
        'convergence_threshold': 0.05,
        'convergence_patience': 8,
        'enhanced_exploration_max': 4,
        'enhanced_mutation_rate': 0.7,
        'converged_mutation_rate': 0.02,
        'local_search_rate': 0.4,
        'fitness_variance_threshold': 0.005,
        'alert_threshold_percent': 0.05,
        'monitoring_threshold_1': 0.01,
        'monitoring_threshold_2': 0.05,
        'monitoring_interval': 1.0,
        'adaptive_mutation_rate': True,
        'adaptive_crossover_rate': True,
        'elite_protection': True,
        'search_range': {
            'x': (0, 30),
            'y': (0, 30),
            'z': (0, 30),
            'rx': (0, 0.03),
            'ry': (0, 0.03)
        }
    }

class DualEndGAOptimizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("双端光纤耦合对准优化器")
        self.root.geometry("1400x1200")
        self.root.minsize(1200, 800)
        
        # 配置ttk样式
        self.style = ttk.Style()
        self.style.configure('Custom.TLabelframe', font=LARGE_FONT)
        self.style.configure('Custom.TLabelframe.Label', font=LARGE_FONT)
        self.style.configure('Custom.TCheckbutton', font=LARGE_FONT)
        self.style.configure('Custom.TButton', font=LARGE_FONT)
        
        # 优化器实例
        self.optimizer = None
        self.is_running = False
        self.optimization_thread = None
        self.result = None
        self.is_converged = False
        self.current_mode = "search"
        self.current_generation = 0
        
        # 修改：优化器自动进入位置锁定模式，高功率模式由GUI按钮触发
        self.high_power_mode_enabled = False  # 新增：高功率模式是否启用
        
        # 双端特定状态（保留优化器阶段记录）
        self.optimization_mode = tk.StringVar(value="single")  # "single" 或 "double"
        self.current_phase = "both_active"  # 优化器当前阶段，由优化器回调更新
        self.light_detected = False  # 保留通光状态
        
        # 功率监控相关
        self.power_monitoring = False
        self.monitoring_thread = None
        self.power_history = []
        self.time_history = []
        self.monitoring_start_time = None
        
        # 参数监控相关
        self.parameters_lock = threading.Lock()
        self.current_parameters = {}
        self.last_parameter_request = 0
        self.parameters_need_update = False
        self.parameter_monitor_timer = None  # 初始化为None
        self.auto_update_params = True
        self.monitor_interval = 2.0
        
        # 参数历史记录
        self.parameter_history = []
        self.parameter_presets = {}
        
        # 图表数据存储
        self.chart_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'power_times': [],
            'power_values': []
        }
        
        # 数据记录结构
        self._init_data_structure()
        self.elite_repository = []
        
        # 创建主框架
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建各个标签页
        self.param_frame = ttk.Frame(self.notebook)
        self.status_frame = ttk.Frame(self.notebook)
        self.results_frame = ttk.Frame(self.notebook)
        self.log_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.param_frame, text="参数设置")
        self.notebook.add(self.status_frame, text="优化状态")
        self.notebook.add(self.results_frame, text="结果可视化")
        self.notebook.add(self.log_frame, text="日志")
        
        # 初始化各个标签页
        self.init_param_frame()
        self.init_status_frame()
        self.init_results_frame()
        self.init_log_frame()
        
        # 绑定事件
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        
        # 硬件适配器
        self.device_manager = GlobalDeviceManager()
        self.hardware_adapter = None
        
        self.device_initialized = False
        self.current_piezo_mode = None

        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.log_power_details = False  # 设置为True可记录详细的功率信息
        self.show_engineering_units = True  # 在日志中显示工程单位
        self.debug_mode = False  # 调试模式
        
        # 启动功率监控
        self.start_power_monitoring()  
        
        self.log("GUI初始化完成，参数传递系统已就绪")
        
        
    def _init_data_structure(self):
        """初始化数据结构 - 移除阶段切换相关数据"""
        self.gui_data = {
            'metadata': {
                'created_time': datetime.now().isoformat(),
                'version': '2.0',
                'data_type': 'dual_end_genetic_algorithm_optimization'
            },
            'optimization_mode': 'single',
            'optimization_parameters': {},
            'optimization_results': {
                'summary': {},
                'best_individual_A': {},
                'best_individual_B': {},
                'convergence_info': {}
            },
            'evaluation_records': [],
            'generation_records': [],
            'power_monitoring_records': [],
            'optimization_history': {
                'best_fitness_history': [],
                'avg_fitness_history': [],
                'mutation_rate_history': [],
                'convergence_status_history': [],
            },
            'dual_end_specific': {
                'elite_repository_A': [],
                'elite_repository_B': [],
                'light_detected': False
            }
        }
        
        self.current_session = {
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0,
            'total_generations': 0,
            'session_id': None
        }

    def init_param_frame(self):
        """初始化参数设置标签页 - 添加应用参数按钮和参数监控选项，删除局部优化按钮"""
        frame = ttk.LabelFrame(self.param_frame, text="算法参数", style='Custom.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 模式选择
        mode_frame = ttk.LabelFrame(frame, text="优化模式", style='Custom.TLabelframe')
        mode_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Radiobutton(mode_frame, text="单端优化", variable=self.optimization_mode, 
                    value="single", command=self.on_optimization_mode_changed, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        tk.Radiobutton(mode_frame, text="双端优化", variable=self.optimization_mode, 
                    value="double", command=self.on_optimization_mode_changed, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        
        # 创建参数网格容器
        param_container = ttk.Frame(frame)
        param_container.pack(fill=tk.BOTH, expand=True)
        
        # 单端参数框架
        self.single_param_frame = ttk.Frame(param_container)
        self.single_param_frame.pack(fill=tk.BOTH, expand=True)
        
        # 双端参数框架
        self.double_param_frame = ttk.Frame(param_container)
        
        # 初始化单端参数
        self.init_single_param_frame()
        # 初始化双端参数
        self.init_double_param_frame()
        
        # 参数监控选项
        monitor_frame = ttk.LabelFrame(frame, text="参数监控", style='Custom.TLabelframe')
        monitor_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 参数自动更新选项
        self.auto_update_params = tk.BooleanVar(value=True)
        tk.Checkbutton(monitor_frame, text="自动更新参数到优化器", 
                    variable=self.auto_update_params, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        
        # 参数监控间隔
        ttk.Label(monitor_frame, text="监控间隔(秒):", font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        self.monitor_interval = tk.StringVar(value="2.0")
        monitor_entry = ttk.Entry(monitor_frame, textvariable=self.monitor_interval, width=10, font=LARGE_FONT)
        monitor_entry.pack(side=tk.LEFT, padx=5)
        
        # 按钮区域 - 删除局部优化按钮，添加高功率参数更新按钮
        btn_frame = ttk.Frame(self.param_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 设备控制按钮
        self.init_device_btn = ttk.Button(btn_frame, text="初始化设备", command=self.initialize_device, style='Custom.TButton')
        self.init_device_btn.pack(side=tk.LEFT, padx=5)
        
        # 参数应用按钮
        self.apply_params_btn = ttk.Button(btn_frame, text="应用参数到优化器", 
                                        command=self.apply_parameters_to_optimizer,
                                        style='Custom.TButton')
        self.apply_params_btn.pack(side=tk.LEFT, padx=5)
        
        # 高功率保持模式参数更新按钮（新增）
        self.apply_high_power_params_btn = ttk.Button(
            btn_frame, 
            text="更新高功率参数", 
            command=self.update_high_power_parameters,
            style='Custom.TButton'
        )
        self.apply_high_power_params_btn.pack(side=tk.LEFT, padx=5)
        
        self.set_initial_pos_btn = ttk.Button(btn_frame, text="设置初始位置", command=self.set_initial_position, style='Custom.TButton')
        self.set_initial_pos_btn.pack(side=tk.LEFT, padx=5)
        
        self.piezo_mode_btn = ttk.Button(btn_frame, text="切换为闭环模式", command=self.toggle_piezo_mode, style='Custom.TButton')
        self.piezo_mode_btn.pack(side=tk.LEFT, padx=5)
        
        # 功率监测数据保存按钮
        self.save_power_monitoring_btn = ttk.Button(
            btn_frame, 
            text="保存功率监测数据", 
            command=self.save_power_monitoring_data, 
            style='Custom.TButton'
        )
        self.save_power_monitoring_btn.pack(side=tk.LEFT, padx=5)
        
        # 优化控制按钮
        self.start_btn = ttk.Button(btn_frame, text="开始优化", command=self.start_optimization, style='Custom.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(btn_frame, text="停止优化", command=self.stop_optimization, state=tk.DISABLED, style='Custom.TButton')
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(btn_frame, text="重置参数", command=self.reset_parameters, style='Custom.TButton')
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame, text="保存参数", command=self.save_parameters, style='Custom.TButton')
        self.save_btn.pack(side=tk.RIGHT, padx=5)
        
        self.load_btn = ttk.Button(btn_frame, text="加载参数", command=self.load_parameters, style='Custom.TButton')
        self.load_btn.pack(side=tk.RIGHT, padx=5)

    def apply_parameters_to_optimizer(self):
        """
        将当前界面参数应用到优化器
        """
        if not self.is_running or not self.optimizer:
            messagebox.showinfo("提示", "优化器未运行，无法应用参数")
            return
        
        try:
            # 获取当前参数
            params = self.get_optimization_parameters()
            
            # 验证参数
            is_valid, message = self.validate_parameters(params)
            if not is_valid:
                messagebox.showerror("参数错误", message)
                return
            
            # 更新优化器参数
            success, msg = self.update_optimizer_parameters(params)
            
            if success:
                self.log(f"参数已成功应用到优化器: {msg}")
                messagebox.showinfo("成功", "参数已成功应用到优化器")
            else:
                self.log(f"参数应用失败: {msg}")
                messagebox.showerror("错误", f"参数应用失败: {msg}")
                
        except Exception as e:
            self.log(f"应用参数到优化器失败: {str(e)}")
            messagebox.showerror("错误", f"应用参数失败: {str(e)}")

    def init_single_param_frame(self):
        """初始化单端参数框架 - 删除监控参数相关部分"""
        frame = self.single_param_frame
        
        # 基本参数网格
        param_grid = ttk.Frame(frame)
        param_grid.pack(fill=tk.X, padx=10, pady=10)
        
        # 第一列参数
        params_col1 = [
            ("种群大小", "population_size", 20, 1, 100, int),
            ("最大代数", "generations", 100, 10, 1000, int),
            ("初始变异率", "mutation_rate", 0.15, 0.01, 1.0, float),
            ("交叉率", "crossover_rate", 0.8, 0.1, 1.0, float),
            ("精英数量", "elite_size", 4, 1, 50, int),
            ("锦标赛大小", "tournament_size", 3, 2, 20, int)
        ]
        
        # 第二列参数
        params_col2 = [
            ("收敛阈值 (%)", "convergence_threshold", 5, 0.1, 20, float),
            ("收敛耐心值", "convergence_patience", 8, 1, 50, int),
            ("增强搜索次数", "enhanced_exploration_max", 4, 1, 20, int),
            ("增强变异率", "enhanced_mutation_rate", 0.7, 0.1, 1.0, float),
            ("收敛后变异率", "converged_mutation_rate", 0.02, 0.01, 0.5, float),
            ("局部搜索率", "local_search_rate", 0.4, 0.1, 1.0, float),
            ("适应度方差阈值", "fitness_variance_threshold", 0.005, 0.001, 0.1, float)
        ]
        
        self.single_param_entries = {}
        
        # 创建参数输入框
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(params_col1):
            ttk.Label(param_grid, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(param_grid, width=15, font=LARGE_FONT)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(param_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=i, column=2, padx=5, pady=5)
            
            self.single_param_entries[key] = (entry, min_val, max_val, dtype)
        
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(params_col2):
            ttk.Label(param_grid, text=label, font=LARGE_FONT).grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(param_grid, width=15, font=LARGE_FONT)
            entry.grid(row=i, column=4, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(param_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=i, column=5, padx=5, pady=5)
            
            self.single_param_entries[key] = (entry, min_val, max_val, dtype)
        
        # 删除监控参数部分，直接转到自适应参数选项
        
        # 自适应参数选项
        adaptive_frame = ttk.LabelFrame(frame, text="自适应参数", style='Custom.TLabelframe')
        adaptive_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.adaptive_mutation = tk.BooleanVar(value=True)
        self.adaptive_crossover = tk.BooleanVar(value=True)
        self.elite_protection = tk.BooleanVar(value=True)
        
        # 使用 tk.Checkbutton 替代 ttk.Checkbutton
        tk.Checkbutton(adaptive_frame, text="自适应变异率", variable=self.adaptive_mutation, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        tk.Checkbutton(adaptive_frame, text="自适应交叉率", variable=self.adaptive_crossover, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        tk.Checkbutton(adaptive_frame, text="精英保护", variable=self.elite_protection, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        
        # 初始位置和搜索范围设置 - 合并到同一行
        position_range_frame = ttk.LabelFrame(frame, text="初始位置和搜索范围", style='Custom.TLabelframe')
        position_range_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # 创建表格布局
        pos_range_grid = ttk.Frame(position_range_frame)
        pos_range_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 表头
        ttk.Label(pos_range_grid, text="参数", font=BOLD_FONT).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Label(pos_range_grid, text="初始位置", font=BOLD_FONT).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(pos_range_grid, text="搜索范围", font=BOLD_FONT).grid(row=0, column=2, columnspan=2, padx=5, pady=5)
        
        position_ranges = [
            ("X (μm)", "x", 15.0, 0, 30),
            ("Y (μm)", "y", 15.0, 0, 30),
            ("Z (μm)", "z", 15.0, 0, 30),
            ("RX (rad)", "rx", 0.015, 0.0, 0.03),
            ("RY (rad)", "ry", 0.015, 0.0, 0.03)
        ]
        
        for i, (label, key, default, min_val, max_val) in enumerate(position_ranges):
            ttk.Label(pos_range_grid, text=label, font=LARGE_FONT).grid(row=i+1, column=0, sticky=tk.W, padx=15, pady=5)
            
            # 初始位置
            initial_entry = ttk.Entry(pos_range_grid, width=12, font=LARGE_FONT)
            initial_entry.grid(row=i+1, column=1, padx=5, pady=5)
            initial_entry.insert(0, str(default))
            self.single_param_entries[f"{key}_initial"] = (initial_entry, None, None, float)
            
            # 搜索范围
            min_entry = ttk.Entry(pos_range_grid, width=10, font=LARGE_FONT)
            min_entry.grid(row=i+1, column=2, padx=2, pady=5)
            min_entry.insert(0, str(min_val))
            
            ttk.Label(pos_range_grid, text="至", font=LARGE_FONT).grid(row=i+1, column=3)
            
            max_entry = ttk.Entry(pos_range_grid, width=10, font=LARGE_FONT)
            max_entry.grid(row=i+1, column=4, padx=2, pady=5)
            max_entry.insert(0, str(max_val))
            
            self.single_param_entries[f"{key}_min"] = (min_entry, None, None, float)
            self.single_param_entries[f"{key}_max"] = (max_entry, None, None, float)
        
        # 优化参数选择
        optimization_params_frame = ttk.LabelFrame(frame, text="优化参数选择", style='Custom.TLabelframe')
        optimization_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        optimization_params_grid = ttk.Frame(optimization_params_frame)
        optimization_params_grid.pack(fill=tk.X, padx=10, pady=5)
        
        self.optimize_x = tk.BooleanVar(value=True)
        self.optimize_y = tk.BooleanVar(value=True)
        self.optimize_z = tk.BooleanVar(value=True)
        self.optimize_rx = tk.BooleanVar(value=True)
        self.optimize_ry = tk.BooleanVar(value=True)
        
        # 创建复选框
        tk.Checkbutton(optimization_params_grid, text="优化X轴", variable=self.optimize_x, font=LARGE_FONT).grid(row=0, column=0, sticky=tk.W, padx=15, pady=5)
        tk.Checkbutton(optimization_params_grid, text="优化Y轴", variable=self.optimize_y, font=LARGE_FONT).grid(row=0, column=1, sticky=tk.W, padx=15, pady=5)
        tk.Checkbutton(optimization_params_grid, text="优化Z轴", variable=self.optimize_z, font=LARGE_FONT).grid(row=0, column=2, sticky=tk.W, padx=15, pady=5)
        tk.Checkbutton(optimization_params_grid, text="优化RX轴", variable=self.optimize_rx, font=LARGE_FONT).grid(row=1, column=0, sticky=tk.W, padx=15, pady=5)
        tk.Checkbutton(optimization_params_grid, text="优化RY轴", variable=self.optimize_ry, font=LARGE_FONT).grid(row=1, column=1, sticky=tk.W, padx=15, pady=5)
        
        # 添加快速选择按钮
        quick_select_frame = ttk.Frame(optimization_params_frame)
        quick_select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(quick_select_frame, text="选择全部", command=self.select_all_params, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_select_frame, text="仅位置(XYZ)", command=self.select_position_params, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_select_frame, text="仅角度(RXRY)", command=self.select_angle_params, style='Custom.TButton').pack(side=tk.LEFT, padx=5)
        ttk.Button(quick_select_frame, text="清空选择", command=self.clear_all_params, style='Custom.TButton').pack(side=tk.LEFT, padx=5)

    def init_double_param_frame(self):
        """初始化双端参数框架 - 添加完整的高功率保持模式参数"""
        frame = self.double_param_frame
        
        # 遗传算法参数
        ga_params_frame = ttk.LabelFrame(frame, text="遗传算法参数", style='Custom.TLabelframe')
        ga_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ga_params_grid = ttk.Frame(ga_params_frame)
        ga_params_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 第一列参数
        ga_params_col1 = [
            ("种群大小", "population_size", 30, 1, 100, int),
            ("最大代数", "generations", 200, 10, 1000, int),
            ("基因变异率", "gene_mutation_rate", 0.15, 0.01, 1.0, float),
            ("基因交叉率", "gene_crossover_rate", 0.8, 0.1, 1.0, float),
            ("染色体交叉率", "chromosome_crossover_rate", 0.2, 0.01, 1.0, float),
            ("精英数量", "elite_size", 8, 1, 50, int),
            ("锦标赛大小", "tournament_size", 10, 2, 20, int),
            ("收敛阈值 (%)", "convergence_threshold", 5, 0.1, 20, float)
        ]
        
        # 第二列参数
        ga_params_col2 = [
            ("收敛耐心值", "convergence_patience", 8, 1, 50, int),
            ("增强搜索次数", "enhanced_exploration_max", 3, 1, 20, int),
            ("增强变异率", "enhanced_mutation_rate", 0.7, 0.1, 1.0, float),
            ("收敛后变异率", "converged_mutation_rate", 0.02, 0.001, 0.5, float),
            ("局部搜索率", "local_search_rate", 0.4, 0.1, 1.0, float),
            ("适应度方差阈值", "fitness_variance_threshold", 0.005, 0.0001, 0.1, float),
        ]
        
        self.double_param_entries = {}
        
        # 创建遗传算法参数输入框
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(ga_params_col1):
            ttk.Label(ga_params_grid, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(ga_params_grid, width=15, font=LARGE_FONT)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(ga_params_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=i, column=2, padx=5, pady=5)
            
            self.double_param_entries[key] = (entry, min_val, max_val, dtype)
        
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(ga_params_col2):
            ttk.Label(ga_params_grid, text=label, font=LARGE_FONT).grid(row=i, column=3, sticky=tk.W, padx=5, pady=5)
            entry = ttk.Entry(ga_params_grid, width=15, font=LARGE_FONT)
            entry.grid(row=i, column=4, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(ga_params_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=i, column=5, padx=5, pady=5)
            
            self.double_param_entries[key] = (entry, min_val, max_val, dtype)
        
        # ==================== 完整的高功率保持模式参数区域 ====================
        high_power_frame = ttk.LabelFrame(frame, text="高功率保持模式参数", style='Custom.TLabelframe')
        high_power_frame.pack(fill=tk.X, padx=10, pady=10)
        
        high_power_grid = ttk.Frame(high_power_frame)
        high_power_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 第一行参数
        high_power_params_row1 = [
            ("高功率种群大小", "high_power_population_size", 20, 5, 50, int),
            ("高功率变异率", "high_power_mutation_rate", 0.05, 0.001, 0.5, float),
            ("高功率交叉率", "high_power_crossover_rate", 0.3, 0.05, 1.0, float)
        ]
        
        # 第二行参数
        high_power_params_row2 = [
            ("高功率搜索范围 (%)", "high_power_search_range_percent", 5.0, 0.1, 20, float),
            ("克隆扰动强度", "high_power_perturbation_strength", 0.01, 0.001, 0.1, float),
            ("精英克隆率", "elite_clone_rate", 0.25, 0.05, 0.5, float)
        ]
        
        # 第一行参数
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(high_power_params_row1):
            ttk.Label(high_power_grid, text=label, font=LARGE_FONT).grid(row=0, column=i*2, sticky=tk.W, padx=15, pady=5)
            entry = ttk.Entry(high_power_grid, width=15, font=LARGE_FONT)
            entry.grid(row=0, column=i*2+1, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(high_power_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=0, column=i*2+2, padx=5, pady=5)
            
            self.double_param_entries[key] = (entry, min_val, max_val, dtype)
        
        # 第二行参数
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(high_power_params_row2):
            ttk.Label(high_power_grid, text=label, font=LARGE_FONT).grid(row=1, column=i*2, sticky=tk.W, padx=15, pady=5)
            entry = ttk.Entry(high_power_grid, width=15, font=LARGE_FONT)
            entry.grid(row=1, column=i*2+1, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(high_power_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=1, column=i*2+2, padx=5, pady=5)
            
            self.double_param_entries[key] = (entry, min_val, max_val, dtype)
        
        # 双端特定参数
        dual_params_frame = ttk.LabelFrame(frame, text="双端优化参数", style='Custom.TLabelframe')
        dual_params_frame.pack(fill=tk.X, padx=10, pady=10)
        
        dual_params_grid = ttk.Frame(dual_params_frame)
        dual_params_grid.pack(fill=tk.X, padx=10, pady=5)
        
        dual_params = [
            ("通光阈值 (mW)", "light_threshold", 0.0002, 0.00000001, 1.0, float),
            ("锁定阈值 (%)", "lock_mode_threshold", 0.1, 0.01, 10, float),
        ]
        
        for i, (label, key, default, min_val, max_val, dtype) in enumerate(dual_params):
            ttk.Label(dual_params_grid, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=15, pady=5)
            entry = ttk.Entry(dual_params_grid, width=15, font=LARGE_FONT)
            entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            entry.insert(0, str(default))
            
            # 添加参数范围提示
            range_label = ttk.Label(dual_params_grid, text=f"[{min_val}-{max_val}]", font=('SimHei', 8))
            range_label.grid(row=i, column=2, padx=5, pady=5)
            
            self.double_param_entries[key] = (entry, min_val, max_val, dtype)
        
        # 自适应参数选项
        adaptive_frame = ttk.LabelFrame(frame, text="自适应参数", style='Custom.TLabelframe')
        adaptive_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.adaptive_mutation = tk.BooleanVar(value=True)
        self.adaptive_crossover = tk.BooleanVar(value=True)
        self.elite_protection = tk.BooleanVar(value=True)
        
        # 使用 tk.Checkbutton 替代 ttk.Checkbutton
        tk.Checkbutton(adaptive_frame, text="自适应变异率", variable=self.adaptive_mutation, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        tk.Checkbutton(adaptive_frame, text="自适应交叉率", variable=self.adaptive_crossover, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        tk.Checkbutton(adaptive_frame, text="精英保护", variable=self.elite_protection, font=LARGE_FONT).pack(side=tk.LEFT, padx=15)
        
        # ==================== 坐标参数设置区域 ====================
        # 创建一个大的框架来包含A端和B端参数设置
        coordinate_frame = ttk.LabelFrame(frame, text="坐标参数设置", style='Custom.TLabelframe')
        coordinate_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建左右两个子框架
        left_frame = ttk.Frame(coordinate_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 5), pady=10)
        
        right_frame = ttk.Frame(coordinate_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 10), pady=10)
        
        # A端参数设置（放在左侧）
        a_end_frame = ttk.LabelFrame(left_frame, text="A端参数设置", style='Custom.TLabelframe')
        a_end_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # A端参数选择
        a_params_frame = ttk.Frame(a_end_frame)
        a_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.optimize_A_x = tk.BooleanVar(value=True)
        self.optimize_A_y = tk.BooleanVar(value=True)
        self.optimize_A_z = tk.BooleanVar(value=True)
        self.optimize_A_rx = tk.BooleanVar(value=True)
        self.optimize_A_ry = tk.BooleanVar(value=True)
        
        # 使用grid布局，5个参数分两行显示
        tk.Checkbutton(a_params_frame, text="优化X轴", variable=self.optimize_A_x, font=LARGE_FONT).grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(a_params_frame, text="优化Y轴", variable=self.optimize_A_y, font=LARGE_FONT).grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(a_params_frame, text="优化Z轴", variable=self.optimize_A_z, font=LARGE_FONT).grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(a_params_frame, text="优化RX轴", variable=self.optimize_A_rx, font=LARGE_FONT).grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(a_params_frame, text="优化RY轴", variable=self.optimize_A_ry, font=LARGE_FONT).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        # A端初始位置和搜索范围
        a_pos_range_frame = ttk.Frame(a_end_frame)
        a_pos_range_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建表格布局
        a_pos_range_grid = ttk.Frame(a_pos_range_frame)
        a_pos_range_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 表头
        ttk.Label(a_pos_range_grid, text="参数", font=BOLD_FONT).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(a_pos_range_grid, text="初始位置", font=BOLD_FONT).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(a_pos_range_grid, text="搜索范围", font=BOLD_FONT).grid(row=0, column=2, columnspan=2, padx=5, pady=2)
        
        a_position_ranges = [
            ("X (μm)", "A_x", 15.0, 0, 30),
            ("Y (μm)", "A_y", 15.0, 0, 30),
            ("Z (μm)", "A_z", 15.0, 0, 30),
            ("RX (rad)", "A_rx", 0.015, 0.0, 0.03),
            ("RY (rad)", "A_ry", 0.015, 0.0, 0.03)
        ]
        
        for i, (label, key, default, min_val, max_val) in enumerate(a_position_ranges):
            ttk.Label(a_pos_range_grid, text=label, font=LARGE_FONT).grid(row=i+1, column=0, sticky=tk.W, padx=10, pady=2)
            
            # 初始位置
            initial_entry = ttk.Entry(a_pos_range_grid, width=10, font=LARGE_FONT)
            initial_entry.grid(row=i+1, column=1, padx=2, pady=2)
            initial_entry.insert(0, str(default))
            self.double_param_entries[f"{key}_initial"] = (initial_entry, None, None, float)
            
            # 搜索范围
            min_entry = ttk.Entry(a_pos_range_grid, width=8, font=LARGE_FONT)
            min_entry.grid(row=i+1, column=2, padx=2, pady=2)
            min_entry.insert(0, str(min_val))
            
            ttk.Label(a_pos_range_grid, text="至", font=LARGE_FONT).grid(row=i+1, column=3)
            
            max_entry = ttk.Entry(a_pos_range_grid, width=8, font=LARGE_FONT)
            max_entry.grid(row=i+1, column=4, padx=2, pady=2)
            max_entry.insert(0, str(max_val))
            
            self.double_param_entries[f"{key}_min"] = (min_entry, None, None, float)
            self.double_param_entries[f"{key}_max"] = (max_entry, None, None, float)
        
        # B端参数设置（放在右侧）
        b_end_frame = ttk.LabelFrame(right_frame, text="B端参数设置", style='Custom.TLabelframe')
        b_end_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # B端参数选择
        b_params_frame = ttk.Frame(b_end_frame)
        b_params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.optimize_B_x = tk.BooleanVar(value=True)
        self.optimize_B_y = tk.BooleanVar(value=True)
        self.optimize_B_z = tk.BooleanVar(value=True)
        self.optimize_B_rx = tk.BooleanVar(value=True)
        self.optimize_B_ry = tk.BooleanVar(value=True)
        
        # 使用grid布局，5个参数分两行显示
        tk.Checkbutton(b_params_frame, text="优化X轴", variable=self.optimize_B_x, font=LARGE_FONT).grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(b_params_frame, text="优化Y轴", variable=self.optimize_B_y, font=LARGE_FONT).grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(b_params_frame, text="优化Z轴", variable=self.optimize_B_z, font=LARGE_FONT).grid(row=0, column=2, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(b_params_frame, text="优化RX轴", variable=self.optimize_B_rx, font=LARGE_FONT).grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        tk.Checkbutton(b_params_frame, text="优化RY轴", variable=self.optimize_B_ry, font=LARGE_FONT).grid(row=1, column=1, sticky=tk.W, padx=10, pady=2)
        
        # B端初始位置和搜索范围
        b_pos_range_frame = ttk.Frame(b_end_frame)
        b_pos_range_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 创建表格布局
        b_pos_range_grid = ttk.Frame(b_pos_range_frame)
        b_pos_range_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 表头
        ttk.Label(b_pos_range_grid, text="参数", font=BOLD_FONT).grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
        ttk.Label(b_pos_range_grid, text="初始位置", font=BOLD_FONT).grid(row=0, column=1, padx=5, pady=2)
        ttk.Label(b_pos_range_grid, text="搜索范围", font=BOLD_FONT).grid(row=0, column=2, columnspan=2, padx=5, pady=2)
        
        b_position_ranges = [
            ("X (μm)", "B_x", 15.0, 0, 30),
            ("Y (μm)", "B_y", 15.0, 0, 30),
            ("Z (μm)", "B_z", 15.0, 0, 30),
            ("RX (rad)", "B_rx", 0.015, 0.0, 0.03),
            ("RY (rad)", "B_ry", 0.015, 0.0, 0.03)
        ]
        
        for i, (label, key, default, min_val, max_val) in enumerate(b_position_ranges):
            ttk.Label(b_pos_range_grid, text=label, font=LARGE_FONT).grid(row=i+1, column=0, sticky=tk.W, padx=10, pady=2)
            
            # 初始位置
            initial_entry = ttk.Entry(b_pos_range_grid, width=10, font=LARGE_FONT)
            initial_entry.grid(row=i+1, column=1, padx=2, pady=2)
            initial_entry.insert(0, str(default))
            self.double_param_entries[f"{key}_initial"] = (initial_entry, None, None, float)
            
            # 搜索范围
            min_entry = ttk.Entry(b_pos_range_grid, width=8, font=LARGE_FONT)
            min_entry.grid(row=i+1, column=2, padx=2, pady=2)
            min_entry.insert(0, str(min_val))
            
            ttk.Label(b_pos_range_grid, text="至", font=LARGE_FONT).grid(row=i+1, column=3)
            
            max_entry = ttk.Entry(b_pos_range_grid, width=8, font=LARGE_FONT)
            max_entry.grid(row=i+1, column=4, padx=2, pady=2)
            max_entry.insert(0, str(max_val))
            
            self.double_param_entries[f"{key}_min"] = (min_entry, None, None, float)
            self.double_param_entries[f"{key}_max"] = (max_entry, None, None, float)
    def on_optimization_mode_changed(self):
        """优化模式改变时的回调"""
        mode = self.optimization_mode.get()
        
        if mode == "single":
            self.single_param_frame.pack(fill=tk.BOTH, expand=True)
            self.double_param_frame.pack_forget()
            self.log("切换到单端优化模式")
        else:
            self.single_param_frame.pack_forget()
            self.double_param_frame.pack(fill=tk.BOTH, expand=True)
            self.log("切换到双端优化模式")

    # 以下方法保持与原始GUI相同，确保所有功能完整
    def get_selected_optimization_params(self):
        """获取选择的优化参数列表"""
        selected_params = []
        if self.optimize_x.get():
            selected_params.append('x')
        if self.optimize_y.get():
            selected_params.append('y')
        if self.optimize_z.get():
            selected_params.append('z')
        if self.optimize_rx.get():
            selected_params.append('rx')
        if self.optimize_ry.get():
            selected_params.append('ry')
        
        if not selected_params:
            selected_params = ['x', 'y', 'z', 'rx', 'ry']
            self.log("警告：未选择任何优化参数，默认选择所有参数")
        
        return selected_params

    def get_selected_optimization_params_A(self):
        """获取A端选择的优化参数列表"""
        selected_params = []
        if self.optimize_A_x.get():
            selected_params.append('x')
        if self.optimize_A_y.get():
            selected_params.append('y')
        if self.optimize_A_z.get():
            selected_params.append('z')
        if self.optimize_A_rx.get():
            selected_params.append('rx')
        if self.optimize_A_ry.get():
            selected_params.append('ry')
        
        if not selected_params:
            selected_params = ['x', 'y', 'z', 'rx', 'ry']
            self.log("警告：A端未选择任何优化参数，默认选择所有参数")
        
        return selected_params

    def get_selected_optimization_params_B(self):
        """获取B端选择的优化参数列表"""
        selected_params = []
        if self.optimize_B_x.get():
            selected_params.append('x')
        if self.optimize_B_y.get():
            selected_params.append('y')
        if self.optimize_B_z.get():
            selected_params.append('z')
        if self.optimize_B_rx.get():
            selected_params.append('rx')
        if self.optimize_B_ry.get():
            selected_params.append('ry')
        
        if not selected_params:
            selected_params = ['x', 'y', 'z', 'rx', 'ry']
            self.log("警告：B端未选择任何优化参数，默认选择所有参数")
        
        return selected_params

    def select_all_params(self):
        """选择所有优化参数"""
        self.optimize_x.set(True)
        self.optimize_y.set(True)
        self.optimize_z.set(True)
        self.optimize_rx.set(True)
        self.optimize_ry.set(True)
        self.log("已选择所有优化参数")

    def select_position_params(self):
        """仅选择位置参数"""
        self.optimize_x.set(True)
        self.optimize_y.set(True)
        self.optimize_z.set(True)
        self.optimize_rx.set(False)
        self.optimize_ry.set(False)
        self.log("已选择位置参数(XYZ)")

    def select_angle_params(self):
        """仅选择角度参数"""
        self.optimize_x.set(False)
        self.optimize_y.set(False)
        self.optimize_z.set(False)
        self.optimize_rx.set(True)
        self.optimize_ry.set(True)
        self.log("已选择角度参数(RXRY)")

    def clear_all_params(self):
        """清空所有参数选择"""
        self.optimize_x.set(False)
        self.optimize_y.set(False)
        self.optimize_z.set(False)
        self.optimize_rx.set(False)
        self.optimize_ry.set(False)
        self.log("已清空所有参数选择")

    def get_parameters(self):
        """获取当前参数设置"""
        mode = self.optimization_mode.get()
        
        if mode == "single":
            return self._get_single_parameters()
        else:
            return self._get_double_parameters()

    def _get_single_parameters(self):
        """获取单端参数 - 删除监控参数"""
        params = {}
        
        # 获取单端基本参数
        for key in self.single_param_entries:
            entry, min_val, max_val, dtype = self.single_param_entries[key]
            try:
                value = dtype(entry.get())
                if min_val is not None and max_val is not None and not (min_val <= value <= max_val):
                    raise ValueError(f"{key} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                params[key] = value
            except ValueError as e:
                raise ValueError(f"参数 {key} 格式错误: {str(e)}")
        
        # 获取选择的优化参数
        selected_params = self.get_selected_optimization_params()
        
        formatted_params = {
            'population_size': params['population_size'],
            'generations': params['generations'],
            'mutation_rate': params['mutation_rate'],
            'crossover_rate': params['crossover_rate'],
            'elite_size': params['elite_size'],
            'tournament_size': params['tournament_size'],
            'convergence_threshold': params['convergence_threshold'] / 100,
            'convergence_patience': params['convergence_patience'],
            'enhanced_exploration_max': params['enhanced_exploration_max'],
            'enhanced_mutation_rate': params['enhanced_mutation_rate'],
            'converged_mutation_rate': params['converged_mutation_rate'],
            'local_search_rate': params['local_search_rate'],
            'fitness_variance_threshold': params['fitness_variance_threshold'],
            
            # 删除监控参数
            # 'monitoring_threshold_1': params['monitoring_threshold_1'] / 100,
            # 'monitoring_threshold_2': params['monitoring_threshold_2'] / 100,
            # 'monitoring_interval': params['monitoring_interval'],
            # 'alert_threshold_percent': params['alert_threshold_percent'] / 100,
            
            # 自适应参数
            'adaptive_mutation_rate': self.adaptive_mutation.get(),
            'adaptive_crossover_rate': self.adaptive_crossover.get(),
            'elite_protection': self.elite_protection.get(),
            
            # 选择的优化参数
            'selected_variables': selected_params,
            
            # 搜索范围
            'search_range': {
                'x': (params['x_min'], params['x_max']),
                'y': (params['y_min'], params['y_max']),
                'z': (params['z_min'], params['z_max']),
                'rx': (params['rx_min'], params['rx_max']),
                'ry': (params['ry_min'], params['ry_max'])
            }
        }
        
        # 记录选择的参数
        self.log(f"单端优化参数: {', '.join(selected_params)}")
        
        return formatted_params
    
    def _get_double_parameters(self):
        """获取双端参数 - 完整获取所有参数，删除警报阈值，添加高功率保持模式参数"""
        # 获取双端基本参数
        params = {}
        
        # 1. 从界面获取所有双端参数
        for key in self.double_param_entries:
            entry, min_val, max_val, dtype = self.double_param_entries[key]
            try:
                value = dtype(entry.get())
                if min_val is not None and max_val is not None and not (min_val <= value <= max_val):
                    raise ValueError(f"{key} 的值 {value} 超出范围 [{min_val}, {max_val}]")
                params[key] = value
            except ValueError as e:
                # 使用默认值
                default_values = {
                    'population_size': 30,
                    'generations': 200,
                    'gene_mutation_rate': 0.15,
                    'gene_crossover_rate': 0.8,
                    'chromosome_crossover_rate': 0.2,
                    'elite_size': 8,
                    'tournament_size': 10,
                    'convergence_threshold': 5.0,
                    'convergence_patience': 8,
                    'enhanced_exploration_max': 3,
                    'enhanced_mutation_rate': 0.7,
                    'converged_mutation_rate': 0.02,
                    'local_search_rate': 0.4,
                    'fitness_variance_threshold': 0.005,
                    # 删除警报阈值
                    # 'alert_threshold_percent': 5.0,
                    'high_power_population_size': 20,
                    'high_power_mutation_rate': 0.05,
                    'high_power_crossover_rate': 0.3,
                    'high_power_search_range_percent': 5.0,
                    'high_power_perturbation_strength': 0.01,
                    'light_threshold': 0.0002,
                    'lock_mode_threshold': 0.1,
                    'elite_clone_rate': 0.25
                }
                params[key] = default_values.get(key, 0)
        
        # 2. 获取A端和B端选择的参数
        selected_params_A = self.get_selected_optimization_params_A()
        selected_params_B = self.get_selected_optimization_params_B()
        
        # 3. 使用双端默认配置作为基础
        config = get_dual_end_config()
        
        # 4. 转换百分比参数
        def convert_percent(value):
            """将可能的百分比值转换为小数"""
            if isinstance(value, (int, float)):
                if value > 1:  # 可能是百分比输入
                    return value / 100.0
            return float(value)
        
        # 5. 更新配置
        config.update({
            # 遗传算法参数
            'population_size': params.get('population_size', 30),
            'generations': params.get('generations', 200),
            'gene_mutation_rate': params.get('gene_mutation_rate', 0.15),
            'gene_crossover_rate': params.get('gene_crossover_rate', 0.8),
            'chromosome_crossover_rate': params.get('chromosome_crossover_rate', 0.2),
            'elite_size': params.get('elite_size', 8),
            'tournament_size': params.get('tournament_size', 10),
            
            # 收敛检测参数
            'convergence_threshold': convert_percent(params.get('convergence_threshold', 5.0)),
            'convergence_patience': params.get('convergence_patience', 8),
            'enhanced_exploration_max': params.get('enhanced_exploration_max', 3),
            'enhanced_mutation_rate': params.get('enhanced_mutation_rate', 0.7),
            'converged_mutation_rate': params.get('converged_mutation_rate', 0.02),
            'local_search_rate': params.get('local_search_rate', 0.4),
            'fitness_variance_threshold': params.get('fitness_variance_threshold', 0.005),
            # 删除警报阈值
            # 'alert_threshold_percent': convert_percent(params.get('alert_threshold_percent', 5.0)),
            
            # 高功率保持模式参数
            'high_power_population_size': params.get('high_power_population_size', 20),
            'high_power_mutation_rate': params.get('high_power_mutation_rate', 0.05),
            'high_power_crossover_rate': params.get('high_power_crossover_rate', 0.3),
            'high_power_search_range_percent': convert_percent(params.get('high_power_search_range_percent', 5.0)),
            'high_power_perturbation_strength': params.get('high_power_perturbation_strength', 0.01),
            
            # 双端特定参数
            'light_threshold': params.get('light_threshold', 0.0002),
            'lock_mode_threshold': convert_percent(params.get('lock_mode_threshold', 0.1)),
            'elite_clone_rate': params.get('elite_clone_rate', 0.25),
            'local_search_range_percent': convert_percent(params.get('local_search_rate', 0.4)),
            
            # 自适应参数
            'adaptive_mutation_rate': self.adaptive_mutation.get(),
            'adaptive_crossover_rate': self.adaptive_crossover.get(),
            'elite_protection': self.elite_protection.get(),
            
            # 更新选择的变量
            'selected_variables_A': selected_params_A,
            'selected_variables_B': selected_params_B,
            
            # 搜索范围
            'search_range_A': {
                'x': (params.get('A_x_min', 0), params.get('A_x_max', 30)),
                'y': (params.get('A_y_min', 0), params.get('A_y_max', 30)),
                'z': (params.get('A_z_min', 0), params.get('A_z_max', 30)),
                'rx': (params.get('A_rx_min', 0.0), params.get('A_rx_max', 0.03)),
                'ry': (params.get('A_ry_min', 0.0), params.get('A_ry_max', 0.03))
            },
            'search_range_B': {
                'x': (params.get('B_x_min', 0), params.get('B_x_max', 30)),
                'y': (params.get('B_y_min', 0), params.get('B_y_max', 30)),
                'z': (params.get('B_z_min', 0), params.get('B_z_max', 30)),
                'rx': (params.get('B_rx_min', 0.0), params.get('B_rx_max', 0.03)),
                'ry': (params.get('B_ry_min', 0.0), params.get('B_ry_max', 0.03))
            }
        })
        
        # 6. 记录参数信息
        self.log(f"双端优化参数 - A端: {', '.join(selected_params_A)}, B端: {', '.join(selected_params_B)}")
        self.log(f"遗传算法参数:")
        self.log(f"  种群大小: {config['population_size']}")
        self.log(f"  基因变异率: {config['gene_mutation_rate']:.3f}")
        self.log(f"  基因交叉率: {config['gene_crossover_rate']:.3f}")
        self.log(f"  染色体交叉率: {config['chromosome_crossover_rate']:.3f}")
        self.log(f"高功率保持模式参数:")
        self.log(f"  种群大小: {config['high_power_population_size']}")
        self.log(f"  变异率: {config['high_power_mutation_rate']:.3f}")
        self.log(f"  交叉率: {config['high_power_crossover_rate']:.3f}")
        self.log(f"  搜索范围: ±{config['high_power_search_range_percent']*100:.2f}%")
        self.log(f"  克隆扰动强度: {config['high_power_perturbation_strength']:.3f}")
        self.log(f"收敛检测参数:")
        self.log(f"  收敛阈值: {config['convergence_threshold']*100:.2f}%")
        self.log(f"  收敛耐心值: {config['convergence_patience']}")
        self.log(f"位置锁定参数:")
        self.log(f"  锁定阈值: {config['lock_mode_threshold']*100:.2f}%")
        
        return config

    def init_status_frame(self):
        """初始化优化状态标签页 - 添加高功率保持模式状态显示，删除局部优化相关状态"""
        status_grid = ttk.Frame(self.status_frame)
        status_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 状态信息区域
        info_frame = ttk.LabelFrame(status_grid, text="优化信息", style='Custom.TLabelframe')
        info_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        # 状态信息 - 删除局部优化相关状态，添加高功率保持模式参数
        status_info = [
            ("优化模式:", "optimization_mode", "单端"),
            ("设备状态:", "device_status", "未初始化"),
            ("压电模式:", "piezo_mode", "未设置"),
            ("当前模式:", "operation_mode", "搜索模式"),
            ("通光状态:", "light_status", "未通光"),
            ("当前状态:", "status", "未运行"),
            ("当前代数:", "generation", "0/0"),
            ("最佳功率:", "best_power", "0.0000 mW"),
            ("当前功率:", "current_power", "0.0000 mW"),
            ("平均功率:", "avg_power", "0.0000 mW"),
            ("当前种群大小:", "current_population_size", "0"),
            ("基因变异率:", "current_gene_mutation_rate", "0.000"),
            ("基因交叉率:", "current_gene_crossover_rate", "0.000"),
            ("染色体交叉率:", "current_chromosome_crossover_rate", "0.000"),
            ("高功率变异率:", "current_high_power_mutation_rate", "0.000"),  # 新增
            ("高功率交叉率:", "current_high_power_crossover_rate", "0.000"),  # 新增
            ("高功率搜索范围:", "current_high_power_search_range", "±0.0%"),  # 新增
            ("评估次数:", "eval_count", "0"),
            ("优化时间:", "opt_time", "00:00:00"),
            ("收敛状态:", "convergence_status", "未收敛"),  # 替换局部收敛
            ("锁定状态:", "lock_status", "未锁定"),
            ("漂移检测:", "drift_detection", "未检测")  # 新增
        ]
        
        self.status_labels = {}
        for i, (label, key, default) in enumerate(status_info):
            ttk.Label(info_frame, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            value_label = ttk.Label(info_frame, text=default, font=BOLD_FONT)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=5)
            self.status_labels[key] = value_label
        
        # 最佳位置信息 - 扩展为双端
        pos_frame = ttk.LabelFrame(status_grid, text="最佳位置（历史最佳）", style='Custom.TLabelframe')
        pos_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        
        # A端位置
        ttk.Label(pos_frame, text="A端位置:", font=BOLD_FONT).grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        positions_A = [("X (μm):", "A_x"), ("Y (μm):", "A_y"), ("Z (μm):", "A_z"), 
                    ("RX (rad):", "A_rx"), ("RY (rad):", "A_ry")]
        
        self.position_labels_A = {}
        for i, (label, key) in enumerate(positions_A):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=0, sticky=tk.W, padx=15, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            self.position_labels_A[key] = value_label
        
        # B端位置
        ttk.Label(pos_frame, text="B端位置:", font=BOLD_FONT).grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        positions_B = [("X (μm):", "B_x"), ("Y (μm):", "B_y"), ("Z (μm):", "B_z"), 
                    ("RX (rad):", "B_rx"), ("RY (rad):", "B_ry")]
        
        self.position_labels_B = {}
        for i, (label, key) in enumerate(positions_B):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=2, sticky=tk.W, padx=15, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=3, sticky=tk.W, padx=5, pady=2)
            self.position_labels_B[key] = value_label
        
        # ==================== 新增：高功率保持模式状态区域 ====================
        high_power_status_frame = ttk.LabelFrame(status_grid, text="高功率保持模式状态", style='Custom.TLabelframe')
        high_power_status_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=5, pady=5)
        
        # 创建三列显示
        high_power_grid = ttk.Frame(high_power_status_frame)
        high_power_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # 第一列：A端信息
        a_end_center_frame = ttk.LabelFrame(high_power_grid, text="A端搜索中心", style='Custom.TLabelframe')
        a_end_center_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=5, pady=5)
        
        self.center_A_labels = {}
        center_A_info = [
            ("X:", "center_A_x", "0.0000"),
            ("Y:", "center_A_y", "0.0000"),
            ("Z:", "center_A_z", "0.0000"),
            ("RX:", "center_A_rx", "0.0000"),
            ("RY:", "center_A_ry", "0.0000")
        ]
        
        for i, (label, key, default) in enumerate(center_A_info):
            ttk.Label(a_end_center_frame, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            value_label = ttk.Label(a_end_center_frame, text=default, font=BOLD_FONT)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.center_A_labels[key] = value_label
        
        # 第二列：B端信息
        b_end_center_frame = ttk.LabelFrame(high_power_grid, text="B端搜索中心", style='Custom.TLabelframe')
        b_end_center_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=5, pady=5)
        
        self.center_B_labels = {}
        center_B_info = [
            ("X:", "center_B_x", "0.0000"),
            ("Y:", "center_B_y", "0.0000"),
            ("Z:", "center_B_z", "0.0000"),
            ("RX:", "center_B_rx", "0.0000"),
            ("RY:", "center_B_ry", "0.0000")
        ]
        
        for i, (label, key, default) in enumerate(center_B_info):
            ttk.Label(b_end_center_frame, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            value_label = ttk.Label(b_end_center_frame, text=default, font=BOLD_FONT)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.center_B_labels[key] = value_label
        
        # 第三列：状态信息
        power_status_frame = ttk.LabelFrame(high_power_grid, text="高功率参数", style='Custom.TLabelframe')
        power_status_frame.grid(row=0, column=2, sticky=tk.NSEW, padx=5, pady=5)
        
        power_status_info = [
            ("搜索范围:", "high_power_search_range", "±5.0%"),
            ("扰动强度:", "high_power_perturbation", "0.010"),
            ("种群大小:", "high_power_population_size", "20"),
            ("变异率:", "high_power_mutation_rate", "0.050"),
            ("交叉率:", "high_power_crossover_rate", "0.300")
        ]
        
        self.power_status_labels = {}
        for i, (label, key, default) in enumerate(power_status_info):
            ttk.Label(power_status_frame, text=label, font=LARGE_FONT).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            value_label = ttk.Label(power_status_frame, text=default, font=BOLD_FONT)
            value_label.grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
            self.power_status_labels[key] = value_label
        
        # 模式切换按钮区域 - 删除局部优化按钮
        mode_btn_frame = ttk.Frame(self.status_frame)
        mode_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.keep_mode_btn = ttk.Button(mode_btn_frame, text="最高功率保持模式", command=self.switch_to_keep_mode, state=tk.DISABLED, style='Custom.TButton')
        self.keep_mode_btn.pack(side=tk.LEFT, padx=5)
        
        self.lock_mode_btn = ttk.Button(mode_btn_frame, text="位置锁定模式", command=self.switch_to_lock_mode, state=tk.NORMAL, style='Custom.TButton')
        self.lock_mode_btn.pack(side=tk.LEFT, padx=5)
        
        # 优化过程图表区域
        optimization_chart_frame = ttk.LabelFrame(self.status_frame, text="优化过程", style='Custom.TLabelframe')
        optimization_chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.fig, (self.ax, self.power_ax) = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=optimization_chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 优化过程图表
        self.best_fitness_line, = self.ax.plot([], [], 'r-', label='最佳功率')
        self.avg_fitness_line, = self.ax.plot([], [], 'b-', label='平均功率')
        self.ax.set_xlabel('代数', fontproperties='SimHei', fontsize=10)
        self.ax.set_ylabel('功率 (mW)', fontproperties='SimHei', fontsize=10)
        self.ax.set_title('优化过程', fontproperties='SimHei', fontsize=12)
        self.ax.legend(prop={'family': 'SimHei', 'size': 10})
        self.ax.grid(True, alpha=0.3)
        
        # 实时功率监控图表
        self.power_line, = self.power_ax.plot([], [], 'g-', label='实时功率')
        self.power_ax.set_xlabel('时间 (秒)', fontproperties='SimHei', fontsize=10)
        self.power_ax.set_ylabel('功率 (mW)', fontproperties='SimHei', fontsize=10)
        self.power_ax.set_title('实时功率监控', fontproperties='SimHei', fontsize=12)
        self.power_ax.legend(prop={'family': 'SimHei', 'size': 10})
        self.power_ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 进度条
        progress_frame = ttk.Frame(self.status_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)

    # 添加功率格式化方法
    def format_power_value(self, power_value):
        """
        根据功率值格式化显示，自动选择合适单位
        
        参数:
            power_value: 功率值（W）
        
        返回:
            带单位的格式化字符串
        """
        if power_value is None or np.isnan(power_value):
            return "0.0000 mW"
        
        # 将功率值转换为浮点数
        try:
            power = float(power_value)
        except (ValueError, TypeError):
            return "0.0000 mW"
        
        # 根据功率大小选择合适的单位
        if power >= 1.0:  # 瓦特级别
            return f"{power:.4f} W"
        elif power >= 0.001:  # 毫瓦级别
            return f"{power*1000:.4f} mW"
        elif power >= 1e-6:  # 微瓦级别
            return f"{power*1e6:.4f} μW"
        elif power >= 1e-9:  # 纳瓦级别
            return f"{power*1e9:.4f} nW"
        elif power >= 1e-12:  # 皮瓦级别
            return f"{power*1e12:.4f} pW"
        else:  # 更小的级别
            return f"{power*1e15:.4f} fW"

    def init_results_frame(self):
        """初始化结果可视化标签页 - 完整保留原结果显示并扩展双端支持"""
        frame = ttk.Frame(self.results_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 结果信息
        info_frame = ttk.LabelFrame(frame, text="优化结果摘要", style='Custom.TLabelframe')
        info_frame.pack(fill=tk.X, padx=10, pady=10)
        
        result_info = [
            ("最佳功率:", "best_power", "0.000000"),
            ("总评估次数:", "total_evaluations", "0"),
            ("总代数:", "total_generations", "0"),
            ("优化时间:", "optimization_time", "00:00:00"),
            ("优化模式:", "result_mode", "单端"),
            ("最终阶段:", "final_phase", "未开始")
        ]
        
        self.result_labels = {}
        for i, (label, key, default) in enumerate(result_info):
            ttk.Label(info_frame, text=label, font=LARGE_FONT).grid(row=i//2, column=(i%2)*2, sticky=tk.W, padx=20, pady=5)
            value_label = ttk.Label(info_frame, text=default, font=BOLD_FONT)
            value_label.grid(row=i//2, column=(i%2)*2 + 1, sticky=tk.W, padx=5, pady=5)
            self.result_labels[key] = value_label
        
        # 最佳位置 - 扩展为双端
        pos_frame = ttk.LabelFrame(frame, text="最佳位置参数", style='Custom.TLabelframe')
        pos_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # A端位置
        ttk.Label(pos_frame, text="A端位置:", font=BOLD_FONT).grid(row=0, column=0, sticky=tk.W, padx=20, pady=2)
        positions_A = [("X (μm):", "A_x"), ("Y (μm):", "A_y"), ("Z (μm):", "A_z")]
        
        self.result_position_labels_A = {}
        for i, (label, key) in enumerate(positions_A):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=0, sticky=tk.W, padx=35, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=1, sticky=tk.W, padx=5, pady=2)
            self.result_position_labels_A[key] = value_label
        
        positions_A2 = [("RX (rad):", "A_rx"), ("RY (rad):", "A_ry")]
        for i, (label, key) in enumerate(positions_A2):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=2, sticky=tk.W, padx=35, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=3, sticky=tk.W, padx=5, pady=2)
            self.result_position_labels_A[key] = value_label
        
        # B端位置
        ttk.Label(pos_frame, text="B端位置:", font=BOLD_FONT).grid(row=0, column=4, sticky=tk.W, padx=20, pady=2)
        positions_B = [("X (μm):", "B_x"), ("Y (μm):", "B_y"), ("Z (μm):", "B_z")]
        
        self.result_position_labels_B = {}
        for i, (label, key) in enumerate(positions_B):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=4, sticky=tk.W, padx=35, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=5, sticky=tk.W, padx=5, pady=2)
            self.result_position_labels_B[key] = value_label
        
        positions_B2 = [("RX (rad):", "B_rx"), ("RY (rad):", "B_ry")]
        for i, (label, key) in enumerate(positions_B2):
            ttk.Label(pos_frame, text=label, font=LARGE_FONT).grid(row=i+1, column=6, sticky=tk.W, padx=35, pady=2)
            value_label = ttk.Label(pos_frame, text="0.000000", font=BOLD_FONT)
            value_label.grid(row=i+1, column=7, sticky=tk.W, padx=5, pady=2)
            self.result_position_labels_B[key] = value_label
        
        # 图表按钮
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.show_charts_btn = ttk.Button(btn_frame, text="显示详细图表", command=self.show_detailed_charts, state=tk.DISABLED, style='Custom.TButton')
        self.show_charts_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_data_btn = ttk.Button(btn_frame, text="保存优化数据", command=self.save_optimization_data, style='Custom.TButton')
        self.save_data_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_results_btn = ttk.Button(btn_frame, text="保存结果", command=self.save_results, state=tk.DISABLED, style='Custom.TButton')
        self.save_results_btn.pack(side=tk.RIGHT, padx=5)
        
        # 结果图表区域
        self.results_fig = None
        self.results_canvas = None

    def init_log_frame(self):
        """初始化日志标签页 - 完整保留原日志功能"""
        frame = ttk.LabelFrame(self.log_frame, text="运行日志", style='Custom.TLabelframe')
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=80, height=20, font=LARGE_FONT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)
        
        # 日志控制按钮
        btn_frame = ttk.Frame(self.log_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.clear_log_btn = ttk.Button(btn_frame, text="清空日志", command=self.clear_log, style='Custom.TButton')
        self.clear_log_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_log_btn = ttk.Button(btn_frame, text="保存日志", command=self.save_log, style='Custom.TButton')
        self.save_log_btn.pack(side=tk.RIGHT, padx=5)

    def log(self, message):
        """添加日志信息"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # 以下方法完整保留原GUI的所有功能
    def start_power_monitoring(self):
        """启动功率监控线程"""
        if not self.power_monitoring:
            self.power_monitoring = True
            self.monitoring_thread = threading.Thread(target=self._power_monitoring_worker, daemon=True)
            self.monitoring_thread.start()
            self.log("启动功率监控")

    def stop_power_monitoring(self):
        """停止功率监控"""
        self.power_monitoring = False
        self.log("停止功率监控")
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
            
            # 可选：记录详细的功率信息
            if hasattr(self, 'log_power_details') and self.log_power_details:
                engineering_notation = power_result.get("engineering_notation", "N/A")
                scientific_notation = power_result.get("scientific_notation", "N/A")
                self.log(f"功率详情: {engineering_notation} ({scientific_notation})")
                
            return power_value
        else:
            # 旧格式：直接返回功率数值
            return float(power_result)

    def update_power_display_with_details(self, power_result):
        """
        使用功率计的详细信息更新显示
        """
        if isinstance(power_result, dict):
            # 显示工程单位格式
            engineering_notation = power_result.get("engineering_notation", "N/A")
            power_value = power_result.get("power", 0.0)
            
            self.status_labels["current_power"]["text"] = f"{power_value:.6f}"
            
            # 可选：在日志中显示格式化的功率值
            if hasattr(self, 'show_engineering_units') and self.show_engineering_units:
                self.log(f"当前功率: {engineering_notation}")
        else:
            # 旧格式处理
            self.status_labels["current_power"]["text"] = f"{power_result:.6f}"
    def _power_monitoring_worker(self):
        """功率监控工作线程 - 修改以适配新的功率计返回格式，确保图表更新"""
        self.monitoring_start_time = time.time()
        
        while self.power_monitoring:
            try:
                # 只有在设备已初始化且有硬件适配器时才进行测量
                if self.device_initialized and self.hardware_adapter:
                    # 获取功率测量结果（字典格式）
                    power_result = self.hardware_adapter.measure_current_power()
                    
                    # 从字典中提取功率值
                    if isinstance(power_result, dict):
                        current_power = power_result.get("power", 0.0)
                        # 可选：记录完整的功率信息用于调试
                        if hasattr(self, 'debug_mode') and self.debug_mode:
                            self.log(f"功率测量结果: {power_result}")
                    else:
                        # 兼容旧版本：直接使用数值
                        current_power = power_result
                    
                    current_time = time.time() - self.monitoring_start_time
                    
                    # 限制历史数据长度，避免内存问题
                    if len(self.power_history) > 1000:
                        self.power_history = self.power_history[-500:]  # 保留最后500个数据点
                        self.time_history = self.time_history[-500:]
                    
                    self.power_history.append(current_power)
                    self.time_history.append(current_time)
                    
                    power_record = self._create_power_record(current_power, current_time)
                    self.gui_data['power_monitoring_records'].append(power_record)
                    
                    # 立即更新功率显示和图表
                    self.root.after(0, self._update_power_display, current_power, current_time)
                    
                else:
                    # 设备未初始化，等待一段时间再重试
                    time.sleep(5.0)
                
                # 固定监控间隔
                time.sleep(1.0)
                
            except Exception as e:
                self.log(f"功率监控错误: {str(e)}")
                time.sleep(5.0)

    def _create_power_record(self, power_data, elapsed_time):
        """
        创建功率记录 - 修改以适配新的功率计返回格式
        """
        # 提取功率值
        if isinstance(power_data, dict):
            power_value = power_data.get("power", 0.0)
            power_range = power_data.get("power_range", None)
            scientific_notation = power_data.get("scientific_notation", "")
        else:
            power_value = power_data
            power_range = None
            scientific_notation = ""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': round(elapsed_time, 2),
            'power': float(power_value),
            'power_range': power_range,
            'scientific_notation': scientific_notation,
            'mode': self.current_mode,
            'optimization_running': self.is_running,
            'generation': self.current_generation,
            'optimization_mode': self.optimization_mode.get(),
            'optimization_phase': self.current_phase
        }

    def _update_power_display(self, power, current_time):
        """更新功率显示 - 使用格式化功率，强制更新图表"""
        formatted_power = self.format_power_value(power)
        if "current_power" in self.status_labels:
            self.status_labels["current_power"]["text"] = formatted_power
        
        # 强制更新功率图表
        try:
            self._update_power_chart()
        except Exception as e:
            self.log(f"更新功率图表失败: {str(e)}")
    def _update_power_chart(self):
        """更新实时功率图表 - 修复数据显示问题"""
        try:
            # 检查是否有功率数据
            if len(self.power_history) > 1 and len(self.time_history) == len(self.power_history):
                # 保存到图表数据
                self.chart_data['power_times'] = self.time_history
                self.chart_data['power_values'] = self.power_history
                
                # 清空现有曲线
                self.power_line.set_data([], [])
                
                # 设置新数据
                self.power_line.set_data(self.time_history, self.power_history)
                
                # 调整坐标轴范围
                self.power_ax.relim()
                self.power_ax.autoscale_view()
                
                # 设置图表标题和标签
                self.power_ax.set_xlabel('时间 (秒)', fontproperties='SimHei', fontsize=10)
                self.power_ax.set_ylabel('功率 (mW)', fontproperties='SimHei', fontsize=10)
                
                # 根据数据长度设置不同的标题
                if len(self.time_history) > 0:
                    max_time = max(self.time_history)
                    self.power_ax.set_title(f'实时功率监控 (时长: {max_time:.1f}秒)', fontproperties='SimHei', fontsize=12)
                else:
                    self.power_ax.set_title('实时功率监控', fontproperties='SimHei', fontsize=12)
                
                self.power_ax.legend(prop={'family': 'SimHei', 'size': 10})
                self.power_ax.grid(True, alpha=0.3)
                
                # 重绘画布
                self.canvas.draw_idle()  # 使用draw_idle避免阻塞
                
        except Exception as e:
            self.log(f"更新功率图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_power_monitoring_data(self):
        """保存功率监测数据 - 完整保留原功能"""
        if not self.gui_data['power_monitoring_records']:
            messagebox.showinfo("提示", "没有功率监测数据可保存")
            return
        
        try:
            # 创建保存对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")],
                title="保存功率监测数据"
            )
            
            if not file_path:
                return
            
            # 准备功率数据
            power_data = {
                'metadata': {
                    'save_time': datetime.now().isoformat(),
                    'total_records': len(self.gui_data['power_monitoring_records']),
                    'data_type': 'power_monitoring',
                    'session_id': self.current_session.get('session_id', '')
                },
                'power_data': self.gui_data['power_monitoring_records']
            }
            
            if file_path.endswith('.csv'):
                # 保存为CSV格式
                self._save_power_monitoring_csv(file_path, power_data)
            else:
                # 默认保存为JSON格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(power_data, f, indent=2, ensure_ascii=False)
            
            self.log(f"功率监测数据已保存到: {file_path}")
            messagebox.showinfo("成功", f"功率监测数据保存成功\n记录数量: {len(self.gui_data['power_monitoring_records'])}")
            
        except Exception as e:
            self.log(f"保存功率监测数据失败: {str(e)}")
            messagebox.showerror("错误", f"保存功率监测数据失败: {str(e)}")

    def _save_power_monitoring_csv(self, file_path, power_data):
        """保存功率监测数据为CSV格式"""
        import csv
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['timestamp', 'elapsed_time', 'power', 'mode', 'optimization_running', 'generation', 'optimization_mode', 'optimization_phase'])
            
            # 写入数据
            for record in power_data.get('power_data', []):
                writer.writerow([
                    record.get('timestamp', ''),
                    record.get('elapsed_time', ''),
                    record.get('power', ''),
                    record.get('mode', ''),
                    record.get('optimization_running', ''),
                    record.get('generation', ''),
                    record.get('optimization_mode', ''),
                    record.get('optimization_phase', '')
                ])

    def initialize_device(self):
        """初始化硬件设备 - 根据选择的模式创建硬件适配器"""
        if self.is_running:
            messagebox.showinfo("提示", "优化正在进行中，无法初始化设备")
            return
        
        try:
            self.log("开始初始化设备...")
            self.status_labels["device_status"]["text"] = "初始化中"
            
            # 根据优化模式创建硬件适配器
            mode = self.optimization_mode.get()
            if mode == "single":
                # 单端模式
                self.hardware_adapter = HardwareAdapter(mode="single")
                self.log("创建单端模式硬件适配器")
            else:
                # 双端模式
                self.hardware_adapter = HardwareAdapter(mode="dual")
                self.log("创建双端模式硬件适配器")
            
            # 1. 初始化功率计
            self.log("正在初始化功率计...")
            success, message = self.device_manager.initialize_power_meter(wavelength=1550)
            if not success:
                raise Exception(f"功率计初始化失败: {message}")
            self.log(f"功率计初始化: {message}")

            # 2. 初始化PZT控制器 - 根据模式决定初始化哪些控制器
            if mode == "single":
                # 单端模式只初始化A端控制器
                controllers = [
                    ("A端位置控制器", "71897216"),
                    ("A端角度控制器", "71450124"),
                ]
            else:
                # 双端模式初始化A端和B端控制器
                controllers = [
                    ("A端位置控制器", "71897156"),
                    ("A端角度控制器", "71910880"),
                    ("B端位置控制器", "71897216"),  
                    ("B端角度控制器", "71450124"),  
                ]
            
            for name, serial_no in controllers:
                self.log(f"正在初始化{name}...")
                success, message = self.device_manager.initialize_pzt_controller(name, serial_no)
                if not success:
                    raise Exception(f"{name}初始化失败: {message}")
                self.log(f"{name}初始化: {message}")

            # 3. 等待设备稳定
            self.log("等待设备稳定...")
            for i in range(10, 0, -1):
                self.status_labels["device_status"]["text"] = f"初始化中...等待{i}秒"
                time.sleep(1)
            
            # 4. 执行设备归零
            self.log("执行设备归零操作...")
            self.hardware_adapter.zero_all()
            
            # 等待归零完成，带进度显示
            self.log("等待归零完成...")
            for i in range(40, 0, -1):
                self.status_labels["device_status"]["text"] = f"归零中...剩余{i}秒"
                time.sleep(1)
            
            # 5. 切换到开环模式作为默认
            self.log("切换设备到开环模式...")
            mode_code = 1  # 1:open-loop 2:closed-loop 
            modes = self.hardware_adapter.mode_switch(mode_code)
            self.current_piezo_mode = mode_code
            self.status_labels["piezo_mode"]["text"] = "开环模式"
            self.piezo_mode_btn["text"] = "切换为闭环模式"
            time.sleep(1)  # 等待模式切换完成
            
            self.device_initialized = True
            self.status_labels["device_status"]["text"] = "已初始化"
            
            # 设备初始化成功后启动功率监控
            self.start_power_monitoring()
            
            self.log("设备初始化成功")
            messagebox.showinfo("成功", "设备初始化成功")
                
        except Exception as e:
            self.device_initialized = False
            self.hardware_adapter = None  # 初始化失败时重置硬件适配器
            self.status_labels["device_status"]["text"] = "初始化失败"
            error_msg = f"设备初始化出错: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)

    def set_initial_position(self):
        """设置初始位置 - 改进版本，解决字典键不一致问题"""
        if self.is_running:
            messagebox.showinfo("提示", "优化正在进行中，无法设置初始位置")
            return
        
        if not self.device_initialized:
            messagebox.showwarning("警告", "请先初始化设备")
            return
        
        try:
            mode = self.optimization_mode.get()
            
            if mode == "single":
                # 单端模式：从界面获取初始位置参数
                initial_pos = {}
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.single_param_entries[f'{param}_initial'][0].get())
                        initial_pos[param] = value
                    except (ValueError, TypeError, KeyError):
                        initial_pos[param] = 0.0
                        self.log(f"警告: {param}初始值无效，使用默认值0.0")
                
                self.log(f"开始设置单端初始位置: {initial_pos}")
                
                # 移动到初始位置
                self.hardware_adapter.set_initial_positions(initial_pos)
                self.hardware_adapter.back_to_initial_positions()
                time.sleep(10)  # 等待位置调整完成
                
                # 更新状态显示
                for key, value in initial_pos.items():
                    if f"A_{key}" in self.position_labels_A:
                        self.position_labels_A[f"A_{key}"]["text"] = f"{value:.6f}"
                
            else:
                # 双端模式：分别设置A端和B端初始位置
                initial_pos = {}  # 使用统一的字典
                
                # A端参数
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.double_param_entries[f'A_{param}_initial'][0].get())
                        initial_pos[f'A_{param}'] = value  # 使用A_x格式
                    except (ValueError, TypeError, KeyError):
                        initial_pos[f'A_{param}'] = 0.0
                        self.log(f"警告: A端{param}初始值无效，使用默认值0.0")
                
                # B端参数
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.double_param_entries[f'B_{param}_initial'][0].get())
                        initial_pos[f'B_{param}'] = value  # 使用B_x格式
                    except (ValueError, TypeError, KeyError):
                        initial_pos[f'B_{param}'] = 0.0
                        self.log(f"警告: B端{param}初始值无效，使用默认值0.0")
                
                self.log(f"开始设置双端初始位置: {initial_pos}")
                
                # 移动到初始位置 - 硬件适配器会自动处理坐标转换
                self.hardware_adapter.set_initial_positions(initial_pos)
                self.hardware_adapter.back_to_initial_positions()
                time.sleep(10)  # 等待位置调整完成
                
                # 更新状态显示
                for key, value in initial_pos.items():
                    if key.startswith('A_'):
                        display_key = key.replace('A_', '')
                        if f"A_{display_key}" in self.position_labels_A:
                            self.position_labels_A[f"A_{display_key}"]["text"] = f"{value:.6f}"
                    elif key.startswith('B_'):
                        display_key = key.replace('B_', '')
                        if f"B_{display_key}" in self.position_labels_B:
                            self.position_labels_B[f"B_{display_key}"]["text"] = f"{value:.6f}"
            
            self.log("初始位置设置成功")
            messagebox.showinfo("成功", "初始位置设置成功")
            
        except Exception as e:
            error_msg = f"设置初始位置失败: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)

    def toggle_piezo_mode(self):
        """切换压电控制器的开环/闭环模式 - 完整保留原功能"""
        if self.is_running and self.current_mode == "search":
            messagebox.showinfo("提示", "优化正在进行中，无法切换模式")
            return
        
        try:
            if self.current_piezo_mode == 1:  # 当前是开环，切换到闭环
                new_mode = 2
                new_mode_text = "闭环模式"
                btn_text = "切换为开环模式"
            else:  # 当前是闭环，切换到开环或未设置
                new_mode = 1
                new_mode_text = "开环模式"
                btn_text = "切换为闭环模式"
            
            self.log(f"切换压电控制器模式至{new_mode_text}...")
            modes = self.hardware_adapter.mode_switch(new_mode)
            self.current_piezo_mode = new_mode
            self.status_labels["piezo_mode"]["text"] = new_mode_text
            self.piezo_mode_btn["text"] = btn_text
            time.sleep(3)  # 等待模式切换完成
            
            self.log(f"压电控制器已切换至{new_mode_text}")
            messagebox.showinfo("成功", f"已切换至{new_mode_text}")
            
        except Exception as e:
            error_msg = f"切换压电模式失败: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)

    # 修改 start_optimization 方法中的优化器初始化和回调设置部分

    # 在 start_optimization 方法中，添加优化器实例的创建代码
    def start_optimization(self):
        """开始优化过程 - 添加完整的参数回调支持"""
        if self.is_running:
            messagebox.showinfo("提示", "优化正在进行中")
            return
        
        if not self.device_initialized:
            messagebox.showwarning("警告", "请先初始化设备")
            return
        
        mode = self.optimization_mode.get()
        
        if mode == "single":
            # 检查单端优化参数选择
            selected_params = self.get_selected_optimization_params()
            if not selected_params:
                messagebox.showwarning("警告", "请至少选择一个优化参数")
                return
        else:
            # 检查双端优化参数选择
            selected_params_A = self.get_selected_optimization_params_A()
            selected_params_B = self.get_selected_optimization_params_B()
            if not selected_params_A and not selected_params_B:
                messagebox.showwarning("警告", "请至少为A端或B端选择一个优化参数")
                return
        
        # 获取并验证参数
        try:
            params = self.get_optimization_parameters()
            is_valid, message = self.validate_parameters(params)
            if not is_valid:
                messagebox.showerror("参数错误", message)
                return
                
            self.gui_data['optimization_parameters'] = self._create_serializable_parameters(params)
            self.gui_data['optimization_mode'] = mode
            
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return
        
        # 初始化新的优化会话
        self._init_optimization_session()
        
        # 记录优化开始时间
        self.optimization_start_time = time.time()
        self.current_session['start_time'] = datetime.now().isoformat()
        
        # 重置UI显示
        self._reset_status_display()
        
        # 初始化优化器实例
        try:
            # 检查GUI窗口是否仍然存在
            if not (hasattr(self, 'root') and self.root and 
                    hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
                self.log("GUI窗口已关闭，无法启动优化")
                return
            
            # 根据模式创建优化器实例
            if mode == "single":
                # 单端优化器
                self.optimizer = GeneticAlgorithmOptimizer(
                    hardware_adapter=self.hardware_adapter,
                    config=params
                )
                self.log("单端遗传算法优化器已创建")
            else:
                # 双端优化器
                self.optimizer = DualEndGeneticAlgorithmOptimizer(
                    hardware_adapter=self.hardware_adapter,
                    config=params
                )
                self.log("双端遗传算法优化器已创建")
            
            # 定义弱引用避免循环引用
            gui_ref = weakref.ref(self)
            
            # 定义参数请求回调函数
            def parameter_request_callback():
                """参数请求回调函数 - 从GUI获取最新参数"""
                gui = gui_ref()
                if gui:
                    try:
                        params = gui.get_optimization_parameters()
                        return params
                    except Exception as e:
                        gui.log(f"参数请求回调失败: {str(e)}")
                        return {}
                return {}
            
            # 设置所有回调函数
            def weak_progress_callback(data):
                gui = gui_ref()
                if gui:
                    gui._handle_optimizer_callback(data)
            
            def weak_finished_callback(result):
                gui = gui_ref()
                if gui:
                    gui._handle_optimization_finished(result)
            
            def weak_convergence_callback(fitness=None):
                gui = gui_ref()
                if gui:
                    gui.detect_convergence(fitness)
            
            def weak_lock_callback(position_dict=None, fitness=None):
                gui = gui_ref()
                if gui:
                    gui._handle_position_locked_callback(position_dict, fitness)
            
            # 设置回调函数（包括参数请求回调）
            self.optimizer.set_callbacks(
                progress_callback=weak_progress_callback,
                finished_callback=weak_finished_callback,
                convergence_callback=weak_convergence_callback,
                lock_callback=weak_lock_callback,
                request_parameters_callback=parameter_request_callback  # 新增参数请求回调
            )
            
            # 更新UI状态
            self.is_running = True
            self.current_mode = "search"
            self.is_converged = False
            self.status_labels["operation_mode"]["text"] = "搜索模式"
            self.status_labels["status"]["text"] = "运行中"
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.init_device_btn.config(state=tk.DISABLED)
            self.set_initial_pos_btn.config(state=tk.DISABLED)
            self.keep_mode_btn.config(state=tk.NORMAL)
            self.lock_mode_btn.config(state=tk.NORMAL)
            
            # 记录启动信息
            if mode == "single":
                self.log("开始单端遗传算法优化...")
                self.log(f"优化参数: {', '.join(selected_params)}")
                self.log(f"初始参数配置:")
                self.log(f"  种群大小: {params.get('population_size', 'N/A')}")
                self.log(f"  变异率: {params.get('mutation_rate', params.get('gene_mutation_rate', 'N/A')):.3f}")
                self.log(f"  交叉率: {params.get('crossover_rate', params.get('gene_crossover_rate', 'N/A')):.3f}")
            else:
                self.log("开始双端遗传算法优化...")
                self.log(f"A端优化参数: {', '.join(selected_params_A)}")
                self.log(f"B端优化参数: {', '.join(selected_params_B)}")
                self.log(f"初始参数配置:")
                self.log(f"  种群大小: {params.get('population_size', 'N/A')}")
                self.log(f"  基因变异率: {params.get('gene_mutation_rate', 'N/A'):.3f}")
                self.log(f"  基因交叉率: {params.get('gene_crossover_rate', 'N/A'):.3f}")
                self.log(f"  染色体交叉率: {params.get('chromosome_crossover_rate', 'N/A'):.3f}")
                self.log(f"  高功率保持模式 - 种群大小: {params.get('high_power_population_size', 'N/A')}")
                self.log(f"  高功率变异率: {params.get('high_power_mutation_rate', 'N/A'):.3f}")
                self.log(f"  位置锁定阈值: {params.get('lock_mode_threshold', 'N/A')*100:.2f}%")
                self.log(f"  局部搜索率: {params.get('local_search_rate', params.get('local_search_range_percent', 'N/A'))*100:.2f}%")
            
            # 启动优化线程
            self.optimization_thread = threading.Thread(
                target=self.optimizer.run,
                daemon=True
            )
            self.optimization_thread.start()
            
            # 启动参数监控
            self.start_parameter_monitoring()
            
        except Exception as e:
            messagebox.showerror("初始化错误", f"优化器初始化失败: {str(e)}")
            self.log(f"优化器初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return
    # 修改 get_optimization_parameters 方法，确保所有参数都被正确获取

    def get_optimization_parameters(self):
        """
        获取当前所有优化参数（线程安全）
        返回包含所有参数的字典
        """
        with self.parameters_lock:
            mode = self.optimization_mode.get()
            
            if mode == "single":
                params = self._get_single_parameters()
            else:
                params = self._get_double_parameters()
            
            # 确保所有必需参数都存在
            params = self._ensure_all_parameters(params)
            
            # 添加额外的元数据
            params['_gui_timestamp'] = datetime.now().isoformat()
            params['_optimization_mode'] = mode
            params['_session_id'] = self.current_session.get('session_id', '')
            
            # 记录获取时间
            self.last_parameter_request = time.time()
            
            return params

    def _ensure_all_parameters(self, params):
        """
        确保参数字典包含所有必需的参数
        """
        # 基本参数默认值
        default_params = {
            # 遗传算法参数
            'population_size': 30,
            'generations': 200,
            'gene_mutation_rate': 0.15,
            'gene_crossover_rate': 0.8,
            'chromosome_crossover_rate': 0.2,
            'elite_size': 8,
            'tournament_size': 10,
            
            # 收敛检测参数
            'convergence_threshold': 0.05,
            'convergence_patience': 8,
            'enhanced_exploration_max': 4,
            'enhanced_mutation_rate': 0.7,
            'converged_mutation_rate': 0.02,
            'fitness_variance_threshold': 0.005,
            
            # 高功率保持模式参数
            'high_power_population_size': 20,
            'high_power_mutation_rate': 0.05,
            'high_power_crossover_rate': 0.3,
            
            # 位置锁定参数
            'lock_mode_threshold': 0.001,
            
            # 警报参数
            'alert_threshold_percent': 0.05,
            
            # 局部优化参数
            'local_search_rate': 0.4,
            'local_search_range_percent': 0.01,
            
            # 其他参数
            'elite_protection': True,
            'elite_clone_rate': 0.25,
            'adaptive_mutation_rate': True,
            'adaptive_crossover_rate': True,
            
            # 双端特定参数
            'light_threshold': 0.0002,
            
            # 搜索范围参数
            'search_range': {
                'x': (0, 30),
                'y': (0, 30),
                'z': (0, 30),
                'rx': (0, 0.03),
                'ry': (0, 0.03)
            }
        }
        
        # 如果参数中已存在，则保留原值，否则使用默认值
        for key, default_value in default_params.items():
            if key not in params:
                params[key] = default_value
            elif params[key] is None:
                params[key] = default_value
        
        # 确保百分比参数正确转换
        percent_params = ['convergence_threshold', 'lock_mode_threshold', 
                        'alert_threshold_percent', 'local_search_range_percent']
        
        for param in percent_params:
            if param in params:
                value = params[param]
                if isinstance(value, (int, float)):
                    # 如果输入的是百分比值（如5表示5%），转换为小数（0.05）
                    if value > 1.0:  # 假设用户输入的是百分比
                        params[param] = value / 100.0
        
        # 确保必要参数存在
        if 'selected_variables_A' not in params:
            params['selected_variables_A'] = self.get_selected_optimization_params_A()
        
        if 'selected_variables_B' not in params:
            params['selected_variables_B'] = self.get_selected_optimization_params_B()
        
        return params

    def validate_parameters(self, params):
        """
        验证参数的有效性
        返回：(是否有效, 错误消息)
        """
        try:
            errors = []
            
            # 1. 遗传算法参数验证
            ga_params = [
                ('population_size', 5, 100, "种群大小"),
                ('generations', 10, 1000, "最大代数"),
                ('elite_size', 1, 50, "精英数量"),
                ('tournament_size', 2, 20, "锦标赛大小")
            ]
            
            for key, min_val, max_val, name in ga_params:
                if key in params:
                    value = params[key]
                    if not isinstance(value, (int, float)):
                        errors.append(f"{name}必须是数值")
                    elif value < min_val or value > max_val:
                        errors.append(f"{name}必须在{min_val}-{max_val}之间")
            
            # 2. 概率参数验证 (0-1之间)
            probability_params = [
                ('gene_mutation_rate', "基因变异率"),
                ('gene_crossover_rate', "基因交叉率"),
                ('chromosome_crossover_rate', "染色体交叉率"),
                ('enhanced_mutation_rate', "增强变异率"),
                ('high_power_mutation_rate', "高功率变异率"),
                ('high_power_crossover_rate', "高功率交叉率"),
                ('elite_clone_rate', "精英克隆率"),
                ('local_search_rate', "局部搜索率"),
                ('local_search_range_percent', "局部搜索范围")
            ]
            
            for key, name in probability_params:
                if key in params:
                    value = params[key]
                    if not isinstance(value, (int, float)):
                        errors.append(f"{name}必须是数值")
                    elif value < 0 or value > 1:
                        errors.append(f"{name}必须在0.0-1.0之间")
            
            # 3. 百分比参数验证 (0-1之间)
            percent_params = [
                ('convergence_threshold', "收敛阈值"),
                ('lock_mode_threshold', "位置锁定阈值"),
                ('alert_threshold_percent', "警报阈值")
            ]
            
            for key, name in percent_params:
                if key in params:
                    value = params[key]
                    if not isinstance(value, (int, float)):
                        errors.append(f"{name}必须是数值")
                    elif value < 0 or value > 1:
                        errors.append(f"{name}必须在0.0-1.0之间")
            
            # 4. 整数参数验证
            integer_params = [
                ('convergence_patience', 3, 50, "收敛耐心值"),
                ('enhanced_exploration_max', 1, 20, "增强探索次数"),
                ('high_power_population_size', 5, 50, "高功率种群大小")
            ]
            
            for key, min_val, max_val, name in integer_params:
                if key in params:
                    value = params[key]
                    if not isinstance(value, int):
                        errors.append(f"{name}必须是整数")
                    elif value < min_val or value > max_val:
                        errors.append(f"{name}必须在{min_val}-{max_val}之间")
            
            # 5. 布尔参数验证
            boolean_params = [
                'adaptive_mutation_rate',
                'adaptive_crossover_rate',
                'elite_protection'
            ]
            
            for key in boolean_params:
                if key in params:
                    value = params[key]
                    if not isinstance(value, bool):
                        errors.append(f"{key}必须是布尔值")
            
            # 6. 搜索范围验证
            search_range_keys = ['search_range', 'search_range_A', 'search_range_B']
            for key in search_range_keys:
                if key in params:
                    search_range = params[key]
                    if not isinstance(search_range, dict):
                        errors.append(f"{key}必须是字典")
                    else:
                        for axis, (min_val, max_val) in search_range.items():
                            if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
                                errors.append(f"{key}.{axis}范围必须是数值")
                            elif min_val >= max_val:
                                errors.append(f"{key}.{axis}最小值必须小于最大值")
            
            if errors:
                return False, "\n".join(errors)
            else:
                return True, "参数验证通过"
            
        except Exception as e:
            return False, f"参数验证异常: {str(e)}"
    # 在 DualEndGAOptimizerGUI 类中添加以下方法
    def update_high_power_parameters(self):
        """更新高功率保持模式参数"""
        try:
            # 检查是否在双端模式
            if self.optimization_mode.get() != "double":
                messagebox.showinfo("提示", "高功率保持模式参数仅适用于双端优化模式")
                return
            
            # 获取高功率保持模式参数
            params = {}
            
            # 检查并获取每个参数
            try:
                params['high_power_search_range_percent'] = float(self.double_param_entries['high_power_search_range_percent'][0].get()) / 100.0
            except ValueError:
                messagebox.showerror("错误", "高功率搜索范围格式错误")
                return
                
            try:
                params['high_power_perturbation_strength'] = float(self.double_param_entries['high_power_perturbation_strength'][0].get())
            except ValueError:
                messagebox.showerror("错误", "克隆扰动强度格式错误")
                return
                
            try:
                params['high_power_population_size'] = int(self.double_param_entries['high_power_population_size'][0].get())
            except ValueError:
                messagebox.showerror("错误", "高功率种群大小格式错误")
                return
                
            try:
                params['high_power_mutation_rate'] = float(self.double_param_entries['high_power_mutation_rate'][0].get())
            except ValueError:
                messagebox.showerror("错误", "高功率变异率格式错误")
                return
                
            try:
                params['high_power_crossover_rate'] = float(self.double_param_entries['high_power_crossover_rate'][0].get())
            except ValueError:
                messagebox.showerror("错误", "高功率交叉率格式错误")
                return
            
            # 验证参数
            errors = []
            if params['high_power_search_range_percent'] <= 0 or params['high_power_search_range_percent'] > 0.2:
                errors.append("高功率搜索范围必须在0.1%-20%之间")
            if params['high_power_perturbation_strength'] <= 0 or params['high_power_perturbation_strength'] > 0.1:
                errors.append("克隆扰动强度必须在0.001-0.1之间")
            if params['high_power_population_size'] < 5 or params['high_power_population_size'] > 50:
                errors.append("高功率种群大小必须在5-50之间")
            if params['high_power_mutation_rate'] <= 0 or params['high_power_mutation_rate'] > 0.5:
                errors.append("高功率变异率必须在0.001-0.5之间")
            if params['high_power_crossover_rate'] < 0.05 or params['high_power_crossover_rate'] > 1.0:
                errors.append("高功率交叉率必须在0.05-1.0之间")
            
            if errors:
                messagebox.showerror("参数错误", "\n".join(errors))
                return
            
            # 发送到优化器
            if self.optimizer and hasattr(self.optimizer, 'update_high_power_parameters'):
                success, message = self.optimizer.update_high_power_parameters(params)
                if success:
                    self.log(f"高功率保持模式参数更新成功: {message}")
                    
                    # 更新状态显示
                    self.power_status_labels["high_power_search_range"]["text"] = f"±{params['high_power_search_range_percent']*100:.1f}%"
                    self.power_status_labels["high_power_perturbation"]["text"] = f"{params['high_power_perturbation_strength']:.3f}"
                    self.status_labels["current_high_power_mutation_rate"]["text"] = f"{params['high_power_mutation_rate']:.3f}"
                    self.status_labels["current_high_power_crossover_rate"]["text"] = f"{params['high_power_crossover_rate']:.3f}"
                    self.status_labels["current_high_power_search_range"]["text"] = f"±{params['high_power_search_range_percent']*100:.1f}%"
                    
                    # 显示成功消息
                    messagebox.showinfo("成功", "高功率保持模式参数更新成功")
                else:
                    self.log(f"高功率保持模式参数更新失败: {message}")
                    messagebox.showerror("错误", f"高功率保持模式参数更新失败: {message}")
            else:
                self.log("优化器不存在或不支持高功率保持模式参数更新")
                messagebox.showwarning("警告", "优化器不存在或不支持高功率保持模式参数更新")
                    
        except ValueError as e:
            self.log(f"高功率保持模式参数格式错误: {str(e)}")
            messagebox.showerror("错误", f"参数格式错误: {str(e)}")
        except Exception as e:
            self.log(f"更新高功率保持模式参数失败: {str(e)}")
            messagebox.showerror("错误", f"更新高功率保持模式参数失败: {str(e)}")
    def _update_high_power_status_display(self, status_data):
        """更新高功率保持模式状态显示"""
        try:
            # 更新中心坐标显示
            center_A = status_data.get('center_individual_A')
            center_B = status_data.get('center_individual_B')
            
            if center_A:
                for i, var in enumerate(['x', 'y', 'z', 'rx', 'ry']):
                    if i < len(center_A) and f"center_A_{var}" in self.center_A_labels:
                        self.center_A_labels[f"center_A_{var}"]["text"] = f"{center_A[i]:.4f}"
            
            if center_B:
                for i, var in enumerate(['x', 'y', 'z', 'rx', 'ry']):
                    if i < len(center_B) and f"center_B_{var}" in self.center_B_labels:
                        self.center_B_labels[f"center_B_{var}"]["text"] = f"{center_B[i]:.4f}"
            
            # 更新功率状态
            best_fitness = status_data.get('best_fitness')
            current_fitness = status_data.get('current_fitness')
            search_range_percent = status_data.get('high_power_search_range_percent', 0.05)
            perturbation_strength = status_data.get('high_power_perturbation_strength', 0.01)
            drift_detected = status_data.get('drift_detected', False)
            
            if best_fitness is not None:
                self.power_status_labels["high_power_best"]["text"] = self.format_power_value(best_fitness)
            
            if current_fitness is not None:
                self.power_status_labels["high_power_current"]["text"] = self.format_power_value(current_fitness)
            
            self.power_status_labels["high_power_search_range"]["text"] = f"±{search_range_percent*100:.1f}%"
            self.power_status_labels["high_power_perturbation"]["text"] = f"{perturbation_strength:.3f}"
            
            # 更新漂移检测状态
            drift_status = "检测到漂移" if drift_detected else "无漂移"
            self.power_status_labels["high_power_drift"]["text"] = drift_status
            self.status_labels["drift_detection"]["text"] = drift_status
            
            # 更新状态栏的高功率参数显示
            high_power_mutation_rate = status_data.get('high_power_mutation_rate')
            high_power_crossover_rate = status_data.get('high_power_crossover_rate')
            
            if high_power_mutation_rate is not None:
                self.status_labels["current_high_power_mutation_rate"]["text"] = f"{high_power_mutation_rate:.3f}"
            
            if high_power_crossover_rate is not None:
                self.status_labels["current_high_power_crossover_rate"]["text"] = f"{high_power_crossover_rate:.3f}"
            
            self.status_labels["current_high_power_search_range"]["text"] = f"±{search_range_percent*100:.1f}%"
            
        except Exception as e:
            self.log(f"更新高功率保持模式状态显示失败: {str(e)}")
    def start_parameter_monitoring(self, interval=2.0):
        """
        启动参数监控定时器
        定期检查参数是否需要更新到优化器
        """
        # 先停止现有的监控
        self.stop_parameter_monitoring()
        
        def monitor_parameters():
            try:
                if self.is_running and self.optimizer:
                    # 检查是否有参数需要更新
                    if hasattr(self, 'parameters_need_update') and self.parameters_need_update:
                        # 获取最新参数
                        params = self.get_optimization_parameters()
                        
                        # 验证参数
                        is_valid, message = self.validate_parameters(params)
                        if is_valid:
                            # 更新优化器参数
                            success, msg = self.update_optimizer_parameters(params)
                            if success:
                                self.log(f"参数已更新到优化器: {msg}")
                                self.parameters_need_update = False
                            else:
                                self.log(f"参数更新失败: {msg}")
                        else:
                            self.log(f"参数验证失败，不更新: {message}")
            except Exception as e:
                self.log(f"参数监控错误: {str(e)}")
                import traceback
                traceback.print_exc()
            
            # 重新启动定时器（只在GUI窗口还存在的情况下）
            if (hasattr(self, 'root') and self.root and 
                hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
                try:
                    self.parameter_monitor_timer = self.root.after(
                        int(interval * 1000), monitor_parameters)
                except tk.TclError as e:
                    if "id must be a valid identifier" not in str(e):
                        self.log(f"定时器启动失败: {str(e)}")
                    # GUI窗口可能已关闭，停止定时器
                    return
            else:
                # GUI窗口不存在，停止定时器
                return
        
        # 启动监控
        try:
            if (hasattr(self, 'root') and self.root and 
                hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
                self.parameter_monitor_timer = self.root.after(1000, monitor_parameters)
                self.log(f"参数监控已启动，间隔: {interval}秒")
        except tk.TclError as e:
            self.log(f"启动参数监控失败: {str(e)}")

    def stop_parameter_monitoring(self):
        """停止参数监控"""
        try:
            if (hasattr(self, 'parameter_monitor_timer') and 
                self.parameter_monitor_timer is not None):
                try:
                    # 只取消有效的定时器ID
                    if (hasattr(self, 'root') and self.root and 
                        hasattr(self.root, 'after_info') and 
                        self.parameter_monitor_timer in self.root.tk.call('after', 'info')):
                        self.root.after_cancel(self.parameter_monitor_timer)
                except (tk.TclError, AttributeError, KeyError):
                    # 定时器ID可能已失效或不存在，忽略此错误
                    pass
                finally:
                    self.parameter_monitor_timer = None
            self.log("参数监控已停止")
        except Exception as e:
            self.log(f"停止参数监控时出错: {str(e)}")

    def update_optimizer_parameters(self, params):
        """
        更新优化器参数
        
        返回:
            success: 是否成功
            message: 结果消息
        """
        if not self.optimizer:
            return False, "优化器不存在"
        
        try:
            # 调用优化器的参数更新方法
            if hasattr(self.optimizer, 'update_parameters_from_gui'):
                self.optimizer.update_parameters_from_gui(params)
                return True, f"更新了 {len(params)} 个参数"
            else:
                # 如果优化器不支持直接更新，通过回调函数传递
                if hasattr(self.optimizer, 'request_parameters_callback'):
                    # 设置参数请求回调，让优化器自己获取
                    return True, "参数更新请求已发送"
                else:
                    return False, "优化器不支持参数更新"
        except Exception as e:
            return False, f"更新失败: {str(e)}"

    def _handle_parameter_update(self, updated_params):
        """
        处理来自优化器的参数更新
        当优化器自适应调整参数时调用
        """
        try:
            if not updated_params:
                return
            
            # 记录更新事件
            update_count = len(updated_params)
            self.log(f"优化器自适应参数更新 ({update_count} 个参数):")
            
            for key, value in updated_params.items():
                self.log(f"  {key}: {value}")
                
                # 更新当前参数存储
                self.current_parameters[key] = value
            
            # 更新UI显示
            self.update_parameter_display()
            
            # 记录到优化历史
            if 'enhanced_exploration_events' in self.gui_data['optimization_history']:
                self.gui_data['optimization_history']['enhanced_exploration_events'].append({
                    'event_type': 'optimizer_adaptive_update',
                    'timestamp': datetime.now().isoformat(),
                    'updated_parameters': updated_params,
                    'generation': self.current_generation
                })
                
        except Exception as e:
            self.log(f"处理参数更新失败: {str(e)}")
    def switch_to_keep_mode(self):
        """
        切换到最高功率保持模式 - 由GUI按钮触发
        """
        if not self.optimizer:
            messagebox.showwarning("警告", "没有优化器实例，无法切换到高功率保持模式")
            return
        
        if not self.is_converged and not (self.result and self.result.get('final_convergence', False)):
            messagebox.showwarning("警告", "优化尚未收敛，无法切换到高功率保持模式")
            return
        
        try:
            self.log("从GUI启动高功率保持模式...")
            
            # 检查优化器是否支持高功率保持模式
            if not hasattr(self.optimizer, 'start_high_power_keep_mode_from_gui'):
                messagebox.showwarning("警告", "优化器不支持从GUI启动高功率保持模式")
                return
            
            # 获取最佳个体
            best_individual_A = None
            best_individual_B = None
            
            if hasattr(self.optimizer, 'best_individual_A_memory'):
                best_individual_A = self.optimizer.best_individual_A_memory
            elif hasattr(self.optimizer, 'best_individual_A'):
                best_individual_A = self.optimizer.best_individual_A
            
            if hasattr(self.optimizer, 'best_individual_B_memory'):
                best_individual_B = self.optimizer.best_individual_B_memory
            elif hasattr(self.optimizer, 'best_individual_B'):
                best_individual_B = self.optimizer.best_individual_B
            
            if best_individual_A is None or best_individual_B is None:
                messagebox.showwarning("警告", "无法获取最佳个体位置")
                return
            
            # 获取当前适应度
            current_fitness = self.optimizer.best_fitness if hasattr(self.optimizer, 'best_fitness') else 0.0
            
            # 从GUI获取高功率保持模式参数
            try:
                high_power_params = {
                    'high_power_search_range_percent': float(self.double_param_entries['high_power_search_range_percent'][0].get()) / 100.0,
                    'high_power_perturbation_strength': float(self.double_param_entries['high_power_perturbation_strength'][0].get()),
                    'high_power_population_size': int(self.double_param_entries['high_power_population_size'][0].get()),
                    'high_power_mutation_rate': float(self.double_param_entries['high_power_mutation_rate'][0].get()),
                    'high_power_crossover_rate': float(self.double_param_entries['high_power_crossover_rate'][0].get())
                }
                
                # 验证参数
                errors = []
                if high_power_params['high_power_search_range_percent'] <= 0 or high_power_params['high_power_search_range_percent'] > 0.2:
                    errors.append("高功率搜索范围必须在0.1%-20%之间")
                if high_power_params['high_power_perturbation_strength'] <= 0 or high_power_params['high_power_perturbation_strength'] > 0.1:
                    errors.append("克隆扰动强度必须在0.001-0.1之间")
                if high_power_params['high_power_population_size'] < 5 or high_power_params['high_power_population_size'] > 50:
                    errors.append("高功率种群大小必须在5-50之间")
                if high_power_params['high_power_mutation_rate'] <= 0 or high_power_params['high_power_mutation_rate'] > 0.5:
                    errors.append("高功率变异率必须在0.001-0.5之间")
                if high_power_params['high_power_crossover_rate'] < 0.05 or high_power_params['high_power_crossover_rate'] > 1.0:
                    errors.append("高功率交叉率必须在0.05-1.0之间")
                
                if errors:
                    messagebox.showerror("参数错误", "\n".join(errors))
                    return
                    
            except ValueError as e:
                messagebox.showerror("错误", f"参数格式错误: {str(e)}")
                return
            
            # 启动高功率保持模式
            success = self.optimizer.start_high_power_keep_mode_from_gui(
                center_individual_A=best_individual_A,
                center_individual_B=best_individual_B,
                current_fitness=current_fitness
            )
            
            if success:
                self.high_power_mode_enabled = True
                self.current_mode = "keep"
                self.status_labels["operation_mode"]["text"] = "高功率保持模式"
                self.status_labels["status"]["text"] = "高功率保持模式"
                
                # 更新高功率参数显示
                self.power_status_labels["high_power_search_range"]["text"] = f"±{high_power_params['high_power_search_range_percent']*100:.1f}%"
                self.power_status_labels["high_power_perturbation"]["text"] = f"{high_power_params['high_power_perturbation_strength']:.3f}"
                
                self.log(f"高功率保持模式已启动，搜索范围: ±{high_power_params['high_power_search_range_percent']*100}%")
                messagebox.showinfo("成功", "高功率保持模式已启动")
                
                # 更新按钮状态
                self.keep_mode_btn.config(state=tk.DISABLED)
                
                # 记录高功率保持模式事件
                self.gui_data['optimization_history']['convergence_status_history'].append({
                    'type': 'switch_to_keep_mode_from_gui',
                    'timestamp': datetime.now().isoformat(),
                    'current_mode': self.current_mode,
                    'high_power_params': high_power_params
                })
            else:
                messagebox.showerror("错误", "启动高功率保持模式失败")
                
        except Exception as e:
            self.log(f"启动高功率保持模式失败: {str(e)}")
            messagebox.showerror("错误", f"启动高功率保持模式失败: {str(e)}")

    def switch_to_lock_mode(self):
        """切换到位置锁定模式 - 修改以适配新的优化器结构"""
        self.log("切换到位置锁定模式...")
        self.current_mode = "lock"
        self.status_labels["operation_mode"]["text"] = "位置锁定模式"
        self.status_labels["status"]["text"] = "等待锁定条件"
        
        # 激活优化器中的位置锁定模式
        if self.optimizer and hasattr(self.optimizer, 'activate_lock_mode'):
            self.optimizer.activate_lock_mode()
            
            # 设置锁定回调 - 使用新的回调机制
            if hasattr(self.optimizer, 'lock_callback'):
                self.optimizer.lock_callback = lambda pos, fit: self._handle_position_locked_callback(pos, fit)
        
        # 如果已经有优化结果且收敛，也可以激活锁定模式
        elif self.result and self.result.get('final_convergence', False):
            self.log("基于优化结果激活位置锁定模式")
            # 这里可以设置从结果中获取最佳位置进行锁定
        
        self.log("位置锁定模式已激活，等待满足锁定条件...")
        
        # 更新按钮状态
        self.lock_mode_btn.config(state=tk.DISABLED)
        
        # 记录位置锁定模式激活事件
        self.gui_data['optimization_history']['convergence_status_history'].append({
            'type': 'switch_to_lock_mode',
            'timestamp': datetime.now().isoformat(),
            'current_mode': self.current_mode
        })
    def _handle_position_locked_callback(self, lock_position, lock_fitness):
        """处理位置锁定回调 - 优化器自动触发"""
        try:
            self.log(f"位置锁定条件满足，功率: {lock_fitness:.6f}")
            
            # 切换至开环模式
            if self.hardware_adapter:
                mode = 1  # 开环模式
                modes = self.hardware_adapter.mode_switch(mode)
                self.current_piezo_mode = mode
                self.status_labels["piezo_mode"]["text"] = "开环模式"
                self.piezo_mode_btn["text"] = "切换为闭环模式"
                time.sleep(3)  # 等待模式切换完成
            
            # 更新UI状态
            self.current_mode = "lock"
            self.status_labels["operation_mode"]["text"] = "位置锁定完成"
            self.status_labels["status"]["text"] = "位置已锁定"
            self.status_labels["best_power"]["text"] = self.format_power_value(lock_fitness)
            
            # 更新位置显示
            if lock_position:
                # 解析位置字典，更新A端和B端显示
                for key, value in lock_position.items():
                    if key.startswith('A_'):
                        display_key = key.replace('A_', '')
                        if f"A_{display_key}" in self.position_labels_A:
                            self.position_labels_A[f"A_{display_key}"]["text"] = f"{value:.6f}"
                    elif key.startswith('B_'):
                        display_key = key.replace('B_', '')
                        if f"B_{display_key}" in self.position_labels_B:
                            self.position_labels_B[f"B_{display_key}"]["text"] = f"{value:.6f}"
            
            self.log("位置锁定完成：已保持当前位置并切换至开环模式")
            
            # 记录位置锁定事件
            self.gui_data['optimization_history']['convergence_status_history'].append({
                'type': 'position_locked_auto',
                'timestamp': datetime.now().isoformat(),
                'lock_fitness': lock_fitness,
                'lock_position': lock_position
            })
            
            # 显示消息框
            messagebox.showinfo("成功", "位置锁定完成\n已切换至开环模式")
            
        except Exception as e:
            error_msg = f"位置锁定完成但模式切换失败: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)
    def on_lock_condition_met(self, lock_position, lock_fitness):
        """当锁定条件满足时的回调函数 - 完整保留原功能"""
        try:
            self.log(f"锁定条件满足，功率: {lock_fitness:.6f}")
            
            # 切换至开环模式
            mode = 1  # 开环模式
            modes = self.hardware_adapter.mode_switch(mode)
            self.current_piezo_mode = mode
            self.status_labels["piezo_mode"]["text"] = "开环模式"
            self.piezo_mode_btn["text"] = "切换为闭环模式"
            time.sleep(3)  # 等待模式切换完成
            
            self.log("位置锁定完成：已保持当前位置并切换至开环模式")
            messagebox.showinfo("成功", "位置锁定完成\n已切换至开环模式")
            
        except Exception as e:
            error_msg = f"位置锁定完成但模式切换失败: {str(e)}"
            self.log(error_msg)
            messagebox.showerror("错误", error_msg)


    def _handle_optimization_finished(self, result):
        """处理优化完成回调 - 更新按钮状态"""
        self.result = result
        self.is_running = False
        
        def update_ui():
            try:
                # 停止参数监控
                self.stop_parameter_monitoring()
                
                if result.get('success', False):
                    self.log(f"优化完成，最佳功率: {result.get('best_power', 0):.6f}mW")
                    
                    # 更新结果显示
                    self.update_results_display(result)
                    
                    # 更新状态显示
                    self.status_labels["status"]["text"] = "已完成"
                    self.status_labels["operation_mode"]["text"] = "优化完成"
                    
                    # 检查是否进入高功率保持模式
                    if result.get('high_power_keep_mode', False):
                        self.current_mode = "keep"
                        self.status_labels["operation_mode"]["text"] = "高功率保持模式"
                        self.log("系统已进入高功率保持模式")
                        self.keep_mode_btn.config(state=tk.DISABLED)  # 已进入高功率保持模式，按钮禁用
                    
                    # 检查是否激活位置锁定模式
                    if result.get('lock_mode_activated', False):
                        self.current_mode = "lock"
                        self.status_labels["operation_mode"]["text"] = "位置锁定模式"
                        
                        # 如果已锁定，更新锁定状态
                        if result.get('lock_position_A') is not None:
                            self.status_labels["status"]["text"] = "位置已锁定"
                            self.log(f"位置已锁定，锁定功率: {result.get('lock_fitness', 0):.6f}mW")
                            self.lock_mode_btn.config(state=tk.DISABLED)  # 已锁定，按钮禁用
                    
                    # 记录最终参数
                    self.log("最终算法参数:")
                    self.log(f"  种群大小: {result.get('final_population_size', 'N/A')}")
                    self.log(f"  基因变异率: {result.get('final_gene_mutation_rate', 'N/A')}")
                    self.log(f"  基因交叉率: {result.get('final_gene_crossover_rate', 'N/A')}")
                    self.log(f"  染色体交叉率: {result.get('final_chromosome_crossover_rate', 'N/A')}")
                    
                    # 更新优化历史
                    self._save_optimization_results(result)
                    
                    # 启用相关按钮
                    self.show_charts_btn.config(state=tk.NORMAL)
                    self.save_results_btn.config(state=tk.NORMAL)
                    # 根据是否锁定来设置锁定按钮状态
                    if result.get('lock_mode_activated', False) and result.get('lock_position_A') is None:
                        self.lock_mode_btn.config(state=tk.NORMAL)  # 已激活锁定模式但未锁定，按钮可用
                    else:
                        self.lock_mode_btn.config(state=tk.DISABLED)
                    
                    # 如果收敛了，确保高功率保持模式按钮可用（除非已经进入高功率保持模式）
                    if result.get('final_convergence', False) or self.is_converged:
                        if not result.get('high_power_keep_mode', False):
                            self.keep_mode_btn.config(state=tk.NORMAL)
                            self.log("系统已收敛，可手动切换到高功率保持模式")
                        
                else:
                    error_msg = result.get('error', '未知错误')
                    self.log(f"优化失败: {error_msg}")
                    self.status_labels["status"]["text"] = "失败"
                    messagebox.showerror("优化失败", f"优化过程中出现错误: {error_msg}")
                
                # 恢复UI状态
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.init_device_btn.config(state=tk.NORMAL)
                self.set_initial_pos_btn.config(state=tk.NORMAL)
                self.apply_params_btn.config(state=tk.NORMAL)
                
                # 记录优化结束时间
                self.current_session['end_time'] = datetime.now().isoformat()
                
                # 如果有优秀个体存储库，更新显示
                if hasattr(self, 'optimizer') and self.optimizer:
                    if hasattr(self.optimizer, 'elite_repository_full'):
                        elite_count = len(self.optimizer.elite_repository_full)
                        if "elite_repo_size" in self.elite_labels:
                            self.elite_labels["elite_repo_size"]["text"] = str(elite_count)
                        
                        # 保存到GUI的优秀个体存储库
                        self.elite_repository = self.optimizer.elite_repository_full.copy()
                        self.log(f"优秀个体存储库已更新，包含 {elite_count} 个配对")
                
            except Exception as e:
                self.log(f"更新完成界面时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        self.root.after(0, update_ui)
    def stop_optimization(self):
        """停止优化过程 - 完整保留原功能"""
        if not self.is_running:
            return
        
        self.log("正在停止优化...")
        self.status_labels["status"]["text"] = "停止中"
        
        if self.optimizer:
            self.optimizer.stop()
        
        # 恢复UI状态
        self.is_running = False
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.init_device_btn.config(state=tk.NORMAL)
        self.set_initial_pos_btn.config(state=tk.NORMAL)
        self.status_labels["status"]["text"] = "已停止"
        self.log("优化已停止")

    def detect_convergence(self, fitness=None):
        """检测到收敛时的处理"""
        self.is_converged = True
        
        def update_ui():
            # 使用传入的 fitness 参数更新显示
            if fitness is not None:
                formatted_power = self.format_power_value(fitness)
                self.status_labels["best_power"]["text"] = formatted_power
            self.log("检测到搜索收敛!")
            self.status_labels["status"]["text"] = "已收敛"
            self.status_labels["operation_mode"]["text"] = "收敛阶段"
            self.status_labels["convergence_status"]["text"] = "已收敛"
            
            # 确保变异率和平均功率显示当前值
            if hasattr(self, 'optimizer') and self.optimizer:
                # 更新变异率
                if hasattr(self.optimizer, 'gene_mutation_rate'):
                    self.status_labels["current_gene_mutation_rate"]["text"] = f"{self.optimizer.gene_mutation_rate:.3f}"
                
                # 更新平均功率
                if hasattr(self.optimizer, 'history') and 'avg_fitness' in self.optimizer.history:
                    if self.optimizer.history['avg_fitness']:
                        last_avg = self.optimizer.history['avg_fitness'][-1]
                        formatted_avg = self.format_power_value(last_avg)
                        self.status_labels["avg_power"]["text"] = formatted_avg
            
            # 启用高功率保持模式按钮（优化器将自动进入位置锁定模式，不启用锁定按钮）
            self.keep_mode_btn.config(state=tk.NORMAL)  # 允许切换到高功率保持模式
            
            # 如果是双端模式，更新阶段显示
            if self.optimization_mode.get() == "double":
                self.status_labels["optimization_phase"]["text"] = "收敛阶段"
        
        self.root.after(0, update_ui)
    def _reset_status_display(self):
        """重置状态显示 - 完整保留原功能，更新新参数显示"""
        self.status_labels["eval_count"]["text"] = "0"
        self.status_labels["generation"]["text"] = "0/0"
        self.status_labels["current_power"]["text"] = "0.000000"
        self.status_labels["best_power"]["text"] = "0.000000"
        self.status_labels["avg_power"]["text"] = "0.000000"
        self.status_labels["current_population_size"]["text"] = "0"
        self.status_labels["current_gene_mutation_rate"]["text"] = "0.000"
        self.status_labels["current_gene_crossover_rate"]["text"] = "0.000"
        self.status_labels["current_chromosome_crossover_rate"]["text"] = "0.000"
        self.status_labels["opt_time"]["text"] = "00:00:00"
        self.status_labels["light_status"]["text"] = "未通光"
        self.status_labels["convergence_status"]["text"] = "未收敛"  # 改为 convergence_status
        self.status_labels["lock_status"]["text"] = "未锁定"
        self.status_labels["operation_mode"]["text"] = "搜索模式"  # 重置操作模式显示
        self.progress_var.set(0)
        
        # 重置位置显示
        for key in self.position_labels_A:
            self.position_labels_A[key]["text"] = "0.000000"
        for key in self.position_labels_B:
            self.position_labels_B[key]["text"] = "0.000000"
        
        # 重置优秀个体存储库显示
        if hasattr(self, 'elite_labels'):
            self.elite_labels["elite_repo_size"]["text"] = "0"
            self.elite_labels["best_pair_count"]["text"] = "0"
            self.elite_labels["elite_last_update"]["text"] = "从未"
        
        # 重置图表
        self._reset_charts()
    def _init_optimization_session(self):
        """初始化优化会话 - 完整保留原功能，重置优化阶段和图表数据"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_session = {
            'start_time': None,
            'end_time': None,
            'total_evaluations': 0,
            'total_generations': 0,
            'session_id': session_id
        }
        
        # 重置双端特定状态
        self.current_phase = "both_active"  # 重置优化阶段
        self.light_detected = False
        
        # 清空当前会话的数据记录
        self.gui_data['evaluation_records'] = []
        self.gui_data['generation_records'] = []
        self.gui_data['optimization_history'] = {
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'mutation_rate_history': [],
            'convergence_status_history': [],
        }
        
        # 清空图表数据
        self.chart_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'power_times': [],
            'power_values': []
        }
        
        # 清空双端特定数据
        self.gui_data['dual_end_specific']['elite_repository_A'] = []
        self.gui_data['dual_end_specific']['elite_repository_B'] = []
        self.gui_data['dual_end_specific']['light_detected'] = False
        
        # 重置图表显示
        self._reset_charts()
    def _reset_charts(self):
        """重置图表显示"""
        try:
            # 清空优化过程图表数据
            self.best_fitness_line.set_data([], [])
            self.avg_fitness_line.set_data([], [])
            
            # 清空功率监控图表数据
            self.power_line.set_data([], [])
            
            # 重置坐标轴范围
            self.ax.relim()
            self.ax.autoscale_view()
            self.power_ax.relim()
            self.power_ax.autoscale_view()
            
            # 更新图表标题
            self.ax.set_title('优化过程', fontproperties='SimHei', fontsize=12)
            self.power_ax.set_title('实时功率监控', fontproperties='SimHei', fontsize=12)
            
            # 重绘画布
            self.canvas.draw_idle()
            
            self.log("图表已重置")
            
        except Exception as e:
            self.log(f"重置图表时出错: {str(e)}")
    def _create_serializable_parameters(self, config):
        """创建可序列化的参数 - 完整保留原功能"""
        serializable_config = {}
        for key, value in config.items():
            if isinstance(value, (int, float, str, bool, list, tuple)):
                serializable_config[key] = value
            elif isinstance(value, dict):
                serializable_config[key] = {}
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool, list, tuple)):
                        serializable_config[key][sub_key] = sub_value
                    else:
                        serializable_config[key][sub_key] = str(sub_value)
            else:
                # 对于其他类型，转换为字符串
                serializable_config[key] = str(value)
        return serializable_config

    def _convert_position_to_serializable(self, position_dict):
        """将位置字典转换为可序列化格式 - 完整保留原功能"""
        if position_dict is None:
            return {}
        serializable_position = {}
        for key, value in position_dict.items():
            try:
                serializable_position[key] = float(value)
            except (TypeError, ValueError):
                serializable_position[key] = 0.0
        return serializable_position

    def _update_optimization_charts(self):
        """更新优化过程图表 - 修复数据更新和图表重绘问题"""
        try:
            # 检查是否有代数据
            if not self.gui_data['generation_records']:
                return
                
            # 从代数据中提取数据
            generations = [gen['generation'] for gen in self.gui_data['generation_records']]
            best_fitness = [gen['best_power'] for gen in self.gui_data['generation_records']]
            avg_fitness = [gen.get('avg_fitness', 0) for gen in self.gui_data['generation_records']]
            
            # 保存到图表数据
            self.chart_data['generations'] = generations
            self.chart_data['best_fitness'] = best_fitness
            self.chart_data['avg_fitness'] = avg_fitness
            
            # 更新适应度曲线 - 检查数据是否有效
            if generations and best_fitness:
                # 清空现有曲线
                self.best_fitness_line.set_data([], [])
                self.avg_fitness_line.set_data([], [])
                
                # 设置新数据
                self.best_fitness_line.set_data(generations, best_fitness)
                
                if avg_fitness and len(avg_fitness) == len(generations):
                    self.avg_fitness_line.set_data(generations, avg_fitness)
                
                # 调整坐标轴范围
                self.ax.relim()
                self.ax.autoscale_view()
                
                # 设置图表标题和标签
                self.ax.set_xlabel('代数', fontproperties='SimHei', fontsize=10)
                self.ax.set_ylabel('功率 (mW)', fontproperties='SimHei', fontsize=10)
                self.ax.set_title(f'优化过程 (当前代数: {generations[-1]})', fontproperties='SimHei', fontsize=12)
                self.ax.legend(prop={'family': 'SimHei', 'size': 10})
                self.ax.grid(True, alpha=0.3)
                
                # 重绘画布
                self.canvas.draw_idle()  # 使用draw_idle避免阻塞
            
            # 更新功率监控图表
            self._update_power_chart()
            
        except Exception as e:
            self.log(f"更新图表时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def _save_optimization_results(self, result):
        """保存优化结果到GUI数据 - 完整保留原功能并扩展双端支持"""
        if not result:
            return
        
        mode = self.optimization_mode.get()
        
        # 保存摘要信息
        self.gui_data['optimization_results']['summary'] = {
            'success': result.get('success', False),
            'best_power': float(result.get('best_power', 0)),
            'total_evaluations': result.get('total_evaluations', 0),
            'total_generations': result.get('total_generations', 0),
            'optimization_time': float(result.get('optimization_time', 0)),
            'final_convergence': result.get('final_convergence', False),
            'lock_mode_activated': result.get('lock_mode_activated', False),
            'error_message': result.get('error', '')
        }
        
        # 保存最佳个体信息
        if mode == "single":
            if 'best_position' in result and result['best_position'] is not None:
                self.gui_data['optimization_results']['best_individual'] = {
                    'position': self._convert_position_to_serializable(result['best_position']),
                    'fitness': float(result.get('best_power', 0))
                }
            else:
                self.gui_data['optimization_results']['best_individual'] = {
                    'position': {},
                    'fitness': float(result.get('best_power', 0))
                }
        else:
            # 双端模式保存A端和B端的最佳个体
            if 'best_position_A' in result and result['best_position_A'] is not None:
                self.gui_data['optimization_results']['best_individual_A'] = {
                    'position': self._convert_position_to_serializable(result['best_position_A']),
                    'fitness': float(result.get('best_power_A', 0))
                }
            
            if 'best_position_B' in result and result['best_position_B'] is not None:
                self.gui_data['optimization_results']['best_individual_B'] = {
                    'position': self._convert_position_to_serializable(result['best_position_B']),
                    'fitness': float(result.get('best_power_B', 0))
                }
        
        # 保存收敛信息
        self.gui_data['optimization_results']['convergence_info'] = {
            'is_converged': self.is_converged,
            'current_mode': self.current_mode,
            'convergence_time': self.current_session.get('end_time', ''),
            'total_sessions_evaluations': self.current_session.get('total_evaluations', 0),
            'total_sessions_generations': self.current_session.get('total_generations', 0),
            'final_phase': self.current_phase,
            'light_detected': self.light_detected
        }


    def save_optimization_data(self):
        """保存优化数据 - 完整保留原功能"""
        if not self.gui_data['evaluation_records']:
            messagebox.showinfo("提示", "没有优化数据可保存")
            return
        
        try:
            # 创建保存对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="保存优化数据"
            )
            
            if not file_path:
                return
            
            # 更新元数据
            self.gui_data['metadata']['last_save_time'] = datetime.now().isoformat()
            self.gui_data['metadata']['session_id'] = self.current_session.get('session_id', '')
            self.gui_data['metadata']['total_records'] = {
                'evaluations': len(self.gui_data['evaluation_records']),
                'generations': len(self.gui_data['generation_records']),
                'power_monitoring': len(self.gui_data['power_monitoring_records'])
            }
            
            # 保存数据
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.gui_data, f, indent=2, ensure_ascii=False)
            
            self.log(f"优化数据已保存到: {file_path}")
            messagebox.showinfo("成功", 
                f"优化数据保存成功\n"
                f"评估次数: {len(self.gui_data['evaluation_records'])}\n"
                f"代数: {len(self.gui_data['generation_records'])}\n"
                f"功率记录: {len(self.gui_data['power_monitoring_records'])}")
            
        except Exception as e:
            self.log(f"保存优化数据失败: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"详细错误信息: {error_details}")
            messagebox.showerror("错误", f"保存优化数据失败: {str(e)}")

    def reset_parameters(self):
        """重置参数为默认值 - 完整保留原功能并扩展双端支持"""
        if self.is_running:
            messagebox.showinfo("提示", "优化正在进行中，无法重置参数")
            return
        
        mode = self.optimization_mode.get()
        
        if mode == "single":
            # 清空现有输入
            for key in self.single_param_entries:
                self.single_param_entries[key][0].delete(0, tk.END)
            
            # 获取默认配置
            default_config = get_optimized_config()
            
            # 填充默认参数
            self._fill_single_default_parameters(default_config)
            
            # 优化参数选择（默认全选）
            self.select_all_params()
            
        else:
            # 清空双端参数输入
            for key in self.double_param_entries:
                self.double_param_entries[key][0].delete(0, tk.END)
            
            # 获取双端默认配置
            default_config = get_dual_end_config()
            
            # 填充双端默认参数
            self._fill_double_default_parameters(default_config)
            
            # 双端优化参数选择（默认全选）
            self.optimize_A_x.set(True)
            self.optimize_A_y.set(True)
            self.optimize_A_z.set(True)
            self.optimize_A_rx.set(True)
            self.optimize_A_ry.set(True)
            self.optimize_B_x.set(True)
            self.optimize_B_y.set(True)
            self.optimize_B_z.set(True)
            self.optimize_B_rx.set(True)
            self.optimize_B_ry.set(True)
        
        self.log("参数已重置为默认值")

    def _fill_single_default_parameters(self, default_config):
        """填充单端默认参数"""
        # 基本参数
        self.single_param_entries['population_size'][0].insert(0, str(default_config['population_size']))
        self.single_param_entries['generations'][0].insert(0, str(default_config['generations']))
        self.single_param_entries['mutation_rate'][0].insert(0, str(default_config['mutation_rate']))
        self.single_param_entries['crossover_rate'][0].insert(0, str(default_config['crossover_rate']))
        self.single_param_entries['elite_size'][0].insert(0, str(default_config['elite_size']))
        self.single_param_entries['tournament_size'][0].insert(0, str(default_config['tournament_size']))
        self.single_param_entries['convergence_threshold'][0].insert(0, str(default_config['convergence_threshold'] * 100))
        self.single_param_entries['convergence_patience'][0].insert(0, str(default_config['convergence_patience']))
        self.single_param_entries['enhanced_exploration_max'][0].insert(0, str(default_config['enhanced_exploration_max']))
        self.single_param_entries['enhanced_mutation_rate'][0].insert(0, str(default_config['enhanced_mutation_rate']))
        self.single_param_entries['converged_mutation_rate'][0].insert(0, str(default_config['converged_mutation_rate']))
        self.single_param_entries['local_search_rate'][0].insert(0, str(default_config['local_search_rate']))
        self.single_param_entries['fitness_variance_threshold'][0].insert(0, str(default_config['fitness_variance_threshold']))
        
        # 监控参数（转换为百分比）
        self.single_param_entries['monitoring_threshold_1'][0].insert(0, str(default_config['monitoring_threshold_1'] * 100))
        self.single_param_entries['monitoring_threshold_2'][0].insert(0, str(default_config['monitoring_threshold_2'] * 100))
        self.single_param_entries['alert_threshold_percent'][0].insert(0, str(default_config['alert_threshold_percent'] * 100))
        self.single_param_entries['monitoring_interval'][0].insert(0, str(default_config['monitoring_interval']))
        
        # 初始位置（使用搜索范围的中间值）
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range'][key]
            initial_val = (min_val + max_val) / 2
            self.single_param_entries[f"{key}_initial"][0].insert(0, str(initial_val))
        
        # 搜索范围
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range'][key]
            self.single_param_entries[f"{key}_min"][0].insert(0, str(min_val))
            self.single_param_entries[f"{key}_max"][0].insert(0, str(max_val))
        
        # 自适应参数
        self.adaptive_mutation.set(default_config['adaptive_mutation_rate'])
        self.adaptive_crossover.set(default_config['adaptive_crossover_rate'])
        self.elite_protection.set(default_config['elite_protection'])

    def _fill_double_default_parameters(self, default_config):
        """填充双端默认参数 - 更新为新参数"""
        # 遗传算法参数
        self.double_param_entries['population_size'][0].insert(0, str(default_config['population_size']))
        self.double_param_entries['generations'][0].insert(0, str(default_config['generations']))
        self.double_param_entries['gene_mutation_rate'][0].insert(0, str(default_config['gene_mutation_rate']))
        self.double_param_entries['gene_crossover_rate'][0].insert(0, str(default_config['gene_crossover_rate']))
        self.double_param_entries['chromosome_crossover_rate'][0].insert(0, str(default_config['chromosome_crossover_rate']))
        self.double_param_entries['elite_size'][0].insert(0, str(default_config['elite_size']))
        self.double_param_entries['tournament_size'][0].insert(0, str(default_config['tournament_size']))
        self.double_param_entries['convergence_threshold'][0].insert(0, str(default_config['convergence_threshold'] * 100))
        self.double_param_entries['convergence_patience'][0].insert(0, str(default_config['convergence_patience']))
        self.double_param_entries['enhanced_exploration_max'][0].insert(0, str(default_config['enhanced_exploration_max']))
        self.double_param_entries['enhanced_mutation_rate'][0].insert(0, str(default_config['enhanced_mutation_rate']))
        self.double_param_entries['converged_mutation_rate'][0].insert(0, str(default_config['converged_mutation_rate']))
        self.double_param_entries['local_search_rate'][0].insert(0, str(default_config['local_search_rate']))
        self.double_param_entries['fitness_variance_threshold'][0].insert(0, str(default_config['fitness_variance_threshold']))
        self.double_param_entries['alert_threshold_percent'][0].insert(0, str(default_config['alert_threshold_percent'] * 100))
        
        # 高功率保持模式参数
        self.double_param_entries['high_power_population_size'][0].insert(0, str(default_config['high_power_population_size']))
        self.double_param_entries['high_power_mutation_rate'][0].insert(0, str(default_config['high_power_mutation_rate']))
        self.double_param_entries['high_power_crossover_rate'][0].insert(0, str(default_config['high_power_crossover_rate']))
        
        # 双端特定参数
        self.double_param_entries['light_threshold'][0].insert(0, str(default_config['light_threshold']))
        self.double_param_entries['lock_mode_threshold'][0].insert(0, str(default_config['lock_mode_threshold'] * 100))
        self.double_param_entries['elite_clone_rate'][0].insert(0, str(default_config['elite_clone_rate']))
        
        # A端初始位置
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range_A'][key]
            initial_val = (min_val + max_val) / 2
            self.double_param_entries[f"A_{key}_initial"][0].insert(0, str(initial_val))
        
        # A端搜索范围
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range_A'][key]
            self.double_param_entries[f"A_{key}_min"][0].insert(0, str(min_val))
            self.double_param_entries[f"A_{key}_max"][0].insert(0, str(max_val))
        
        # B端初始位置
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range_B'][key]
            initial_val = (min_val + max_val) / 2
            self.double_param_entries[f"B_{key}_initial"][0].insert(0, str(initial_val))
        
        # B端搜索范围
        for key in ['x', 'y', 'z', 'rx', 'ry']:
            min_val, max_val = default_config['search_range_B'][key]
            self.double_param_entries[f"B_{key}_min"][0].insert(0, str(min_val))
            self.double_param_entries[f"B_{key}_max"][0].insert(0, str(max_val))

    def save_parameters(self):
        """保存参数配置 - 修复：确保所有参数完整保存"""
        try:
            mode = self.optimization_mode.get()
            
            # 获取当前参数配置
            params = {}
            if mode == "single":
                # 获取单端参数
                params = self.get_parameters()
                
                # 添加额外信息用于识别
                params['optimization_mode'] = 'single'
                
                # 确保保存所有必要的参数
                params['optimize_x'] = self.optimize_x.get()
                params['optimize_y'] = self.optimize_y.get()
                params['optimize_z'] = self.optimize_z.get()
                params['optimize_rx'] = self.optimize_rx.get()
                params['optimize_ry'] = self.optimize_ry.get()
                
                # 保存初始位置
                initial_pos = {}
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.single_param_entries[f'{param}_initial'][0].get())
                        initial_pos[f'{param}_initial'] = value
                    except:
                        initial_pos[f'{param}_initial'] = 15.0  # 默认值
                params['initial_positions'] = initial_pos
                
                # 保存自适应参数
                params['adaptive_mutation_rate'] = self.adaptive_mutation.get()
                params['adaptive_crossover_rate'] = self.adaptive_crossover.get()
                params['elite_protection'] = self.elite_protection.get()
                
            else:
                # 获取双端参数
                params = self.get_parameters()
                
                # 添加额外信息用于识别
                params['optimization_mode'] = 'double'
                
                # 确保保存所有必要的参数
                params['optimize_A_x'] = self.optimize_A_x.get()
                params['optimize_A_y'] = self.optimize_A_y.get()
                params['optimize_A_z'] = self.optimize_A_z.get()
                params['optimize_A_rx'] = self.optimize_A_rx.get()
                params['optimize_A_ry'] = self.optimize_A_ry.get()
                params['optimize_B_x'] = self.optimize_B_x.get()
                params['optimize_B_y'] = self.optimize_B_y.get()
                params['optimize_B_z'] = self.optimize_B_z.get()
                params['optimize_B_rx'] = self.optimize_B_rx.get()
                params['optimize_B_ry'] = self.optimize_B_ry.get()
                
                # 保存A端初始位置
                initial_pos_A = {}
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.double_param_entries[f'A_{param}_initial'][0].get())
                        initial_pos_A[f'A_{param}_initial'] = value
                    except:
                        initial_pos_A[f'A_{param}_initial'] = 15.0  # 默认值
                params['initial_positions_A'] = initial_pos_A
                
                # 保存B端初始位置
                initial_pos_B = {}
                for param in ['x', 'y', 'z', 'rx', 'ry']:
                    try:
                        value = float(self.double_param_entries[f'B_{param}_initial'][0].get())
                        initial_pos_B[f'B_{param}_initial'] = value
                    except:
                        initial_pos_B[f'B_{param}_initial'] = 15.0  # 默认值
                params['initial_positions_B'] = initial_pos_B
                
                # 保存自适应参数
                params['adaptive_mutation_rate'] = self.adaptive_mutation.get()
                params['adaptive_crossover_rate'] = self.adaptive_crossover.get()
                params['elite_protection'] = self.elite_protection.get()
            
            # 添加保存时间戳和版本信息
            params['saved_time'] = datetime.now().isoformat()
            params['gui_version'] = '2.0'
            
            # 创建保存对话框
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # 保存参数
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2, ensure_ascii=False)
            
            self.log(f"参数已保存到: {file_path}")
            messagebox.showinfo("成功", "参数保存成功")
            
            # 调试：打印保存的参数结构
            self.log(f"保存的参数键: {list(params.keys())}")
            if mode == "single":
                self.log(f"单端模式参数数量: {len(params)}")
            else:
                self.log(f"双端模式参数数量: {len(params)}")
                
        except Exception as e:
            self.log(f"保存参数失败: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"详细错误信息: {error_details}")
            messagebox.showerror("错误", f"保存参数失败: {str(e)}")
    def load_parameters(self):
        """加载参数配置 - 修复：改进模式判断和错误处理"""
        if self.is_running:
            messagebox.showinfo("提示", "优化正在进行中，无法加载参数")
            return
        
        try:
            # 创建打开对话框
            file_path = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
            
            # 加载参数
            with open(file_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
            
            self.log(f"正在加载参数文件: {file_path}")
            
            # 更可靠的模式判断
            is_double_mode = False
            
            # 方法1：检查优化模式字段
            if 'optimization_mode' in params:
                is_double_mode = (params['optimization_mode'] == 'double')
                self.log(f"通过optimization_mode判断: {'双端' if is_double_mode else '单端'}模式")
            # 方法2：检查特定双端参数
            elif 'search_range_A' in params or 'optimize_A_x' in params:
                is_double_mode = True
                self.log(f"通过双端参数判断: 双端模式")
            # 方法3：检查是否有A/B端初始位置
            elif 'initial_positions_A' in params and 'initial_positions_B' in params:
                is_double_mode = True
                self.log(f"通过初始位置判断: 双端模式")
            else:
                # 默认为单端模式
                is_double_mode = False
                self.log(f"默认为单端模式")
            
            # 设置优化模式并切换界面
            if is_double_mode:
                self.optimization_mode.set("double")
                self.on_optimization_mode_changed()
                
                # 清空现有输入
                for key in self.double_param_entries:
                    self.double_param_entries[key][0].delete(0, tk.END)
                
                # 填充双端参数
                self._load_double_parameters(params)
                
            else:
                self.optimization_mode.set("single")
                self.on_optimization_mode_changed()
                
                # 清空现有输入
                for key in self.single_param_entries:
                    self.single_param_entries[key][0].delete(0, tk.END)
                
                # 填充单端参数
                self._load_single_parameters(params)
            
            # 记录加载的参数数量
            self.log(f"已加载参数文件: {file_path}，参数数量: {len(params)}")
            
            # 显示保存时间（如果存在）
            if 'saved_time' in params:
                try:
                    saved_time = datetime.fromisoformat(params['saved_time'])
                    self.log(f"参数保存时间: {saved_time.strftime('%Y-%m-%d %H:%M:%S')}")
                except:
                    pass
            
            messagebox.showinfo("成功", "参数加载成功")
            
        except json.JSONDecodeError as e:
            self.log(f"JSON解析错误: {str(e)}")
            messagebox.showerror("错误", f"参数文件格式错误: {str(e)}")
        except Exception as e:
            self.log(f"加载参数失败: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"详细错误信息: {error_details}")
            messagebox.showerror("错误", f"加载参数失败: {str(e)}")

    def _load_single_parameters(self, params):
        """加载单端参数 - 改进：更好的错误处理"""
        try:
            # 基本参数
            basic_params = [
                ('population_size', 'population_size', 30),
                ('generations', 'generations', 200),
                ('mutation_rate', 'mutation_rate', 0.65),
                ('crossover_rate', 'crossover_rate', 0.7),
                ('elite_size', 'elite_size', 8),
                ('tournament_size', 'tournament_size', 10),
                ('convergence_threshold', 'convergence_threshold', 5.0),
                ('convergence_patience', 'convergence_patience', 8),
                ('enhanced_exploration_max', 'enhanced_exploration_max', 3),
                ('enhanced_mutation_rate', 'enhanced_mutation_rate', 0.7),
                ('converged_mutation_rate', 'converged_mutation_rate', 0.02),
                ('local_search_rate', 'local_search_rate', 0.4),
                ('fitness_variance_threshold', 'fitness_variance_threshold', 0.005),
                ('alert_threshold_percent', 'alert_threshold_percent', 5.0),
                ('monitoring_threshold_1', 'monitoring_threshold_1', 1.0),
                ('monitoring_threshold_2', 'monitoring_threshold_2', 5.0),
                ('monitoring_interval', 'monitoring_interval', 1.0)
            ]
            
            for gui_key, param_key, default in basic_params:
                if param_key in params:
                    value = params[param_key]
                    # 处理百分比参数
                    if param_key in ['convergence_threshold', 'alert_threshold_percent', 
                                'monitoring_threshold_1', 'monitoring_threshold_2']:
                        value = value * 100  # 转换为百分比显示
                    
                    if gui_key in self.single_param_entries:
                        self.single_param_entries[gui_key][0].delete(0, tk.END)
                        self.single_param_entries[gui_key][0].insert(0, str(value))
                else:
                    # 使用默认值
                    if gui_key in self.single_param_entries:
                        self.single_param_entries[gui_key][0].delete(0, tk.END)
                        self.single_param_entries[gui_key][0].insert(0, str(default))
            
            # 搜索范围
            search_range = params.get('search_range', {
                'x': (0, 30), 'y': (0, 30), 'z': (0, 30), 'rx': (0.0, 0.03), 'ry': (0.0, 0.03)
            })
            
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                min_val, max_val = search_range[key]
                
                # 最小值
                min_entry_key = f"{key}_min"
                if min_entry_key in self.single_param_entries:
                    self.single_param_entries[min_entry_key][0].delete(0, tk.END)
                    self.single_param_entries[min_entry_key][0].insert(0, str(min_val))
                
                # 最大值
                max_entry_key = f"{key}_max"
                if max_entry_key in self.single_param_entries:
                    self.single_param_entries[max_entry_key][0].delete(0, tk.END)
                    self.single_param_entries[max_entry_key][0].insert(0, str(max_val))
            
            # 初始位置
            initial_pos = params.get('initial_positions', {})
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                entry_key = f"{key}_initial"
                if entry_key in self.single_param_entries:
                    self.single_param_entries[entry_key][0].delete(0, tk.END)
                    
                    # 优先使用保存的初始位置
                    if f'{key}_initial' in initial_pos:
                        value = initial_pos[f'{key}_initial']
                    elif key in initial_pos:
                        value = initial_pos[key]
                    else:
                        # 使用搜索范围的中间值
                        min_val, max_val = search_range[key]
                        value = (min_val + max_val) / 2
                    
                    self.single_param_entries[entry_key][0].insert(0, str(value))
            
            # 优化参数选择
            if 'optimize_x' in params:
                self.optimize_x.set(params['optimize_x'])
                self.optimize_y.set(params['optimize_y'])
                self.optimize_z.set(params['optimize_z'])
                self.optimize_rx.set(params['optimize_rx'])
                self.optimize_ry.set(params['optimize_ry'])
            else:
                # 如果没有保存优化参数选择，默认全选
                self.select_all_params()
            
            # 自适应参数
            if 'adaptive_mutation_rate' in params:
                self.adaptive_mutation.set(params['adaptive_mutation_rate'])
            if 'adaptive_crossover_rate' in params:
                self.adaptive_crossover.set(params['adaptive_crossover_rate'])
            if 'elite_protection' in params:
                self.elite_protection.set(params['elite_protection'])
            
            self.log(f"单端参数加载完成")
                
        except Exception as e:
            self.log(f"加载单端参数时出错: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"详细错误信息: {error_details}")
            raise

    def _load_double_parameters(self, params):
        """加载双端参数 - 改进：更好的错误处理和参数兼容性，添加新参数"""
        try:
            # 遗传算法参数
            genetic_params = [
                ('population_size', 'population_size', 30),
                ('generations', 'generations', 200),
                ('gene_mutation_rate', 'gene_mutation_rate', 0.15),
                ('gene_crossover_rate', 'gene_crossover_rate', 0.8),
                ('chromosome_crossover_rate', 'chromosome_crossover_rate', 0.2),
                ('elite_size', 'elite_size', 8),
                ('tournament_size', 'tournament_size', 10),
                ('convergence_threshold', 'convergence_threshold', 5.0),
                ('convergence_patience', 'convergence_patience', 8),
                ('enhanced_exploration_max', 'enhanced_exploration_max', 3),
                ('enhanced_mutation_rate', 'enhanced_mutation_rate', 0.7),
                ('converged_mutation_rate', 'converged_mutation_rate', 0.02),
                ('local_search_rate', 'local_search_rate', 0.4),
                ('fitness_variance_threshold', 'fitness_variance_threshold', 0.005),
                ('alert_threshold_percent', 'alert_threshold_percent', 5.0)
            ]
            
            # 高功率保持模式参数
            high_power_params = [
                ('high_power_population_size', 'high_power_population_size', 20),
                ('high_power_mutation_rate', 'high_power_mutation_rate', 0.05),
                ('high_power_crossover_rate', 'high_power_crossover_rate', 0.3)
            ]
            
            # 双端特定参数
            dual_params = [
                ('light_threshold', 'light_threshold', 0.0002),
                ('lock_mode_threshold', 'lock_mode_threshold', 0.1),
                ('elite_clone_rate', 'elite_clone_rate', 0.25)
            ]
            
            # 加载所有参数
            all_params = genetic_params + high_power_params + dual_params
            
            for gui_key, param_key, default in all_params:
                if param_key in params:
                    value = params[param_key]
                    # 处理百分比参数
                    if param_key in ['convergence_threshold', 'alert_threshold_percent', 'lock_mode_threshold']:
                        value = value * 100  # 转换为百分比显示
                    
                    if gui_key in self.double_param_entries:
                        self.double_param_entries[gui_key][0].delete(0, tk.END)
                        self.double_param_entries[gui_key][0].insert(0, str(value))
                else:
                    # 使用默认值
                    if gui_key in self.double_param_entries:
                        self.double_param_entries[gui_key][0].delete(0, tk.END)
                        self.double_param_entries[gui_key][0].insert(0, str(default))
            
            # A端搜索范围
            search_range_A = params.get('search_range_A', {
                'x': (0, 30), 'y': (0, 30), 'z': (0, 30), 'rx': (0.0, 0.03), 'ry': (0.0, 0.03)
            })
            
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                min_val, max_val = search_range_A[key]
                
                # 最小值
                min_entry_key = f"A_{key}_min"
                if min_entry_key in self.double_param_entries:
                    self.double_param_entries[min_entry_key][0].delete(0, tk.END)
                    self.double_param_entries[min_entry_key][0].insert(0, str(min_val))
                
                # 最大值
                max_entry_key = f"A_{key}_max"
                if max_entry_key in self.double_param_entries:
                    self.double_param_entries[max_entry_key][0].delete(0, tk.END)
                    self.double_param_entries[max_entry_key][0].insert(0, str(max_val))
            
            # B端搜索范围
            search_range_B = params.get('search_range_B', {
                'x': (0, 30), 'y': (0, 30), 'z': (0, 30), 'rx': (0.0, 0.03), 'ry': (0.0, 0.03)
            })
            
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                min_val, max_val = search_range_B[key]
                
                # 最小值
                min_entry_key = f"B_{key}_min"
                if min_entry_key in self.double_param_entries:
                    self.double_param_entries[min_entry_key][0].delete(0, tk.END)
                    self.double_param_entries[min_entry_key][0].insert(0, str(min_val))
                
                # 最大值
                max_entry_key = f"B_{key}_max"
                if max_entry_key in self.double_param_entries:
                    self.double_param_entries[max_entry_key][0].delete(0, tk.END)
                    self.double_param_entries[max_entry_key][0].insert(0, str(max_val))
            
            # A端初始位置
            initial_pos_A = params.get('initial_positions_A', {})
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                entry_key = f"A_{key}_initial"
                if entry_key in self.double_param_entries:
                    self.double_param_entries[entry_key][0].delete(0, tk.END)
                    
                    # 优先使用保存的初始位置
                    if f'A_{key}_initial' in initial_pos_A:
                        value = initial_pos_A[f'A_{key}_initial']
                    elif f'A_{key}' in initial_pos_A:
                        value = initial_pos_A[f'A_{key}']
                    else:
                        # 使用搜索范围的中间值
                        min_val, max_val = search_range_A[key]
                        value = (min_val + max_val) / 2
                    
                    self.double_param_entries[entry_key][0].insert(0, str(value))
            
            # B端初始位置
            initial_pos_B = params.get('initial_positions_B', {})
            for key in ['x', 'y', 'z', 'rx', 'ry']:
                entry_key = f"B_{key}_initial"
                if entry_key in self.double_param_entries:
                    self.double_param_entries[entry_key][0].delete(0, tk.END)
                    
                    # 优先使用保存的初始位置
                    if f'B_{key}_initial' in initial_pos_B:
                        value = initial_pos_B[f'B_{key}_initial']
                    elif f'B_{key}' in initial_pos_B:
                        value = initial_pos_B[f'B_{key}']
                    else:
                        # 使用搜索范围的中间值
                        min_val, max_val = search_range_B[key]
                        value = (min_val + max_val) / 2
                    
                    self.double_param_entries[entry_key][0].insert(0, str(value))
            
            # 优化参数选择
            if 'optimize_A_x' in params:
                self.optimize_A_x.set(params['optimize_A_x'])
                self.optimize_A_y.set(params['optimize_A_y'])
                self.optimize_A_z.set(params['optimize_A_z'])
                self.optimize_A_rx.set(params['optimize_A_rx'])
                self.optimize_A_ry.set(params['optimize_A_ry'])
                self.optimize_B_x.set(params['optimize_B_x'])
                self.optimize_B_y.set(params['optimize_B_y'])
                self.optimize_B_z.set(params['optimize_B_z'])
                self.optimize_B_rx.set(params['optimize_B_rx'])
                self.optimize_B_ry.set(params['optimize_B_ry'])
            else:
                # 如果没有保存优化参数选择，默认全选
                self.optimize_A_x.set(True)
                self.optimize_A_y.set(True)
                self.optimize_A_z.set(True)
                self.optimize_A_rx.set(True)
                self.optimize_A_ry.set(True)
                self.optimize_B_x.set(True)
                self.optimize_B_y.set(True)
                self.optimize_B_z.set(True)
                self.optimize_B_rx.set(True)
                self.optimize_B_ry.set(True)
            
            # 自适应参数
            if 'adaptive_mutation_rate' in params:
                self.adaptive_mutation.set(params['adaptive_mutation_rate'])
            if 'adaptive_crossover_rate' in params:
                self.adaptive_crossover.set(params['adaptive_crossover_rate'])
            if 'elite_protection' in params:
                self.elite_protection.set(params['elite_protection'])
            
            self.log(f"双端参数加载完成")
                
        except Exception as e:
            self.log(f"加载双端参数时出错: {str(e)}")
            import traceback
            error_details = traceback.format_exc()
            self.log(f"详细错误信息: {error_details}")
            raise
    def clear_log(self):
        """清空日志 - 完整保留原功能"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log("日志已清空")

    def save_log(self):
        """保存日志 - 完整保留原功能"""
        log_content = self.log_text.get(1.0, tk.END)
        
        if not log_content.strip():
            messagebox.showinfo("提示", "日志为空，无需保存")
            return
        
        # 创建保存对话框
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(log_content)
            
            self.log(f"日志已保存到: {file_path}")
            messagebox.showinfo("成功", "日志保存成功")
        except Exception as e:
            self.log(f"保存日志失败: {str(e)}")
            messagebox.showerror("错误", f"保存日志失败: {str(e)}")

    def on_tab_changed(self, event):
        """标签页切换事件 - 完整保留原功能"""
        current_tab = self.notebook.select()
        
        # 如果切换到结果标签页且有结果数据，更新图表
        if current_tab == str(self.results_frame) and self.result and self.result['success']:
            pass  # 可以在这里添加结果图表的更新逻辑

    def cleanup(self):
        """清理资源，断开设备连接，停止参数监控"""
        try:
            self.log("正在清理资源...")
            
            # 停止功率监控
            self.stop_power_monitoring()
            
            # 停止参数监控
            self.stop_parameter_monitoring()
            
            # 停止优化（如果正在运行）
            if self.is_running and self.optimizer:
                try:
                    self.optimizer.stop()
                    self.log("已停止优化进程")
                except Exception as e:
                    self.log(f"停止优化进程时出错: {str(e)}")
            
            # 保存当前参数快照
            try:
                self.save_current_parameter_snapshot()
            except Exception as e:
                self.log(f"保存参数快照失败: {str(e)}")
            
            # 保存参数预设
            try:
                self.save_parameter_presets()
            except Exception as e:
                self.log(f"保存参数预设失败: {str(e)}")
            
            # 断开设备连接
            if hasattr(self, 'device_manager'):
                try:
                    self.device_manager.disconnect_all()
                    self.log("已断开所有设备连接")
                except Exception as e:
                    self.log(f"断开设备连接时出错: {str(e)}")
            
            # 清理硬件适配器
            if self.hardware_adapter:
                try:
                    self.hardware_adapter.disconnect()
                    self.hardware_adapter = None
                    self.log("已清理硬件适配器")
                except Exception as e:
                    self.log(f"清理硬件适配器时出错: {str(e)}")
            
            # 保存未保存的数据
            if (self.gui_data['evaluation_records'] or 
                self.gui_data['power_monitoring_records']):
                self.log("检测到未保存的数据，建议手动保存")
            
            self.log("资源清理完成")
            
        except Exception as e:
            self.log(f"清理资源时出错: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_closing(self):
        """窗口关闭事件处理 - 完整保留原功能"""
        if messagebox.askokcancel("退出", "确定要退出程序吗？\n这将断开所有设备连接。"):
            self.cleanup()
            self.root.destroy()

    # 双端特定方法

    def _get_phase_display_name(self, phase):
        """获取阶段显示名称"""
        phase_names = {
            'both_active': '双端激活',
            'both_fixed': '高功率保持',
            'search': '搜索模式',
            'keep': '高功率保持模式',
            'lock': '位置锁定模式',
            'local': '局部优化模式'
        }
        return phase_names.get(phase, phase)

    def _handle_optimizer_callback(self, data):
        """处理优化器回调数据 - 添加高功率状态参数更新"""
        def process():
            try:
                # 检查GUI窗口是否还存在
                if not (hasattr(self, 'root') and self.root and 
                        hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
                    return
                        
                data_type = data.get('type')
                
                if data_type == 'evaluation' and 'evaluation_data' in data:
                    # 处理评估数据
                    self._process_evaluation_data(data['evaluation_data'])
                    
                elif data_type == 'generation' and 'generation_data' in data:
                    # 处理代数据
                    self._process_generation_data(data['generation_data'])
                    
                    # 更新UI
                    self._update_generation_ui(data['generation_data'])
                    
                    # 更新参数显示
                    self.update_parameter_display()
                    
                    # 强制更新图表
                    self._update_optimization_charts()
                    
                    # 新增：更新高功率保持模式状态
                    if 'high_power_status' in data['generation_data']:
                        self._update_high_power_status_display(data['generation_data']['high_power_status'])
                    
                elif data_type == 'parameters_updated' and 'updated_parameters' in data:
                    # 处理参数更新通知
                    updated_params = data.get('updated_parameters', {})
                    update_count = data.get('update_count', 0)
                    
                    self.log(f"收到优化器参数更新通知: {update_count} 个参数已更新")
                    for key, value in updated_params.items():
                        self.log(f"  {key}: {value}")
                    
                    # 更新当前参数存储
                    self.current_parameters.update(updated_params)
                    
                    # 更新UI显示
                    self.update_parameter_display()
                    
                elif data_type == 'parameter_request':
                    # 参数请求日志
                    self.log("优化器请求参数")
                    
                elif data_type == 'global_convergence_detected':
                    # 新增：处理全局收敛通知
                    self.log("收到全局收敛通知，优化器将自动进入位置锁定模式")
                    self.status_labels["convergence_status"]["text"] = "全局收敛"
                    self.status_labels["operation_mode"]["text"] = "位置锁定模式"
                    self.lock_mode_btn.config(state=tk.DISABLED)  # 锁定模式已激活，按钮禁用
                    
                elif data_type == 'position_locked':
                    # 处理位置锁定通知
                    self._handle_position_locked_callback(data.get('lock_position'), data.get('lock_fitness'))
                
                # 每次回调都尝试更新图表
                if hasattr(self, '_update_optimization_charts'):
                    self.root.after(100, self._update_optimization_charts)
                
            except Exception as e:
                self.log(f"处理优化器回调数据时出错: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # 安全检查：只在GUI窗口存在时执行
        if (hasattr(self, 'root') and self.root and 
            hasattr(self.root, 'winfo_exists') and self.root.winfo_exists()):
            self.root.after(0, process)
    def update_parameter_display(self, params=None):
        """
        更新参数显示
        """
        if params is None:
            params = self.current_parameters
        
        if not params:
            return
        
        def update_ui():
            try:
                # # 更新关键参数显示
                # if 'param_population_size' in self.param_labels:
                #     self.param_labels["param_population_size"]["text"] = str(params.get('population_size', 0))
                
                # if 'param_mutation_rate' in self.param_labels:
                #     mutation_rate = params.get('gene_mutation_rate', params.get('mutation_rate', 0))
                #     self.param_labels["param_mutation_rate"]["text"] = f"{mutation_rate:.3f}"
                
                # if 'param_crossover_rate' in self.param_labels:
                #     crossover_rate = params.get('gene_crossover_rate', params.get('crossover_rate', 0))
                #     self.param_labels["param_crossover_rate"]["text"] = f"{crossover_rate:.3f}"
                
                # if 'param_elite_size' in self.param_labels:
                #     self.param_labels["param_elite_size"]["text"] = str(params.get('elite_size', 0))
                
                # if 'param_convergence_threshold' in self.param_labels:
                #     threshold = params.get('convergence_threshold', 0) * 100
                #     self.param_labels["param_convergence_threshold"]["text"] = f"{threshold:.1f}%"
                
                # 更新状态栏的参数信息
                if hasattr(self, 'status_labels') and 'current_gene_mutation_rate' in self.status_labels:
                    mutation_rate = params.get('gene_mutation_rate', params.get('mutation_rate', 0))
                    self.status_labels["current_gene_mutation_rate"]["text"] = f"{mutation_rate:.3f}"
                
                if hasattr(self, 'status_labels') and 'current_gene_crossover_rate' in self.status_labels:
                    crossover_rate = params.get('gene_crossover_rate', params.get('crossover_rate', 0))
                    self.status_labels["current_gene_crossover_rate"]["text"] = f"{crossover_rate:.3f}"
                
                if hasattr(self, 'status_labels') and 'current_chromosome_crossover_rate' in self.status_labels:
                    chromosome_crossover_rate = params.get('chromosome_crossover_rate', 0)
                    self.status_labels["current_chromosome_crossover_rate"]["text"] = f"{chromosome_crossover_rate:.3f}"
                
                # 更新高功率保持模式参数
                if hasattr(self, 'status_labels') and 'current_population_size' in self.status_labels:
                    population_size = params.get('population_size', 0)
                    self.status_labels["current_population_size"]["text"] = str(population_size)
                
                # 更新锁定阈值显示
                if hasattr(self, 'status_labels') and 'lock_status' in self.status_labels:
                    lock_threshold = params.get('lock_mode_threshold', 0) * 100
                    # 可以在这里添加锁定阈值的显示
                
                # 更新局部搜索范围显示
                if hasattr(self, 'status_labels'):
                    local_search_range = params.get('local_search_range_percent', params.get('local_search_rate', 0)) * 100
                    # 可以在这里添加局部搜索范围的显示
                
            except Exception as e:
                self.log(f"更新参数显示失败: {str(e)}")
        
        # 在主线程中更新UI
        if self.root:
            self.root.after(0, update_ui)
    def _process_evaluation_data(self, eval_data):
        """处理评估数据 - 扩展支持双端，修复current_phase引用"""
        # 修复：检查position是否为None
        position_data = eval_data.get('position')
        if position_data is None:
            position_data = {}
            
        # 创建评估记录
        eval_record = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_count': eval_data.get('evaluation_count', 0),
            'generation': self.current_generation,
            'power': float(eval_data.get('power', 0)),
            'position': self._convert_position_to_serializable(position_data),
            'individual': [float(x) for x in eval_data.get('individual', [])],
            'is_best': False,
            'optimization_mode': self.optimization_mode.get(),
            'optimization_phase': self.current_phase  # 使用当前优化阶段
        }
        
        # 双端模式特定数据
        if self.optimization_mode.get() == "double":
            eval_record['light_detected'] = self.light_detected
        
        # 添加到GUI数据记录
        self.gui_data['evaluation_records'].append(eval_record)
        
        # 更新评估计数显示
        if "eval_count" in self.status_labels:
            self.status_labels["eval_count"]["text"] = str(eval_data.get('evaluation_count', 0))
        
        # 更新会话统计
        self.current_session['total_evaluations'] = len(self.gui_data['evaluation_records'])

    def _process_generation_data(self, gen_data):
        """处理代数据 - 扩展支持双端，确保新参数更新"""
        current_generation = gen_data.get('iteration', 0)
        self.current_generation = current_generation
        
        # 更新优化阶段（如果回调数据中有）
        if 'optimization_phase' in gen_data:
            self.current_phase = gen_data['optimization_phase']
        
        # 修复：确保从 gen_data 获取新参数
        mutation_rate = gen_data.get('gene_mutation_rate', gen_data.get('mutation_rate', 0.0))
        avg_fitness = gen_data.get('avg_fitness', gen_data.get('avg_power', 0.0))
        
        # 获取新参数
        population_size = gen_data.get('population_size', 0)
        gene_mutation_rate = gen_data.get('gene_mutation_rate', mutation_rate)
        gene_crossover_rate = gen_data.get('gene_crossover_rate', 0.0)
        chromosome_crossover_rate = gen_data.get('chromosome_crossover_rate', 0.0)
        
        # 修复：检查position是否为None
        position_data = gen_data.get('position')
        if position_data is None:
            position_data = {}
            
        # 创建代记录
        gen_record = {
            'timestamp': datetime.now().isoformat(),
            'generation': current_generation,
            'best_power': float(gen_data.get('best_power', 0)),
            'current_power': float(gen_data.get('current_power', 0)),
            'avg_fitness': float(avg_fitness),
            'best_position': self._convert_position_to_serializable(position_data),
            'mutation_rate': float(mutation_rate),
            'population_size': population_size,
            'gene_mutation_rate': gene_mutation_rate,
            'gene_crossover_rate': gene_crossover_rate,
            'chromosome_crossover_rate': chromosome_crossover_rate,
            'evaluation_count': len(self.gui_data['evaluation_records']),
            'converged': gen_data.get('converged', False),
            'enhanced_exploration': gen_data.get('enhanced_exploration', False),
            'lock_mode_activated': gen_data.get('lock_mode_activated', False),
            'optimization_mode': self.optimization_mode.get(),
            'optimization_phase': self.current_phase  # 使用当前优化阶段
        }
        
        # 双端模式特定数据
        if self.optimization_mode.get() == "double":
            # 更新通光状态
            if 'light_detected' in gen_data:
                self.light_detected = gen_data['light_detected']
            
            gen_record['light_detected'] = self.light_detected
            
            # 保存A端和B端的最佳位置
            if 'best_position_A' in gen_data:
                gen_record['best_position_A'] = self._convert_position_to_serializable(gen_data['best_position_A'])
            if 'best_position_B' in gen_data:
                gen_record['best_position_B'] = self._convert_position_to_serializable(gen_data['best_position_B'])
            if 'best_power_A' in gen_data:
                gen_record['best_power_A'] = float(gen_data['best_power_A'])
            if 'best_power_B' in gen_data:
                gen_record['best_power_B'] = float(gen_data['best_power_B'])
        
        # 添加到GUI数据记录
        self.gui_data['generation_records'].append(gen_record)
        
        # 更新优化历史
        self.gui_data['optimization_history']['best_fitness_history'].append(float(gen_data.get('best_power', 0)))
        self.gui_data['optimization_history']['avg_fitness_history'].append(float(avg_fitness))
        self.gui_data['optimization_history']['mutation_rate_history'].append(float(mutation_rate))
        self.gui_data['optimization_history']['convergence_status_history'].append(gen_data.get('converged', False))
        
        # 标记当前代的最佳个体
        self._mark_best_individual(current_generation, gen_data.get('best_power', 0))
        
        # 更新会话统计
        self.current_session['total_generations'] = len(self.gui_data['generation_records'])

    def _mark_best_individual(self, generation, best_power):
        """标记当前代的最佳个体"""
        if not self.gui_data['evaluation_records']:
            return
        
        # 在当前代的评估中寻找最佳个体
        generation_evals = [e for e in self.gui_data['evaluation_records'] 
                        if e.get('generation', 0) == generation]
        
        if not generation_evals:
            return
        
        # 找到功率最接近最佳功率的个体
        best_eval = min(generation_evals, 
                    key=lambda x: abs(x.get('power', 0) - best_power))
        
        # 标记为最佳
        for eval_record in self.gui_data['evaluation_records']:
            if (eval_record.get('evaluation_count') == best_eval.get('evaluation_count') and
                eval_record.get('generation') == generation):
                eval_record['is_best'] = True

    def _update_generation_ui(self, gen_data):
        """更新代进度UI - 扩展支持双端，使用格式化功率，添加新参数显示"""
        try:
            # 更新代数显示
            if 'iteration' in gen_data and 'total_iterations' in gen_data:
                self.status_labels["generation"]["text"] = f"{gen_data['iteration']}/{gen_data['total_iterations']}"
                progress = (gen_data['iteration'] / gen_data['total_iterations']) * 100
                self.progress_var.set(progress)
            
            # 更新功率显示（使用格式化）
            if 'current_power' in gen_data:
                formatted_power = self.format_power_value(gen_data['current_power'])
                self.status_labels["current_power"]["text"] = formatted_power
            if 'best_power' in gen_data:
                formatted_best_power = self.format_power_value(gen_data['best_power'])
                self.status_labels["best_power"]["text"] = formatted_best_power
            
            # 更新新参数显示
            if 'population_size' in gen_data:
                self.status_labels["current_population_size"]["text"] = str(gen_data['population_size'])
            
            # 更新遗传算法参数显示
            if 'gene_mutation_rate' in gen_data:
                self.status_labels["current_gene_mutation_rate"]["text"] = f"{gen_data['gene_mutation_rate']:.3f}"
            elif 'mutation_rate' in gen_data:
                self.status_labels["current_gene_mutation_rate"]["text"] = f"{gen_data['mutation_rate']:.3f}"
            
            if 'gene_crossover_rate' in gen_data:
                self.status_labels["current_gene_crossover_rate"]["text"] = f"{gen_data['gene_crossover_rate']:.3f}"
            
            if 'chromosome_crossover_rate' in gen_data:
                self.status_labels["current_chromosome_crossover_rate"]["text"] = f"{gen_data['chromosome_crossover_rate']:.3f}"
            
            # 新增：更新高功率保持模式参数显示
            if 'high_power_search_range_percent' in gen_data:
                self.status_labels["current_high_power_search_range"]["text"] = f"±{gen_data['high_power_search_range_percent']*100:.1f}%"
            
            if 'high_power_perturbation_strength' in gen_data:
                # 如果有对应的标签，可以更新
                pass
            
            # 更新通光状态
            if 'light_detected' in gen_data:
                light_status = "通光" if gen_data['light_detected'] else "未通光"
                self.status_labels["light_status"]["text"] = light_status
            
            
            # 更新位置显示 - 显示历史最佳位置
            if hasattr(self, 'optimizer') and self.optimizer:
                # 获取历史最佳个体
                if hasattr(self.optimizer, 'best_individual_A_memory') and self.optimizer.best_individual_A_memory is not None:
                    for i, var in enumerate(['x', 'y', 'z', 'rx', 'ry']):
                        if i < len(self.optimizer.best_individual_A_memory):
                            if f"A_{var}" in self.position_labels_A:
                                self.position_labels_A[f"A_{var}"]["text"] = f"{self.optimizer.best_individual_A_memory[i]:.6f}"
                
                if hasattr(self.optimizer, 'best_individual_B_memory') and self.optimizer.best_individual_B_memory is not None:
                    for i, var in enumerate(['x', 'y', 'z', 'rx', 'ry']):
                        if i < len(self.optimizer.best_individual_B_memory):
                            if f"B_{var}" in self.position_labels_B:
                                self.position_labels_B[f"B_{var}"]["text"] = f"{self.optimizer.best_individual_B_memory[i]:.6f}"
            
            # 更新平均功率显示（使用格式化）- 增强健壮性
            avg_fitness = gen_data.get('avg_fitness')
            if avg_fitness is None:
                # 尝试从其他可能的字段获取
                avg_fitness = gen_data.get('avg_power', 0.0)
                if hasattr(self, 'optimizer') and self.optimizer:
                    # 如果回调数据中没有，尝试从优化器直接获取
                    if hasattr(self.optimizer, 'avg_fitness_history') and self.optimizer.avg_fitness_history:
                        avg_fitness = self.optimizer.avg_fitness_history[-1] if self.optimizer.avg_fitness_history else 0.0
            
            if avg_fitness is not None:
                formatted_avg_power = self.format_power_value(avg_fitness)
                self.status_labels["avg_power"]["text"] = formatted_avg_power
            else:
                self.status_labels["avg_power"]["text"] = "0.000000"
                
            # 更新优化时间（实时计算）
            if hasattr(self, 'optimization_start_time'):
                elapsed_time = time.time() - self.optimization_start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
                self.status_labels["opt_time"]["text"] = time_str
            
            # 更新优化模式信息
            if 'optimization_phase' in gen_data:
                phase_display = self._get_phase_display_name(gen_data['optimization_phase'])
                if "operation_mode" in self.status_labels:
                    self.status_labels["operation_mode"]["text"] = phase_display
            
            # 更新收敛信息
            if 'local_convergence_count' in gen_data:
                # 更新收敛状态显示，可以根据需要显示局部收敛次数或其他信息
                convergence_text = f"已收敛 ({gen_data['local_convergence_count']}次)" if gen_data['local_convergence_count'] > 0 else "未收敛"
                self.status_labels["convergence_status"]["text"] = convergence_text
            
            # 更新锁定状态
            if 'lock_mode_activated' in gen_data:
                lock_status = "已锁定" if gen_data['lock_mode_activated'] else "未锁定"
                self.status_labels["lock_status"]["text"] = lock_status
            
            # 更新局部优化信息
            if 'is_local_optimization' in gen_data:
                local_status = "局部优化中" if gen_data['is_local_optimization'] else "全局优化"
                if "status" in self.status_labels:
                    self.status_labels["status"]["text"] = local_status
            
        except Exception as e:
            self.log(f"更新UI时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    def _update_optimization_history(self, gen_data):
        """更新优化历史数据，确保所有字段都有值"""
        try:
            # 确保所有必需的字段都有默认值
            best_power = gen_data.get('best_power', 0.0)
            avg_fitness = gen_data.get('avg_fitness', gen_data.get('avg_power', 0.0))
            mutation_rate = gen_data.get('mutation_rate', gen_data.get('current_mutation_rate', 0.0))
            
            # 更新优化历史记录
            self.gui_data['optimization_history']['best_fitness_history'].append(float(best_power))
            self.gui_data['optimization_history']['avg_fitness_history'].append(float(avg_fitness))
            self.gui_data['optimization_history']['mutation_rate_history'].append(float(mutation_rate))
            self.gui_data['optimization_history']['convergence_status_history'].append(gen_data.get('converged', False))
            self.gui_data['optimization_history']['optimization_phase_history'].append(self.current_phase)
            
        except Exception as e:
            self.log(f"更新优化历史时出错: {str(e)}")
    def update_results_display(self, result):
        """更新结果显示 - 扩展支持双端，使用格式化功率，添加新参数显示"""
        mode = self.optimization_mode.get()
        
        # 更新结果摘要（使用格式化功率）
        best_power_formatted = self.format_power_value(result['best_power'])
        self.result_labels["best_power"]["text"] = best_power_formatted
        self.result_labels["total_evaluations"]["text"] = str(result['total_evaluations'])
        self.result_labels["total_generations"]["text"] = str(result['total_generations'])
        self.result_labels["result_mode"]["text"] = "双端优化" if mode == "double" else "单端优化"
        
        # 格式化时间
        opt_time = result['optimization_time']
        hours, remainder = divmod(opt_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.result_labels["optimization_time"]["text"] = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # 双端模式显示算法参数结果
        if mode == "double":
            # 创建算法参数结果显示框架
            algorithm_frame = ttk.LabelFrame(self.results_frame, text="算法参数结果", style='Custom.TLabelframe')
            
            # 移除 before 参数，使用简单的 pack 布局
            algorithm_frame.pack(fill=tk.X, padx=10, pady=10)
            
            algorithm_params = [
                ("最终种群大小:", "final_population_size", "0"),
                ("基因变异率:", "final_gene_mutation_rate", "0.000"),
                ("基因交叉率:", "final_gene_crossover_rate", "0.000"),
                ("染色体交叉率:", "final_chromosome_crossover_rate", "0.000"),
                ("局部收敛次数:", "local_convergence_count", "0"),
                ("高功率保持模式:", "high_power_keep_mode", "否"),
                ("位置锁定:", "lock_mode_activated", "否")
            ]
            
            self.algorithm_result_labels = {}
            for i, (label, key, default) in enumerate(algorithm_params):
                row = i // 2
                col = (i % 2) * 2
                ttk.Label(algorithm_frame, text=label, font=LARGE_FONT).grid(row=row, column=col, sticky=tk.W, padx=20, pady=5)
                value_label = ttk.Label(algorithm_frame, text=default, font=BOLD_FONT)
                value_label.grid(row=row, column=col+1, sticky=tk.W, padx=5, pady=5)
                self.algorithm_result_labels[key] = value_label
            
            # 更新算法参数结果
            if hasattr(self, 'algorithm_result_labels'):
                self.algorithm_result_labels["final_population_size"]["text"] = str(result.get('final_population_size', 0))
                self.algorithm_result_labels["final_gene_mutation_rate"]["text"] = f"{result.get('final_gene_mutation_rate', 0):.3f}"
                self.algorithm_result_labels["final_gene_crossover_rate"]["text"] = f"{result.get('final_gene_crossover_rate', 0):.3f}"
                self.algorithm_result_labels["final_chromosome_crossover_rate"]["text"] = f"{result.get('final_chromosome_crossover_rate', 0):.3f}"
                self.algorithm_result_labels["local_convergence_count"]["text"] = str(result.get('local_convergence_count', 0))
                self.algorithm_result_labels["high_power_keep_mode"]["text"] = "是" if result.get('high_power_keep_mode', False) else "否"
                self.algorithm_result_labels["lock_mode_activated"]["text"] = "是" if result.get('lock_mode_activated', False) else "否"
        
        # 更新位置信息
        if mode == "single":
            if result.get('best_position') is not None:
                for key, value in result['best_position'].items():
                    if f"A_{key}" in self.result_position_labels_A:
                        self.result_position_labels_A[f"A_{key}"]["text"] = f"{value:.6f}"
        else:
            # 双端模式分别显示A端和B端位置
            if 'best_position_A' in result and result['best_position_A'] is not None:
                for key, value in result['best_position_A'].items():
                    if key in self.result_position_labels_A:
                        self.result_position_labels_A[key]["text"] = f"{value:.6f}"
            
            if 'best_position_B' in result and result['best_position_B'] is not None:
                for key, value in result['best_position_B'].items():
                    if key in self.result_position_labels_B:
                        self.result_position_labels_B[key]["text"] = f"{value:.6f}"
        
        # 更新通光状态
        if 'light_detected' in result:
            light_status = "通光" if result['light_detected'] else "未通光"
            # 可以在结果页面添加通光状态显示
            if not hasattr(self, 'light_status_label'):
                light_frame = ttk.LabelFrame(self.results_frame, text="通光状态", style='Custom.TLabelframe')
                light_frame.pack(fill=tk.X, padx=10, pady=10)
                self.light_status_label = ttk.Label(light_frame, text=light_status, font=BOLD_FONT)
                self.light_status_label.pack(padx=10, pady=5)
            else:
                self.light_status_label["text"] = light_status

    def show_detailed_charts(self):
        """显示详细图表 - 扩展支持双端"""
        if not self.result or not self.result['success']:
            return
        
        mode = self.optimization_mode.get()
        if mode == "single":
            visualize_ga_results(self.result)
        else:
            visualize_dual_end_results(self.result)

    def save_results(self):
        """保存优化结果 - 扩展支持双端"""
        if not self.result or not self.result['success']:
            messagebox.showinfo("提示", "没有可保存的优化结果")
            return
        
        mode = self.optimization_mode.get()
        if mode == "single":
            save_ga_data(self.result)
        else:
            save_dual_end_ga_data(self.result)
            # 生成双端优化报告
            report_path = create_dual_end_report(self.result)
            if report_path:
                self.log(f"双端优化报告已生成: {report_path}")
        
        self.log("优化结果已保存")

if __name__ == "__main__":
    root = tk.Tk()
    app = DualEndGAOptimizerGUI(root)
    root.mainloop()