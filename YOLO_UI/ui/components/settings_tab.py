import os
import json
import subprocess
import sys
import platform
import re
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QLineEdit, QSpinBox, 
                            QDoubleSpinBox, QGroupBox, QCheckBox, QMessageBox,
                            QFormLayout, QGridLayout, QTabWidget, QFileDialog,
                            QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication
from utils.theme_manager import ThemeManager
import torch

class CUDAInstallerThread(QThread):
    """Thread for CUDA installation process."""
    progress = pyqtSignal(str)
    percent = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    cuda_url = None  # 新增属性
    
    def run(self):
        try:
            # 检测操作系统
            system = platform.system()
            if system != "Windows":
                self.finished.emit(False, "目前仅支持 Windows 系统自动安装 CUDA")
                return
                
            # 检测是否已安装 CUDA
            try:
                nvcc_output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT)
                self.progress.emit("已检测到 CUDA 安装")
                self.percent.emit(100)
                self.finished.emit(True, "CUDA 已安装")
                return
            except:
                self.progress.emit("未检测到 CUDA，准备安装...")
            
            # 下载 CUDA 安装程序
            self.progress.emit("正在下载 CUDA 安装程序...")
            # 使用传入的下载链接
            cuda_url = self.cuda_url or "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_546.12_windows.exe"
            installer_path = os.path.join(os.environ['TEMP'], "cuda_installer.exe")
            
            try:
                import urllib.request
                def reporthook(blocknum, blocksize, totalsize):
                    if totalsize > 0:
                        percent = int(blocknum * blocksize * 100 / totalsize)
                        if percent > 100:
                            percent = 100
                        self.percent.emit(percent)
                urllib.request.urlretrieve(cuda_url, installer_path, reporthook)
                self.percent.emit(100)
            except Exception as e:
                self.finished.emit(False, f"下载 CUDA 安装程序失败: {str(e)}")
                return
            
            # 运行安装程序
            self.progress.emit("正在安装 CUDA...")
            self.percent.emit(-1)  # -1 表示切换为文本进度
            try:
                subprocess.run([installer_path, '-s'], check=True)
                self.progress.emit("CUDA 安装完成")
                self.finished.emit(True, "CUDA 安装成功")
            except Exception as e:
                self.finished.emit(False, f"CUDA 安装失败: {str(e)}")
            
        except Exception as e:
            self.finished.emit(False, f"安装过程出错: {str(e)}")

class PipTorchInstallerThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    cuda_tag = None
    def run(self):
        try:
            self.progress.emit(f"正在通过 pip 安装 PyTorch (CUDA {self.cuda_tag})...")
            cmd = [sys.executable, '-m', 'pip', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', f'https://download.pytorch.org/whl/{self.cuda_tag}']
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8')
            if proc.returncode == 0:
                # 检查 CUDA 是否可用
                import importlib
                importlib.reload(torch)
                if torch.cuda.is_available():
                    self.finished.emit(True, f"PyTorch (CUDA {self.cuda_tag}) 安装成功，已检测到 CUDA 可用！")
                else:
                    self.finished.emit(False, f"PyTorch (CUDA {self.cuda_tag}) 安装成功，但未检测到 CUDA 可用。请检查驱动或重启电脑。\n\n安装输出：\n{proc.stdout}")
            else:
                self.finished.emit(False, f"pip 安装失败，返回码 {proc.returncode}。\n\n安装输出：\n{proc.stdout}")
        except Exception as e:
            self.finished.emit(False, f"pip 安装过程出错: {e}")

class SettingsTab(QWidget):
    """Tab for application settings and preferences."""
    
    # Signal to notify other tabs when settings are updated
    settings_updated = pyqtSignal(dict)
    theme_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        
        # Initialize default settings
        self.settings = {
            'default_model': 'yolov8n',
            'default_batch_size': 16,
            'default_img_size': 640,
            'default_conf_thresh': 0.25,
            'default_iou_thresh': 0.45,
            'use_gpu': True,
            'gpu_device': 0,
            'default_train_dir': '',
            'default_val_dir': '',
            'default_test_dir': '',
            'default_output_dir': '',
            'default_train_model_path': '',
            'default_test_model_path': '',
            'theme': 'tech',  # 默认主题为科技感主题
            'default_train_images_dir': '',
            'default_train_labels_dir': '',
            'default_val_images_dir': '',
            'default_val_labels_dir': '',
            'default_test_images_dir': '',
            'default_test_labels_dir': ''
        }
        
        # Load settings if available
        self.settings_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'settings.json')
        self.setup_ui()
        self.load_settings()
        
        # 初始化 CUDA 安装线程
        self.cuda_installer = None
    
    def setup_ui(self):
        """Create and arrange UI elements."""
        main_layout = QVBoxLayout(self)
        
        # Create tabs for different settings categories
        settings_tabs = QTabWidget()
        
        # General settings tab
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        # Hardware settings
        hardware_group = QGroupBox("硬件设置")
        hardware_layout = QFormLayout()
        
        # 添加 CUDA 状态检测
        self.cuda_status_label = QLabel("CUDA 状态: 检测中...")
        self.check_cuda_status()
        hardware_layout.addRow("", self.cuda_status_label)
        
        # 添加安装 CUDA 按钮
        self.install_cuda_btn = QPushButton("安装/更新 CUDA")
        self.install_cuda_btn.clicked.connect(self.install_cuda)
        hardware_layout.addRow("", self.install_cuda_btn)
        
        self.use_gpu_check = QCheckBox("使用GPU")
        self.use_gpu_check.setChecked(self.settings['use_gpu'])
        
        self.gpu_device_spin = QSpinBox()
        self.gpu_device_spin.setRange(0, 8)
        self.gpu_device_spin.setValue(self.settings['gpu_device'])
        self.gpu_device_spin.setEnabled(self.settings['use_gpu'])
        
        hardware_layout.addRow("", self.use_gpu_check)
        hardware_layout.addRow("GPU设备:", self.gpu_device_spin)
        hardware_group.setLayout(hardware_layout)
        
        # Default model settings
        model_group = QGroupBox("默认模型设置")
        model_layout = QFormLayout()
        
        # self.model_combo = QComboBox()
        # self.model_combo.addItems(["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        #                           "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
        #                           "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x"])
        # index = self.model_combo.findText(self.settings['default_model'])
        # if index >= 0:
        #     self.model_combo.setCurrentIndex(index)
        # model_layout.addRow("默认模型:", self.model_combo)
        # model_group.setLayout(model_layout)

        # UI settings group
        ui_group = QGroupBox("界面设置")
        ui_layout = QFormLayout()
        
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["浅色主题", "深色主题", "科技感主题"])
        
        # 根据设置确定当前主题索引
        if self.settings['theme'] == 'light':
            theme_index = 0
        elif self.settings['theme'] == 'dark':
            theme_index = 1
        else:  # tech theme
            theme_index = 2
            
        self.theme_combo.setCurrentIndex(theme_index)
        
        ui_layout.addRow("主题:", self.theme_combo)
        ui_group.setLayout(ui_layout)
        
        # Add groups to general tab
        general_layout.addWidget(hardware_group)
        # general_layout.addWidget(model_group) # 注释掉默认模型设置
        general_layout.addWidget(ui_group)
        
        # Path settings tab
        paths_tab = QWidget()
        paths_layout = QFormLayout(paths_tab)
        
        # Default path settings
        paths_group = QGroupBox("默认目录")
        paths_form_layout = QFormLayout()
        
        # Training data path
        self.train_dir_layout = QHBoxLayout()
        self.train_dir_edit = QLineEdit(self.settings['default_train_dir'])
        self.train_dir_edit.setReadOnly(True)
        self.train_dir_btn = QPushButton("浏览...")
        self.train_dir_layout.addWidget(self.train_dir_edit)
        self.train_dir_layout.addWidget(self.train_dir_btn)
        
        # Validation data path
        self.val_dir_layout = QHBoxLayout()
        self.val_dir_edit = QLineEdit(self.settings['default_val_dir'])
        self.val_dir_edit.setReadOnly(True)
        self.val_dir_btn = QPushButton("浏览...")
        self.val_dir_layout.addWidget(self.val_dir_edit)
        self.val_dir_layout.addWidget(self.val_dir_btn)
        
        # Test data path
        self.test_dir_layout = QHBoxLayout()
        self.test_dir_edit = QLineEdit(self.settings['default_test_dir'])
        self.test_dir_edit.setReadOnly(True)
        self.test_dir_btn = QPushButton("浏览...")
        self.test_dir_layout.addWidget(self.test_dir_edit)
        self.test_dir_layout.addWidget(self.test_dir_btn)
        
        # Output path
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit(self.settings['default_output_dir'])
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
        # Training model path
        self.train_model_layout = QHBoxLayout()
        self.train_model_edit = QLineEdit(self.settings['default_train_model_path'])
        self.train_model_edit.setReadOnly(True)
        self.train_model_btn = QPushButton("浏览...")
        self.train_model_layout.addWidget(self.train_model_edit)
        self.train_model_layout.addWidget(self.train_model_btn)
        
        # Testing model path
        self.test_model_layout = QHBoxLayout()
        self.test_model_edit = QLineEdit(self.settings['default_test_model_path'])
        self.test_model_edit.setReadOnly(True)
        self.test_model_btn = QPushButton("浏览...")
        self.test_model_layout.addWidget(self.test_model_edit)
        self.test_model_layout.addWidget(self.test_model_btn)
        
        # 新增：默认训练图像目录
        self.default_train_images_layout = QHBoxLayout()
        self.default_train_images_edit = QLineEdit(self.settings.get('default_train_images_dir', ''))
        self.default_train_images_edit.setReadOnly(True)
        self.default_train_images_btn = QPushButton("浏览...")
        self.default_train_images_layout.addWidget(self.default_train_images_edit)
        self.default_train_images_layout.addWidget(self.default_train_images_btn)
        paths_form_layout.addRow("默认训练图像目录:", self.default_train_images_layout)

        # 新增：默认训练标签目录
        self.default_train_labels_layout = QHBoxLayout()
        self.default_train_labels_edit = QLineEdit(self.settings.get('default_train_labels_dir', ''))
        self.default_train_labels_edit.setReadOnly(True)
        self.default_train_labels_btn = QPushButton("浏览...")
        self.default_train_labels_layout.addWidget(self.default_train_labels_edit)
        self.default_train_labels_layout.addWidget(self.default_train_labels_btn)
        paths_form_layout.addRow("默认训练标签目录:", self.default_train_labels_layout)

        # 新增：默认验证图像目录
        self.default_val_images_layout = QHBoxLayout()
        self.default_val_images_edit = QLineEdit(self.settings.get('default_val_images_dir', ''))
        self.default_val_images_edit.setReadOnly(True)
        self.default_val_images_btn = QPushButton("浏览...")
        self.default_val_images_layout.addWidget(self.default_val_images_edit)
        self.default_val_images_layout.addWidget(self.default_val_images_btn)
        paths_form_layout.addRow("默认验证图像目录:", self.default_val_images_layout)

        # 新增：默认验证标签目录
        self.default_val_labels_layout = QHBoxLayout()
        self.default_val_labels_edit = QLineEdit(self.settings.get('default_val_labels_dir', ''))
        self.default_val_labels_edit.setReadOnly(True)
        self.default_val_labels_btn = QPushButton("浏览...")
        self.default_val_labels_layout.addWidget(self.default_val_labels_edit)
        self.default_val_labels_layout.addWidget(self.default_val_labels_btn)
        paths_form_layout.addRow("默认验证标签目录:", self.default_val_labels_layout)

        # 新增：默认测试图像目录
        self.default_test_images_layout = QHBoxLayout()
        self.default_test_images_edit = QLineEdit(self.settings.get('default_test_images_dir', ''))
        self.default_test_images_edit.setReadOnly(True)
        self.default_test_images_btn = QPushButton("浏览...")
        self.default_test_images_layout.addWidget(self.default_test_images_edit)
        self.default_test_images_layout.addWidget(self.default_test_images_btn)
        paths_form_layout.addRow("默认测试图像目录:", self.default_test_images_layout)

        # 新增：默认测试标签目录
        self.default_test_labels_layout = QHBoxLayout()
        self.default_test_labels_edit = QLineEdit(self.settings.get('default_test_labels_dir', ''))
        self.default_test_labels_edit.setReadOnly(True)
        self.default_test_labels_btn = QPushButton("浏览...")
        self.default_test_labels_layout.addWidget(self.default_test_labels_edit)
        self.default_test_labels_layout.addWidget(self.default_test_labels_btn)
        paths_form_layout.addRow("默认测试标签目录:", self.default_test_labels_layout)
        
        paths_form_layout.addRow("输出目录:", self.output_dir_layout)
        paths_form_layout.addRow("训练模型:", self.train_model_layout)
        paths_form_layout.addRow("测试模型:", self.test_model_layout)
        
        paths_group.setLayout(paths_form_layout)
        paths_layout.addWidget(paths_group)
        
        # Training settings tab
        training_tab = QWidget()
        training_layout = QFormLayout(training_tab)
        
        # Default training parameters
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(self.settings['default_batch_size'])
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(self.settings['default_img_size'])
        self.img_size_spin.setSingleStep(32)
        
        training_layout.addRow("默认批次大小:", self.batch_size_spin)
        training_layout.addRow("默认图像尺寸:", self.img_size_spin)
        
        # Testing settings tab
        testing_tab = QWidget()
        testing_layout = QFormLayout(testing_tab)
        
        # Default testing parameters
        self.conf_thresh_spin = QDoubleSpinBox()
        self.conf_thresh_spin.setRange(0.1, 1.0)
        self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
        self.conf_thresh_spin.setSingleStep(0.05)
        
        self.iou_thresh_spin = QDoubleSpinBox()
        self.iou_thresh_spin.setRange(0.1, 1.0)
        self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
        self.iou_thresh_spin.setSingleStep(0.05)
        
        testing_layout.addRow("默认置信度阈值:", self.conf_thresh_spin)
        testing_layout.addRow("默认IoU阈值:", self.iou_thresh_spin)
        
        # Add tabs to the settings tab widget
        settings_tabs.addTab(general_tab, "通用")
        settings_tabs.addTab(paths_tab, "路径")
        settings_tabs.addTab(training_tab, "训练")
        settings_tabs.addTab(testing_tab, "测试")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("保存设置")
        self.save_btn.setMinimumHeight(40)
        
        self.reset_btn = QPushButton("恢复默认")
        self.reset_btn.setMinimumHeight(40)
        
        button_layout.addWidget(self.save_btn)
        button_layout.addWidget(self.reset_btn)
        
        # Add widgets to main layout
        main_layout.addWidget(settings_tabs)
        main_layout.addLayout(button_layout)
        
        # Connect signals
        self.connect_signals()
    
    def connect_signals(self):
        """Connect UI signals to slots."""
        self.save_btn.clicked.connect(self.save_settings)
        self.reset_btn.clicked.connect(self.reset_settings)
        self.use_gpu_check.toggled.connect(self.toggle_gpu_settings)
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)
        
        # Connect path selection buttons
        self.train_dir_btn.clicked.connect(self.select_train_dir)
        self.val_dir_btn.clicked.connect(self.select_val_dir)
        self.test_dir_btn.clicked.connect(self.select_test_dir)
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        self.train_model_btn.clicked.connect(self.select_train_model)
        self.test_model_btn.clicked.connect(self.select_test_model)
        
        # 新增：默认训练图像目录
        self.default_train_images_btn.clicked.connect(lambda: self.select_directory(self.default_train_images_edit, "选择默认训练图像目录"))
        self.default_train_labels_btn.clicked.connect(lambda: self.select_directory(self.default_train_labels_edit, "选择默认训练标签目录"))
        self.default_val_images_btn.clicked.connect(lambda: self.select_directory(self.default_val_images_edit, "选择默认验证图像目录"))
        self.default_val_labels_btn.clicked.connect(lambda: self.select_directory(self.default_val_labels_edit, "选择默认验证标签目录"))
        self.default_test_images_btn.clicked.connect(lambda: self.select_directory(self.default_test_images_edit, "选择默认测试图像目录"))
        self.default_test_labels_btn.clicked.connect(lambda: self.select_directory(self.default_test_labels_edit, "选择默认测试标签目录"))
    
    def toggle_gpu_settings(self, checked):
        """Enable or disable GPU-related settings based on checkbox."""
        self.gpu_device_spin.setEnabled(checked)
    
    def apply_theme(self, index):
        """应用选中的主题"""
        app = QApplication.instance()
        if index == 0:  # 浅色主题
            ThemeManager.apply_light_theme(app)
            self.settings['theme'] = 'light'
        elif index == 1:  # 深色主题
            ThemeManager.apply_dark_theme(app)
            self.settings['theme'] = 'dark'
        else:  # tech theme
            ThemeManager.apply_tech_theme(app)
            self.settings['theme'] = 'tech'
        
        # 发送主题已更改的信号
        self.theme_changed.emit(self.settings['theme'])
    
    def select_train_dir(self):
        """Open dialog to select training data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认训练数据目录")
        if dir_path:
            self.train_dir_edit.setText(dir_path)
    
    def select_val_dir(self):
        """Open dialog to select validation data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认验证数据目录")
        if dir_path:
            self.val_dir_edit.setText(dir_path)
    
    def select_test_dir(self):
        """Open dialog to select test data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认测试数据目录")
        if dir_path:
            self.test_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "选择默认输出目录")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_train_model(self):
        """Open dialog to select training model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择默认训练模型", "", "模型文件 (*.pt *.pth *.weights);;所有文件 (*)"
        )
        if file_path:
            self.train_model_edit.setText(file_path)
    
    def select_test_model(self):
        """Open dialog to select testing model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择默认测试模型", "", "模型文件 (*.pt *.pth *.weights);;所有文件 (*)"
        )
        if file_path:
            self.test_model_edit.setText(file_path)
    
    def select_directory(self, line_edit, title):
        dir_path = QFileDialog.getExistingDirectory(self, title)
        if dir_path:
            line_edit.setText(dir_path)
    
    def save_settings(self, show_message=True):
        """Save current settings to file and emit signal to update other tabs."""
        # Update settings dict from UI
        # self.settings['default_model'] = self.model_combo.currentText()  # 已无model_combo，注释掉
        self.settings['default_batch_size'] = self.batch_size_spin.value()
        self.settings['default_img_size'] = self.img_size_spin.value()
        self.settings['default_conf_thresh'] = self.conf_thresh_spin.value()
        self.settings['default_iou_thresh'] = self.iou_thresh_spin.value()
        self.settings['use_gpu'] = self.use_gpu_check.isChecked()
        self.settings['gpu_device'] = self.gpu_device_spin.value()
        
        # 更新主题设置
        theme_index = self.theme_combo.currentIndex()
        if theme_index == 0:
            self.settings['theme'] = 'light'
        elif theme_index == 1:
            self.settings['theme'] = 'dark' 
        else:
            self.settings['theme'] = 'tech'
        
        # Update path settings
        self.settings['default_train_dir'] = self.train_dir_edit.text()
        self.settings['default_val_dir'] = self.val_dir_edit.text()
        self.settings['default_test_dir'] = self.test_dir_edit.text()
        self.settings['default_output_dir'] = self.output_dir_edit.text()
        self.settings['default_train_model_path'] = self.train_model_edit.text()
        self.settings['default_test_model_path'] = self.test_model_edit.text()
        
        # 新增：默认训练图像目录
        self.settings['default_train_images_dir'] = self.default_train_images_edit.text()
        self.settings['default_train_labels_dir'] = self.default_train_labels_edit.text()
        self.settings['default_val_images_dir'] = self.default_val_images_edit.text()
        self.settings['default_val_labels_dir'] = self.default_val_labels_edit.text()
        self.settings['default_test_images_dir'] = self.default_test_images_edit.text()
        self.settings['default_test_labels_dir'] = self.default_test_labels_edit.text()
        
        # Save to file
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=4)
            
            # Notify other tabs
            self.settings_updated.emit(self.settings)
            
            if show_message:
                QMessageBox.information(self, "设置已保存", "设置已成功保存。")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存设置失败: {str(e)}")
    
    def load_settings(self):
        """Load settings from file if it exists."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                for key, value in loaded_settings.items():
                    if key in self.settings:
                        self.settings[key] = value
                # self.model_combo.setCurrentText(self.settings['default_model'])  # 已无model_combo，注释掉
                self.batch_size_spin.setValue(self.settings['default_batch_size'])
                self.img_size_spin.setValue(self.settings['default_img_size'])
                self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
                self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
                self.use_gpu_check.setChecked(self.settings['use_gpu'])
                self.gpu_device_spin.setValue(self.settings['gpu_device'])
                self.train_dir_edit.setText(self.settings['default_train_dir'])
                self.val_dir_edit.setText(self.settings['default_val_dir'])
                self.test_dir_edit.setText(self.settings['default_test_dir'])
                self.output_dir_edit.setText(self.settings['default_output_dir'])
                self.train_model_edit.setText(self.settings['default_train_model_path'])
                self.test_model_edit.setText(self.settings['default_test_model_path'])
                self.theme_combo.setCurrentIndex(2)  # 设置为科技感主题
                
                # 应用默认主题
                app = QApplication.instance()
                ThemeManager.apply_tech_theme(app)
                
                # 新增：默认训练图像目录
                self.default_train_images_edit.setText(self.settings.get('default_train_images_dir', ''))
                self.default_train_labels_edit.setText(self.settings.get('default_train_labels_dir', ''))
                self.default_val_images_edit.setText(self.settings.get('default_val_images_dir', ''))
                self.default_val_labels_edit.setText(self.settings.get('default_val_labels_dir', ''))
                self.default_test_images_edit.setText(self.settings.get('default_test_images_dir', ''))
                self.default_test_labels_edit.setText(self.settings.get('default_test_labels_dir', ''))
                
                # Save to file and notify other tabs（不弹窗）
                self.save_settings(show_message=False)
            
            except Exception as e:
                print(f"加载设置时出错: {str(e)}")
    
    def reset_settings(self):
        """Reset settings to default values."""
        reply = QMessageBox.question(
            self, '确认重置',
            "你确定要将所有设置重置为默认值吗？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.settings = {
                'default_model': 'yolov8n',
                'default_batch_size': 16,
                'default_img_size': 640,
                'default_conf_thresh': 0.25,
                'default_iou_thresh': 0.45,
                'use_gpu': True,
                'gpu_device': 0,
                'default_train_dir': '',
                'default_val_dir': '',
                'default_test_dir': '',
                'default_output_dir': '',
                'default_train_model_path': '',
                'default_test_model_path': '',
                'theme': 'tech',  # 默认重置为科技感主题
                'default_train_images_dir': '',
                'default_train_labels_dir': '',
                'default_val_images_dir': '',
                'default_val_labels_dir': '',
                'default_test_images_dir': '',
                'default_test_labels_dir': ''
            }
            # self.model_combo.setCurrentText(self.settings['default_model'])  # 已无model_combo，注释掉
            self.batch_size_spin.setValue(self.settings['default_batch_size'])
            self.img_size_spin.setValue(self.settings['default_img_size'])
            self.conf_thresh_spin.setValue(self.settings['default_conf_thresh'])
            self.iou_thresh_spin.setValue(self.settings['default_iou_thresh'])
            self.use_gpu_check.setChecked(self.settings['use_gpu'])
            self.gpu_device_spin.setValue(self.settings['gpu_device'])
            self.train_dir_edit.setText(self.settings['default_train_dir'])
            self.val_dir_edit.setText(self.settings['default_val_dir'])
            self.test_dir_edit.setText(self.settings['default_test_dir'])
            self.output_dir_edit.setText(self.settings['default_output_dir'])
            self.train_model_edit.setText(self.settings['default_train_model_path'])
            self.test_model_edit.setText(self.settings['default_test_model_path'])
            self.theme_combo.setCurrentIndex(2)  # 设置为科技感主题
            
            # 应用默认主题
            app = QApplication.instance()
            ThemeManager.apply_tech_theme(app)
            
            # 新增：默认训练图像目录
            self.default_train_images_edit.setText(self.settings['default_train_images_dir'])
            self.default_train_labels_edit.setText(self.settings['default_train_labels_dir'])
            self.default_val_images_edit.setText(self.settings['default_val_images_dir'])
            self.default_val_labels_edit.setText(self.settings['default_val_labels_dir'])
            self.default_test_images_edit.setText(self.settings['default_test_images_dir'])
            self.default_test_labels_edit.setText(self.settings['default_test_labels_dir'])
            
            # Save to file and notify other tabs
            self.save_settings()
    
    def check_cuda_status(self):
        """检查 CUDA 状态并更新显示"""
        try:
            # 检查 CUDA 是否可用
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                self.cuda_status_label.setText(f"CUDA 状态: 可用 (检测到 {device_count} 个 GPU)\nGPU: {device_name}")
                self.cuda_status_label.setStyleSheet("color: green;")
            else:
                self.cuda_status_label.setText("CUDA 状态: 不可用")
                self.cuda_status_label.setStyleSheet("color: red;")
        except Exception as e:
            self.cuda_status_label.setText(f"CUDA 状态: 检测失败 ({str(e)})")
            self.cuda_status_label.setStyleSheet("color: orange;")
    
    def get_supported_cuda_version(self):
        """检测本机支持的 CUDA 版本（通过 nvidia-smi）"""
        try:
            output = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
            match = re.search(r'CUDA Version: ([\\d.]+)', output)
            if match:
                return match.group(1)
        except Exception as e:
            return None

    def get_cuda_download_url(self, version):
        """根据版本返回 CUDA 安装包下载链接（可扩展更多版本）"""
        cuda_links = {
            '12.3': "https://developer.download.nvidia.com/compute/cuda/12.3.2/local_installers/cuda_12.3.2_546.12_windows.exe",
            '12.2': "https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe",
            '11.8': "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe"
        }
        # 只取主次版本号
        for k in cuda_links:
            if version.startswith(k):
                return cuda_links[k], k
        # 默认返回 12.3
        return cuda_links['12.3'], '12.3'

    def get_pip_cuda_tag(self, version):
        """根据CUDA版本返回pip源tag"""
        # 只取主次版本号
        if version.startswith('12.8') or version.startswith('12.3') or version.startswith('12.2'):
            return 'cu128'
        elif version.startswith('12.1'):
            return 'cu121'
        elif version.startswith('11.8'):
            return 'cu118'
        else:
            return 'cu118'  # 默认用cu118

    def install_cuda(self):
        """安装或更新 CUDA，先检测支持的版本"""
        supported_version = self.get_supported_cuda_version()
        if supported_version:
            url, recommend_version = self.get_cuda_download_url(supported_version)
            pip_tag = self.get_pip_cuda_tag(supported_version)
            msg = f"检测到你的 GPU 支持 CUDA {supported_version}\n推荐安装 CUDA {recommend_version} 版本。\n\n你可以选择：\n1. 直接用 pip 安装 PyTorch (含 CUDA 支持)\n2. 下载并安装 NVIDIA 官方 CUDA Toolkit\n\n是否直接用 pip 安装 PyTorch (推荐)？"
            reply = QMessageBox.question(
                self, '安装 PyTorch (含 CUDA)',
                msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.install_pip_torch(pip_tag)
                return
        else:
            url, recommend_version = self.get_cuda_download_url('12.3')
            pip_tag = 'cu128'
            msg = "未检测到 NVIDIA GPU 或驱动，或未安装 nvidia-smi。\n将默认下载安装 CUDA 12.3。\n建议优先用 pip 安装 PyTorch (含 CUDA 支持)。\n\n是否直接用 pip 安装 PyTorch？"
            reply = QMessageBox.question(
                self, '安装 PyTorch (含 CUDA)',
                msg,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.install_pip_torch(pip_tag)
                return
        # 用户选择否，继续原有 CUDA Toolkit 安装流程
        progress = QProgressDialog("正在下载 CUDA...", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("CUDA 安装")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        self.cuda_installer = CUDAInstallerThread()
        self.cuda_installer.cuda_url = url
        self.cuda_installer.progress.connect(lambda msg: progress.setLabelText(msg))
        self.cuda_installer.percent.connect(lambda p: progress.setValue(p) if p >= 0 else progress.setRange(0,0))
        self.cuda_installer.finished.connect(
            lambda success, msg: self.on_cuda_installation_finished(success, msg, progress)
        )
        self.cuda_installer.start()

    def install_pip_torch(self, pip_tag):
        progress = QProgressDialog(f"正在通过 pip 安装 PyTorch (CUDA {pip_tag})...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("PyTorch 安装")
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        self.pip_torch_installer = PipTorchInstallerThread()
        self.pip_torch_installer.cuda_tag = pip_tag
        self.pip_torch_installer.progress.connect(lambda msg: progress.setLabelText(msg))
        self.pip_torch_installer.finished.connect(
            lambda success, msg: self.on_pip_torch_installation_finished(success, msg, progress)
        )
        self.pip_torch_installer.start()

    def on_pip_torch_installation_finished(self, success, message, progress):
        progress.close()
        if success:
            QMessageBox.information(self, "PyTorch 安装", message)
        else:
            QMessageBox.critical(self, "PyTorch 安装失败", message)
        self.pip_torch_installer = None

    def on_cuda_installation_finished(self, success, message, progress):
        progress.close()
        if success:
            # 安装成功后重新检测 CUDA
            self.check_cuda_status()
            # 检查后再判断
            if torch.cuda.is_available():
                QMessageBox.information(self, "CUDA 安装", message + "\nCUDA 已可用。")
            else:
                QMessageBox.warning(self, "CUDA 安装", message + "\n但未检测到 CUDA 可用，建议重启电脑或使用 CPU 进行训练。")
                self.cuda_status_label.setText("CUDA 状态: 不可用，建议重启或使用 CPU")
                self.cuda_status_label.setStyleSheet("color: orange;")
        else:
            QMessageBox.critical(self, "CUDA 安装失败", message + "\n建议使用 CPU 进行训练。")
            self.cuda_status_label.setText("CUDA 状态: 不可用，建议使用 CPU")
            self.cuda_status_label.setStyleSheet("color: orange;")
        self.cuda_installer = None