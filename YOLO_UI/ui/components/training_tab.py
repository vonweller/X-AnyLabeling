import os
import sys
import time
import subprocess
import platform
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QScrollArea,
                            QRadioButton, QButtonGroup, QFormLayout, QSlider,
                            QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QColor, QFont, QDesktopServices
import yaml

from utils.training_worker import TrainingWorker, check_ultralytics_version_compatibility
from utils.data_validator import validate_yolo_dataset, inspect_dataset_structure
from utils.theme_manager import ThemeManager
from ultralytics.models import yolo # Import yolo for model list

class TrainingTab(QWidget):
    """Tab for YOLO model training configuration and execution."""
    
    def __init__(self):
        super().__init__()
        self.is_training = False
        self.training_worker = None
        self.training_thread = None
        
        # Default settings
        self.task_type = "detect" # "detect" or "classify"
        self.dataset_format = "YOLO"  # Default dataset format
        self.model_type = "yolov8n.pt"  # Default YOLO model, ensure .pt extension
        self.train_mode = "pretrained"  # Default to using pretrained weights
        self.model_source_option = "download" # "download", "local_folder", "custom_file"
        
        # Default paths (will be updated from settings if available)
        self.default_train_dir = ""
        self.default_val_dir = ""
        self.default_output_dir = ""
        self.default_model_path = ""  # For custom .pt file
        self.default_local_model_search_dir = "" # For local model folder
        
        # Create scroll area with improved settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QScrollArea.NoFrame)  # 移除边框
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                width: 12px;
            }
            QScrollBar:horizontal {
                height: 12px;
            }
        """)
        
        # Create container widget for scroll area
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(15)  # 增加组件之间的间距
        main_layout.setContentsMargins(5, 5, 5, 5)  # 设置边距
        
        # Set up UI
        self.setup_ui(main_layout)
        
        # Set container as scroll area widget
        scroll.setWidget(container)
        
        # Create layout for this widget and add scroll area
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)
        
        # Initialize UI states
        self.update_task_specific_ui() # Call this to set initial UI based on task
        self.update_model_list() # Populate model_combo early
        self.download_model_radio.setChecked(True) # Set default source
        self.update_model_source_ui_state() # Apply initial UI state for model source
        self.update_fine_tuning_state()
        
        # 确保"打开模型目录"按钮初始状态正确
        self.open_model_folder_btn.setVisible(True)  # 默认在下载模式时显示
        
        # 设置所有📁按钮的统一样式
        self.setup_folder_button_styles()
    
    def setup_folder_button_styles(self):
        """设置所有📁按钮的统一样式"""
        folder_buttons = [
            self.train_images_open_btn,
            self.train_labels_open_btn,
            self.val_images_open_btn,
            self.val_labels_open_btn,
            self.data_yaml_open_btn,
            self.local_model_folder_open_btn,
            self.custom_model_path_open_btn,
            self.output_dir_open_btn
        ]
        
        button_style = """
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 3px;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                border: 1px solid #666666;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border: 1px solid #333333;
            }
        """
        
        for button in folder_buttons:
            button.setStyleSheet(button_style)
    
    def setup_ui(self, main_layout):
        """Create and arrange UI elements."""
        # Task Type Selection
        task_group = QGroupBox("任务类型")
        task_layout = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["目标检测 (Detection)", "图像分类 (Classification)"])
        self.task_combo.currentIndexChanged.connect(self.on_task_type_changed)
        task_layout.addWidget(QLabel("选择任务:"))
        task_layout.addWidget(self.task_combo)
        task_group.setLayout(task_layout)
        main_layout.addWidget(task_group)

        # Data section
        self.data_group = QGroupBox("数据集 (目标检测)")
        data_layout = QFormLayout()
        
        # Training data
        self.train_images_layout = QHBoxLayout()
        self.train_images_edit = QLineEdit()
        self.train_images_edit.setReadOnly(True)
        self.train_images_btn = QPushButton("浏览...")
        self.train_images_open_btn = QPushButton("📁")
        self.train_images_open_btn.setToolTip("打开训练图像目录")
        self.train_images_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.train_images_open_btn.clicked.connect(lambda: self.open_folder(self.train_images_edit.text()))
        self.train_images_layout.addWidget(self.train_images_edit)
        self.train_images_layout.addWidget(self.train_images_btn)
        self.train_images_layout.addWidget(self.train_images_open_btn)
        self.train_images_label = QLabel("训练图像目录:")
        data_layout.addRow(self.train_images_label, self.train_images_layout)
        
        self.train_labels_layout = QHBoxLayout()
        self.train_labels_edit = QLineEdit()
        self.train_labels_edit.setReadOnly(True)
        self.train_labels_btn = QPushButton("浏览...")
        self.train_labels_open_btn = QPushButton("📁")
        self.train_labels_open_btn.setToolTip("打开训练标签目录")
        self.train_labels_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.train_labels_open_btn.clicked.connect(lambda: self.open_folder(self.train_labels_edit.text()))
        self.train_labels_layout.addWidget(self.train_labels_edit)
        self.train_labels_layout.addWidget(self.train_labels_btn)
        self.train_labels_layout.addWidget(self.train_labels_open_btn)
        self.train_labels_label = QLabel("训练标签目录:")
        data_layout.addRow(self.train_labels_label, self.train_labels_layout)
        
        # Validation data
        self.val_images_layout = QHBoxLayout()
        self.val_images_edit = QLineEdit()
        self.val_images_edit.setReadOnly(True)
        self.val_images_btn = QPushButton("浏览...")
        self.val_images_open_btn = QPushButton("📁")
        self.val_images_open_btn.setToolTip("打开验证图像目录")
        self.val_images_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.val_images_open_btn.clicked.connect(lambda: self.open_folder(self.val_images_edit.text()))
        self.val_images_layout.addWidget(self.val_images_edit)
        self.val_images_layout.addWidget(self.val_images_btn)
        self.val_images_layout.addWidget(self.val_images_open_btn)
        self.val_images_label = QLabel("验证图像目录:")
        data_layout.addRow(self.val_images_label, self.val_images_layout)
        
        self.val_labels_layout = QHBoxLayout()
        self.val_labels_edit = QLineEdit()
        self.val_labels_edit.setReadOnly(True)
        self.val_labels_btn = QPushButton("浏览...")
        self.val_labels_open_btn = QPushButton("📁")
        self.val_labels_open_btn.setToolTip("打开验证标签目录")
        self.val_labels_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.val_labels_open_btn.clicked.connect(lambda: self.open_folder(self.val_labels_edit.text()))
        self.val_labels_layout.addWidget(self.val_labels_edit)
        self.val_labels_layout.addWidget(self.val_labels_btn)
        self.val_labels_layout.addWidget(self.val_labels_open_btn)
        self.val_labels_label = QLabel("验证标签目录:")
        data_layout.addRow(self.val_labels_label, self.val_labels_layout)
        
        # Data YAML path
        self.data_yaml_layout = QHBoxLayout()
        self.data_yaml_path_edit = QLineEdit()
        self.data_yaml_path_edit.setReadOnly(True)
        self.data_yaml_btn = QPushButton("浏览...")
        self.data_yaml_open_btn = QPushButton("📁")
        self.data_yaml_open_btn.setToolTip("打开数据配置文件所在目录")
        self.data_yaml_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.data_yaml_open_btn.clicked.connect(lambda: self.open_folder(os.path.dirname(self.data_yaml_path_edit.text()) if self.data_yaml_path_edit.text() else ""))
        self.data_yaml_layout.addWidget(self.data_yaml_path_edit)
        self.data_yaml_layout.addWidget(self.data_yaml_btn)
        self.data_yaml_layout.addWidget(self.data_yaml_open_btn)
        data_layout.addRow("数据配置文件:", self.data_yaml_layout)
        
        self.data_group.setLayout(data_layout)

        # Model section
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        model_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow) # Ensure fields expand

        # Model type selection (e.g., yolov8n.pt)
        self.model_combo = QComboBox()
        # self.model_combo.currentTextChanged.connect(self.on_model_selection_changed) # Connection moved to connect_signals
        model_layout.addRow(QLabel("模型类型:"), self.model_combo)
        
        # Device selection
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "GPU (CUDA:0)", "GPU (CUDA:1)", "GPU (CUDA:2)", "GPU (CUDA:3)"])
        model_layout.addRow(QLabel("设备:"), self.device_combo)

        # Hyperparameters text edit
        self.hyperparameters_edit = QTextEdit()
        self.hyperparameters_edit.setPlaceholderText("输入额外的超参数，每行一个，格式为 key=value\n例如:\nlr0=0.01\nmomentum=0.937\nweight_decay=0.0005")
        self.hyperparameters_edit.setMaximumHeight(100)
        model_layout.addRow(QLabel("额外超参数:"), self.hyperparameters_edit)

        # Model Source Options (Radio Buttons)
        model_source_box = QGroupBox("模型来源")
        model_source_box_layout = QVBoxLayout() # Changed to QVBoxLayout for better spacing if needed

        self.model_source_group = QButtonGroup(self)

        self.download_model_radio = QRadioButton("下载官方预训练模型")
        self.download_model_radio.setToolTip("从 Ultralytics 下载所选类型的官方预训练模型。")
        self.model_source_group.addButton(self.download_model_radio)
        model_source_box_layout.addWidget(self.download_model_radio)
        
        # 模型状态和下载控制（仅在下载模式时显示）
        download_status_layout = QHBoxLayout()
        download_status_layout.setContentsMargins(20, 0, 0, 0)  # 左边距缩进
        
        # 模型状态标签
        self.model_status_label = QLabel("✓ 模型可用")
        self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.model_status_label.setVisible(False)
        download_status_layout.addWidget(self.model_status_label)
        
        # 下载按钮
        self.download_model_btn = QPushButton("下载模型")
        self.download_model_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        self.download_model_btn.setVisible(False)
        self.download_model_btn.clicked.connect(self.download_selected_model)
        download_status_layout.addWidget(self.download_model_btn)
        
        # 打开模型目录按钮
        self.open_model_folder_btn = QPushButton("📁 打开模型目录")
        self.open_model_folder_btn.setToolTip("打开模型缓存文件夹")
        self.open_model_folder_btn.clicked.connect(self.open_model_cache_folder)
        download_status_layout.addWidget(self.open_model_folder_btn)
        
        download_status_layout.addStretch()  # 右侧弹性空间
        model_source_box_layout.addLayout(download_status_layout)
        
        # 模型检查状态计时器
        self.model_check_timer = QTimer()
        self.model_check_timer.setSingleShot(True)
        self.model_check_timer.timeout.connect(self.check_selected_model_status)

        self.local_folder_model_radio = QRadioButton("从本地文件夹选择预训练模型")
        self.local_folder_model_radio.setToolTip("从您指定的本地文件夹中加载所选类型的预训练模型。")
        self.model_source_group.addButton(self.local_folder_model_radio)
        model_source_box_layout.addWidget(self.local_folder_model_radio)
        
        self.local_model_folder_layout = QHBoxLayout()
        self.local_model_folder_edit = QLineEdit()
        self.local_model_folder_edit.setPlaceholderText("选择包含模型的文件夹")
        self.local_model_folder_edit.setReadOnly(True)
        self.local_model_folder_btn = QPushButton("浏览...")
        self.local_model_folder_open_btn = QPushButton("📁")
        self.local_model_folder_open_btn.setToolTip("打开本地模型文件夹")
        self.local_model_folder_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.local_model_folder_open_btn.clicked.connect(lambda: self.open_folder(self.local_model_folder_edit.text()))
        self.local_model_folder_layout.addWidget(self.local_model_folder_edit)
        self.local_model_folder_layout.addWidget(self.local_model_folder_btn)
        self.local_model_folder_layout.addWidget(self.local_model_folder_open_btn)
        model_source_box_layout.addLayout(self.local_model_folder_layout)

        self.custom_weights_radio = QRadioButton("使用自定义权重文件 (.pt)")
        self.custom_weights_radio.setToolTip("加载一个特定的 .pt 权重文件，用于继续训练或使用自定义模型。")
        self.model_source_group.addButton(self.custom_weights_radio)
        model_source_box_layout.addWidget(self.custom_weights_radio)

        self.custom_model_path_layout = QHBoxLayout()
        self.custom_model_path_edit = QLineEdit()
        self.custom_model_path_edit.setPlaceholderText("选择 .pt 模型文件")
        self.custom_model_path_edit.setReadOnly(True)
        self.custom_model_path_btn = QPushButton("浏览...")
        self.custom_model_path_open_btn = QPushButton("📁")
        self.custom_model_path_open_btn.setToolTip("打开自定义模型文件所在目录")
        self.custom_model_path_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.custom_model_path_open_btn.clicked.connect(lambda: self.open_folder(os.path.dirname(self.custom_model_path_edit.text()) if self.custom_model_path_edit.text() else ""))
        self.custom_model_path_layout.addWidget(self.custom_model_path_edit)
        self.custom_model_path_layout.addWidget(self.custom_model_path_btn)
        self.custom_model_path_layout.addWidget(self.custom_model_path_open_btn)
        model_source_box_layout.addLayout(self.custom_model_path_layout)
        
        model_source_box.setLayout(model_source_box_layout)
        model_layout.addRow(model_source_box)

        # Fine-tuning mode (Moved out of init_group_box for clarity with new structure)
        self.fine_tuning_mode = QCheckBox("微调模式（冻结骨干网络，仅训练检测头）")
        self.fine_tuning_mode.setChecked(False)
        # self.fine_tuning_mode.toggled.connect(self.update_fine_tuning_state) # Connection moved
        model_layout.addRow(self.fine_tuning_mode)

        # Model Initialization Options (Original: pretrained, scratch - This is now partially covered by source selection)
        # For simplicity, "pretrained" is implied by "Download" or "Local Folder" + a .pt model.
        # "From scratch" training needs to be handled.
        init_options_group = QGroupBox("训练方式")
        init_options_layout = QHBoxLayout()
        self.train_mode_group = QButtonGroup(self)

        self.use_selected_weights_radio = QRadioButton("使用选定权重 (下载/本地/自定义)") # New default
        self.use_selected_weights_radio.setChecked(True)
        self.train_mode_group.addButton(self.use_selected_weights_radio)
        init_options_layout.addWidget(self.use_selected_weights_radio)
        
        self.from_scratch_radio = QRadioButton("从头开始训练 (随机初始化)")
        self.train_mode_group.addButton(self.from_scratch_radio)
        init_options_layout.addWidget(self.from_scratch_radio)
        
        init_options_group.setLayout(init_options_layout)
        model_layout.addRow(init_options_group)

        # Hyperparameters
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 128)
        self.batch_size_spin.setValue(16)
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 1280)
        self.img_size_spin.setValue(640)
        self.img_size_spin.setSingleStep(32)
        
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.001)
        
        # Add widgets to form layout
        model_layout.addRow("模型:", self.model_combo)
        model_layout.addRow("批次大小:", self.batch_size_spin)
        model_layout.addRow("训练轮数:", self.epochs_spin)
        model_layout.addRow("图像尺寸:", self.img_size_spin)
        model_layout.addRow("学习率:", self.lr_spin)
        model_layout.addWidget(init_options_group)
        model_layout.addWidget(self.fine_tuning_mode)
        model_group.setLayout(model_layout)
        
        # Output section
        output_group = QGroupBox("输出")
        output_layout = QFormLayout()
        
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_open_btn = QPushButton("📁")
        self.output_dir_open_btn.setToolTip("打开输出目录")
        self.output_dir_open_btn.setFixedSize(32, 23)  # 固定大小确保显示完整
        self.output_dir_open_btn.clicked.connect(lambda: self.open_folder(self.output_dir_edit.text()))
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        self.output_dir_layout.addWidget(self.output_dir_open_btn)
        
        self.project_name_edit = QLineEdit("yolo_project")
        
        # Add widgets to form layout
        output_layout.addRow("输出目录:", self.output_dir_layout)
        output_layout.addRow("项目名称:", self.project_name_edit)
        output_group.setLayout(output_layout)
        
        # Control section
        control_layout = QHBoxLayout()
        self.validate_btn = QPushButton("验证数据")
        self.validate_btn.setMinimumHeight(40)
        self.start_btn = QPushButton("开始训练")
        self.start_btn.setMinimumHeight(40)
        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        
        control_layout.addWidget(self.validate_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        
        # Progress section
        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(200)
        self.log_text.setAcceptRichText(True)
        self.log_text.document().setDefaultStyleSheet("pre {margin: 0; padding: 0;}")
        
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.log_text)
        progress_group.setLayout(progress_layout)
        
        # Add all sections to main layout with proper spacing
        main_layout.addWidget(self.data_group)
        main_layout.addSpacing(10)
        main_layout.addWidget(model_group)
        main_layout.addSpacing(10)
        main_layout.addWidget(output_group)
        main_layout.addSpacing(10)
        main_layout.addLayout(control_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(progress_group)
        
        # Add stretch at the end to push everything up
        main_layout.addStretch()
        
        # Connect signals
        self.connect_signals()
        
        # For first initialization
        self.model_combo.setCurrentIndex(0)
        self.update_parameters_display()
    
    def connect_signals(self):
        """Connect UI signals to handlers."""
        self.train_images_btn.clicked.connect(lambda: self.select_directory("选择训练图像目录", self.train_images_edit))
        self.train_labels_btn.clicked.connect(lambda: self.select_directory("选择训练标签目录", self.train_labels_edit))
        self.val_images_btn.clicked.connect(lambda: self.select_directory("选择验证图像目录", self.val_images_edit))
        self.val_labels_btn.clicked.connect(lambda: self.select_directory("选择验证标签目录", self.val_labels_edit))
        self.output_dir_btn.clicked.connect(self.select_output_dir)
        
        self.model_combo.currentTextChanged.connect(self.on_model_selection_changed)
        
        # Model Source Radio Buttons
        self.download_model_radio.toggled.connect(self.on_model_source_changed)
        self.local_folder_model_radio.toggled.connect(self.on_model_source_changed)
        self.custom_weights_radio.toggled.connect(self.on_model_source_changed)

        # Browse buttons for model paths
        self.local_model_folder_btn.clicked.connect(self.select_local_model_folder)
        self.custom_model_path_btn.clicked.connect(self.select_custom_model_file)

        # Training Mode Radio Buttons
        self.use_selected_weights_radio.toggled.connect(self.on_train_mode_changed)
        self.from_scratch_radio.toggled.connect(self.on_train_mode_changed)

        # Fine-tuning checkbox
        self.fine_tuning_mode.toggled.connect(self.update_fine_tuning_state)

        # Start/stop training buttons
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        
        # Connect dataset validation button
        self.validate_btn.clicked.connect(self.validate_dataset)
        
        # Task type combo
        self.task_combo.currentIndexChanged.connect(self.on_task_type_changed)

        # Data.yaml 浏览按钮
        self.data_yaml_btn.clicked.connect(self.on_data_yaml_btn_clicked)

    def on_data_yaml_btn_clicked(self):
        """选择数据集根目录后自动生成data.yaml"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集根目录（images/labels的上级目录）")
        if dir_path:
            # 自动推断images/train, images/val, labels/train, labels/val
            images_train = os.path.join(dir_path, "images", "train")
            images_val = os.path.join(dir_path, "images", "val")
            labels_train = os.path.join(dir_path, "labels", "train")
            labels_val = os.path.join(dir_path, "labels", "val")
            # 检查目录是否存在
            missing = []
            for p, name in zip([images_train, images_val, labels_train, labels_val],
                               ["images/train", "images/val", "labels/train", "labels/val"]):
                if not os.path.isdir(p):
                    missing.append(name)
            if missing:
                QMessageBox.warning(self, "目录缺失", f"以下目录不存在，请检查数据集结构：\n" + "\n".join(missing))
                return
            self.train_images_edit.setText(images_train)
            self.val_images_edit.setText(images_val)
            self.train_labels_edit.setText(labels_train)
            self.val_labels_edit.setText(labels_val)
            # 自动生成data.yaml
            self.try_create_data_yaml()
            # 若生成成功，data_yaml_path_edit会被自动填入
            if self.data_yaml_path_edit.text():
                QMessageBox.information(self, "data.yaml已生成", f"已生成配置文件: {self.data_yaml_path_edit.text()}\n请根据需要检查和修改。")

    def on_model_source_changed(self):
        """Handles changes in model source selection (download vs. local)."""
        self.update_model_source_ui_state()

    def on_train_mode_changed(self, checked=None):
        if checked is None or checked:
            self.update_model_source_ui_state() # Re-evaluate UI state, especially for fine-tuning

    def select_output_dir(self):
        self.select_directory("选择输出目录", self.output_dir_edit)

    def select_local_model_folder(self):
        """Selects a folder containing local models."""
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择本地模型文件夹", 
            self.default_local_model_search_dir if self.default_local_model_search_dir else os.getcwd()
        )
        if dir_path:
            self.local_model_folder_edit.setText(dir_path)
            self.default_local_model_search_dir = dir_path # Update default path for next time
            # Optionally, you could try to find a model in this folder that matches self.model_combo.currentText()
            # and update an internal variable for the model path.

    def select_custom_model_file(self):
        """Selects a custom model file (.pt)."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择自定义模型权重文件 (.pt)", 
            self.default_model_path, 
            "PyTorch Model Files (*.pt)"
        )
        if file_path:
            self.custom_model_path_edit.setText(file_path)
            self.default_model_path = os.path.dirname(file_path) # Update default dir for next time

    def start_training(self):
        if self.is_training:
            QMessageBox.warning(self, "警告", "训练已在进行中。")
            return

        if not self.validate_inputs():
            return

        # Determine the model weights to use
        model_weights = ""
        train_from_scratch = self.from_scratch_radio.isChecked()

        if not train_from_scratch:
            if self.download_model_radio.isChecked():
                selected_model_name = self.model_combo.currentText()
                if not selected_model_name.endswith((".pt", ".pth")):
                    selected_model_name += ".pt"
                model_weights = selected_model_name
                self.log_message(f"准备下载模型: {model_weights}")
            elif self.local_folder_model_radio.isChecked():
                folder_path = self.local_model_folder_edit.text()
                selected_model_name = self.model_combo.currentText()
                if not selected_model_name.endswith((".pt", ".pth")):
                    selected_model_name += ".pt"
                potential_path = os.path.join(folder_path, selected_model_name)
                if os.path.isfile(potential_path):
                    model_weights = potential_path
                    self.log_message(f"使用本地模型: {model_weights}")
                else:
                    QMessageBox.warning(self, "错误", f"在指定文件夹中未找到模型文件: {selected_model_name}")
                    return
            elif self.custom_weights_radio.isChecked():
                model_weights = self.custom_model_path_edit.text()
                if not model_weights or not os.path.isfile(model_weights):
                    QMessageBox.warning(self, "错误", "自定义权重文件路径无效或未选择。")
                    return
                self.log_message(f"使用自定义权重: {model_weights}")
        else:
            model_weights = self.model_type.replace(".pt", "")
            self.log_message(f"从头开始训练模型: {model_weights} (使用相应配置)")

        self.is_training = True
        self.set_ui_enabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.log_message("训练开始...")

        # Get parameters
        # 分类任务下data参数为训练集根目录，检测任务下为yaml
        if self.task_type == "classify":
            data_yaml_path = self.train_images_edit.text()
        else:
            data_yaml_path = self.data_yaml_path_edit.text()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_size_spin.value()
        img_size = self.img_size_spin.value()
        output_dir = self.output_dir_edit.text()
        # 修正device解析逻辑
        device_text = self.device_combo.currentText()
        if "CPU" in device_text:
            device = "cpu"
        elif "GPU" in device_text:
            import re
            m = re.search(r"CUDA:(\\d+)", device_text)
            if m:
                device = m.group(1)
            else:
                device = ""  # fallback
        else:
            device = ""  # fallback
        task = self.task_type
        
        # Hyperparameters from text edit
        try:
            hyperparameters_str = self.hyperparameters_edit.toPlainText() if hasattr(self, 'hyperparameters_edit') else ""
            other_args = self.parse_hyperparameters(hyperparameters_str)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"解析超参数时出错: {e}")
            self.set_ui_enabled(True)
            self.is_training = False
            return
            
        freeze_backbone = self.fine_tuning_mode.isChecked() if task == "detect" else False
        if task == "classify" and self.fine_tuning_mode.isChecked():
            other_args['freeze'] = 10

        # 创建训练工作线程
        self.training_thread = QThread()
        self.training_worker = TrainingWorker(
            model_name=model_weights,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            batch_size=batch_size,
            img_size=img_size,
            output_dir=output_dir,
            device=device,
            task=task,
            is_from_scratch=train_from_scratch,
            freeze_backbone=freeze_backbone,
            other_args=other_args,
            model_source_option=self.model_source_option,
            local_model_search_dir=self.local_model_folder_edit.text() if self.local_folder_model_radio.isChecked() else None,
            project_name=self.project_name_edit.text()
        )

        # 将worker移动到线程
        self.training_worker.moveToThread(self.training_thread)

        # 连接信号
        self.training_worker.progress_update.connect(self.progress_bar.setValue)
        self.training_worker.log_update.connect(self.log_message)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.training_error.connect(self.on_training_error)

        # 连接线程启动信号到worker的run方法
        self.training_thread.started.connect(self.training_worker.run)

        # 启动线程
        self.training_thread.start()
        self.log_message("训练线程已启动...")

    def stop_training(self):
        """Stop the training process."""
        self.log_message("Stopping training (please wait)...")
        if self.training_worker:
            self.training_worker.stop()
    
    def on_training_complete(self):
        """Handle training completion."""
        self.is_training = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        self.log_message("训练成功完成！")
        QMessageBox.information(self, "训练完成", "训练已成功完成。")
    
    def on_training_error(self, error_msg):
        """Handle training error."""
        self.is_training = False
        self.clean_up_thread()
        self.set_ui_enabled(True)
        self.log_message(f"错误: {error_msg}")
        QMessageBox.critical(self, "训练错误", f"训练过程中发生错误:\n{error_msg}")
    
    def clean_up_thread(self):
        """Clean up thread and worker resources."""
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()
            self.training_thread = None
            self.training_worker = None
    
    def update_progress(self, progress):
        """Update progress bar."""
        self.progress_bar.setValue(progress)
    
    def log_message(self, message):
        """Add a message to the log display."""
        # 检查消息类型
        if "Epoch" in message and ("GPU_mem" in message or "box_loss" in message):
            # 如果是训练进度信息，使用特殊格式
            self.log_text.append(f"\n<span style='color:#0066CC; font-family:Courier;'>{message}</span>")
            # 确保光标可见
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.End)
            self.log_text.setTextCursor(cursor)
            # 自动滚动到底部
            self.log_text.ensureCursorVisible()
        else:
            # 普通消息
            self.log_text.append(message)
        
        # Also print to stdout for terminal redirection
        print(f"[Training] {message}")
    
    def set_ui_enabled(self, enabled):
        """Enable or disable UI elements during training."""
        self.start_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(not enabled)
        self.train_images_btn.setEnabled(enabled)
        self.train_images_open_btn.setEnabled(enabled)
        self.train_labels_btn.setEnabled(enabled)
        self.train_labels_open_btn.setEnabled(enabled)
        self.val_images_btn.setEnabled(enabled)
        self.val_images_open_btn.setEnabled(enabled)
        self.val_labels_btn.setEnabled(enabled)
        self.val_labels_open_btn.setEnabled(enabled)
        self.data_yaml_btn.setEnabled(enabled)
        self.data_yaml_open_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        self.output_dir_open_btn.setEnabled(enabled)
        
        # Model path button should only be enabled if custom weights is checked
        model_path_enabled = enabled and self.custom_weights_radio.isChecked()
        self.custom_model_path_btn.setEnabled(model_path_enabled)
        self.custom_model_path_edit.setEnabled(model_path_enabled)
        self.custom_model_path_open_btn.setEnabled(model_path_enabled)
        
        # Local model folder buttons
        local_folder_enabled = enabled and self.local_folder_model_radio.isChecked()
        self.local_model_folder_btn.setEnabled(local_folder_enabled)
        self.local_model_folder_edit.setEnabled(local_folder_enabled)
        self.local_model_folder_open_btn.setEnabled(local_folder_enabled)
        
        self.model_combo.setEnabled(enabled)
        self.batch_size_spin.setEnabled(enabled)
        self.epochs_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        
        # Radio buttons for model initialization
        self.use_selected_weights_radio.setEnabled(enabled)
        self.from_scratch_radio.setEnabled(enabled)
        self.custom_weights_radio.setEnabled(enabled)
        
        # Fine-tuning is only enabled if using pretrained or custom weights
        fine_tuning_enabled = enabled and (self.use_selected_weights_radio.isChecked() or self.custom_weights_radio.isChecked())
        self.fine_tuning_mode.setEnabled(fine_tuning_enabled)
        
        self.project_name_edit.setEnabled(enabled)
    
    def validate_inputs(self):
        """验证训练输入参数"""
        # 只在检测任务下要求yaml
        if self.task_type == "detect":
            if not self.data_yaml_path_edit.text():
                QMessageBox.warning(self, "错误", "请选择数据配置文件 (data.yaml)")
                return False
        # 分类任务不检查data_yaml

        # 验证输出目录
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "错误", "请选择输出目录")
            return False

        # 验证项目名称
        if not self.project_name_edit.text():
            QMessageBox.warning(self, "错误", "请输入项目名称")
            return False

        # 验证模型选择
        if not self.model_combo.currentText():
            QMessageBox.warning(self, "错误", "请选择模型类型")
            return False

        # 验证模型来源
        if self.download_model_radio.isChecked():
            # 下载模式不需要额外验证
            pass
        elif self.local_folder_model_radio.isChecked():
            if not self.local_model_folder_edit.text():
                QMessageBox.warning(self, "错误", "请选择本地模型文件夹")
                return False
        elif self.custom_weights_radio.isChecked():
            if not self.custom_model_path_edit.text():
                QMessageBox.warning(self, "错误", "请选择自定义权重文件")
                return False
            if not os.path.isfile(self.custom_model_path_edit.text()):
                QMessageBox.warning(self, "错误", "自定义权重文件不存在")
                return False

        # 验证训练参数
        if self.epochs_spin.value() <= 0:
            QMessageBox.warning(self, "错误", "训练轮数必须大于0")
            return False
        if self.batch_size_spin.value() <= 0:
            QMessageBox.warning(self, "错误", "批次大小必须大于0")
            return False
        if self.img_size_spin.value() <= 0:
            QMessageBox.warning(self, "错误", "图像尺寸必须大于0")
            return False

        # 验证数据集路径
        if self.task_type == "detect":
            if not self.train_images_edit.text():
                QMessageBox.warning(self, "错误", "请选择训练图像目录")
                return False
            if not self.train_labels_edit.text():
                QMessageBox.warning(self, "错误", "请选择训练标签目录")
                return False
            if not self.val_images_edit.text():
                QMessageBox.warning(self, "错误", "请选择验证图像目录")
                return False
            if not self.val_labels_edit.text():
                QMessageBox.warning(self, "错误", "请选择验证标签目录")
                return False
        else:  # classify
            if not self.train_images_edit.text():
                QMessageBox.warning(self, "错误", "请选择训练集根目录")
                return False
            if not self.val_images_edit.text():
                QMessageBox.warning(self, "错误", "请选择验证集根目录")
                return False

        return True
    
    def update_settings(self, settings):
        """Update UI elements with loaded settings."""
        self.default_train_dir = settings.get('default_train_dir', os.getcwd())
        self.default_val_dir = settings.get('default_val_dir', os.getcwd())
        self.default_output_dir = settings.get('default_output_dir', os.getcwd())
        self.default_model_path = settings.get('default_model_path', os.getcwd()) # For custom .pt
        self.default_local_model_search_dir = settings.get('default_local_model_search_dir', os.getcwd())

        self.train_images_edit.setText(settings.get('train_images_path', self.default_train_dir))
        # ... (other settings for paths)
        self.data_yaml_path_edit.setText(settings.get('data_yaml_path', ''))
        self.output_dir_edit.setText(settings.get('output_dir', self.default_output_dir))

        # Update model combo selection
        saved_model_type = settings.get('model_type', 'yolov8n.pt')
        if self.model_combo.findText(saved_model_type) != -1:
            self.model_combo.setCurrentText(saved_model_type)
        else:
            self.model_combo.setCurrentIndex(0) # Fallback

        # Update model source option and related paths
        model_source = settings.get('model_source_option', 'download')
        if model_source == 'local_folder':
            self.local_folder_model_radio.setChecked(True)
            self.local_model_folder_edit.setText(settings.get('local_model_folder_path', self.default_local_model_search_dir))
            self.custom_model_path_edit.clear()
        elif model_source == 'custom_file':
            self.custom_weights_radio.setChecked(True)
            self.custom_model_path_edit.setText(settings.get('custom_model_file_path', self.default_model_path))
            self.local_model_folder_edit.clear()
        else: # download
            self.download_model_radio.setChecked(True)
            self.local_model_folder_edit.clear()
            self.custom_model_path_edit.clear()
        
        # Training mode (selected weights vs from scratch)
        train_mode_val = settings.get('train_mode', 'selected_weights')
        if train_mode_val == 'from_scratch':
            self.from_scratch_radio.setChecked(True)
        else:
            self.use_selected_weights_radio.setChecked(True)

        self.fine_tuning_mode.setChecked(settings.get('fine_tuning_mode', False))
        
        # Hyperparameters
        self.batch_size_spin.setValue(settings.get('batch_size', 16))
        self.epochs_spin.setValue(settings.get('epochs', 100))
        self.img_size_spin.setValue(settings.get('img_size', 640))
        # ... (other hyperparams like learning rate etc.)

        # Device setting
        saved_device = settings.get('device', 'cpu')
        device_index = self.device_combo.findText(saved_device, Qt.MatchContains)
        if device_index != -1:
            self.device_combo.setCurrentIndex(device_index)
        
        self.update_model_source_ui_state() # IMPORTANT: Update UI based on loaded settings
        self.update_fine_tuning_state()
        
        # 检查模型状态
        self.check_selected_model_status()

    def update_model_source_ui_state(self):
        """Updates UI elements based on the selected model source and train mode."""
        is_download = self.download_model_radio.isChecked()
        is_local_folder = self.local_folder_model_radio.isChecked()
        is_custom_file = self.custom_weights_radio.isChecked()
        is_from_scratch = self.from_scratch_radio.isChecked()

        # Enable/Disable Local Model Folder selection
        self.local_model_folder_edit.setEnabled(is_local_folder and not is_from_scratch)
        self.local_model_folder_btn.setEnabled(is_local_folder and not is_from_scratch)
        self.local_model_folder_open_btn.setEnabled(is_local_folder and not is_from_scratch)
        if not (is_local_folder and not is_from_scratch):
            self.local_model_folder_edit.clear() # Clear if not active

        # Enable/Disable Custom Model File selection
        self.custom_model_path_edit.setEnabled(is_custom_file and not is_from_scratch)
        self.custom_model_path_btn.setEnabled(is_custom_file and not is_from_scratch)
        if not (is_custom_file and not is_from_scratch):
            self.custom_model_path_edit.clear() # Clear if not active
            
        # Model combo is always enabled, but its meaning changes slightly
        # Fine tuning mode might depend on whether we are training from scratch or not
        self.fine_tuning_mode.setEnabled(not is_from_scratch and self.task_type == "detect") # Or general logic

        # Update placeholder texts
        if is_download and not is_from_scratch:
            self.model_combo.setToolTip("选择要自动下载的官方预训练模型。")
            self.local_model_folder_edit.setPlaceholderText("通过上方选择'从本地文件夹...'")
            self.custom_model_path_edit.setPlaceholderText("通过上方选择'使用自定义权重文件...'")
        elif is_local_folder and not is_from_scratch:
            self.model_combo.setToolTip("选择模型类型，然后在下方指定包含该类型模型的文件夹。")
            self.local_model_folder_edit.setPlaceholderText("选择包含所选类型模型的本地文件夹")
        elif is_custom_file and not is_from_scratch:
            self.model_combo.setToolTip("模型类型(仅参考), 将使用下方指定的.pt文件。") # Model combo becomes less critical here
            self.custom_model_path_edit.setPlaceholderText("选择您的自定义 .pt 模型文件")
        elif is_from_scratch:
            self.model_combo.setToolTip("选择要从零开始训练的模型架构 (例如 yolov8n, yolov8s-cls)。")
            self.local_model_folder_edit.setEnabled(False)
            self.local_model_folder_btn.setEnabled(False)
            self.custom_model_path_edit.setEnabled(False)
            self.custom_model_path_btn.setEnabled(False)

        # Store the current selection
        if is_download:
            self.model_source_option = "download"
        elif is_local_folder:
            self.model_source_option = "local_folder"
        elif is_custom_file:
            self.model_source_option = "custom_file"
        
        if is_from_scratch:
            self.train_mode = "scratch"
        else:
            self.train_mode = "pretrained" # or 'custom' depending on source

        # Update fine-tuning state which depends on whether a model is loaded/selected
        self.update_fine_tuning_state()
        
        # 控制"打开模型目录"按钮的显示
        self.open_model_folder_btn.setVisible(is_download and not is_from_scratch)
        
        # 检查模型状态
        self.check_selected_model_status()

    def update_fine_tuning_state(self, checked=None):
        """Enable/Disable fine-tuning checkbox based on current settings."""
        # Fine-tuning is usually applicable when using pre-trained weights and for detection tasks.
        # It might not make sense or have a different meaning "from scratch".
        can_fine_tune = not self.from_scratch_radio.isChecked() and self.task_type == "detect"
        self.fine_tuning_mode.setEnabled(can_fine_tune)
        if not can_fine_tune:
            self.fine_tuning_mode.setChecked(False) # Uncheck if disabled
        
        # Update the text based on task type, even if disabled, to be informative
        if self.task_type == "detect":
            self.fine_tuning_mode.setText("微调模式 (冻结主干网络，仅训练检测头)")
        elif self.task_type == "classify":
            # For classification, ultralytics uses 'freeze' argument (e.g., freeze=10 for first 10 layers)
            self.fine_tuning_mode.setText("微调模式 (例如，冻结分类模型的部分层)") 
            # self.fine_tuning_mode.setEnabled(not self.from_scratch_radio.isChecked()) # Classifier fine-tuning might be possible
        else: # segment, pose, etc.
            self.fine_tuning_mode.setText("微调模式 (特定于任务)")
            # self.fine_tuning_mode.setEnabled(not self.from_scratch_radio.isChecked())

    def select_directory(self, title, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, title)
        if dir_path:
            dir_path = dir_path.strip('\'"')  # 自动去除首尾引号
            line_edit.setText(dir_path)
        # 不再自动同步到settings_tab，也不自动保存设置

    def validate_dataset(self):
        """Validate the dataset structure and image-label matching."""
        # Get the directory paths
        train_images_dir = self.train_images_edit.text()
        val_images_dir = self.val_images_edit.text()

        if not train_images_dir:
            QMessageBox.warning(self, "缺少路径", "请先选择训练数据目录")
            return

        self.log_message(f"开始验证 {self.task_type} 数据集...")

        if self.task_type == "detect":
            train_labels_dir = self.train_labels_edit.text()
            val_labels_dir = self.val_labels_edit.text()
            if not train_labels_dir:
                QMessageBox.warning(self, "缺少路径", "目标检测任务请提供训练标签目录。")
                return
            
            train_results = validate_yolo_dataset(train_images_dir, train_labels_dir)
            self.log_message(f"训练数据集 (检测) 验证: {train_results['message']}")
            if val_images_dir and not val_labels_dir:
                QMessageBox.warning(self, "缺少路径", "目标检测任务请为验证集提供标签目录。")
                return
            if val_images_dir and val_labels_dir:
                val_results = validate_yolo_dataset(val_images_dir, val_labels_dir)
                self.log_message(f"验证数据集 (检测) 验证: {val_results['message']}")
            else:
                val_results = {"success": True} # No validation set to check against labels
            
            # Inspect dataset structure for more detailed information
            # This part might need adjustment if train_images_dir is not the root of images/labels structure
            # base_dir_train = os.path.dirname(os.path.dirname(train_images_dir)) if 'images' in train_images_dir else os.path.dirname(train_images_dir)
            # structure_report = inspect_dataset_structure(base_dir_train) # Assuming inspect_dataset_structure works with this base
            # self.log_message("\n数据集结构分析 (检测):\n" + structure_report)

            if train_results["success"] and val_results["success"]:
                QMessageBox.information(self, "验证成功", "目标检测数据集验证通过。")
            else:
                QMessageBox.warning(self, "验证问题", "目标检测数据集验证失败，请检查日志。")

        elif self.task_type == "classify":
            # For classification, we check if train_images_dir (and val_images_dir if provided)
            # contain subdirectories (which represent classes).
            valid_train = False
            if os.path.isdir(train_images_dir):
                subdirs = [d for d in os.listdir(train_images_dir) if os.path.isdir(os.path.join(train_images_dir, d))]
                if subdirs:
                    self.log_message(f"训练数据集 (分类) 验证: 在 {train_images_dir} 中找到 {len(subdirs)} 个可能的类别子文件夹: {subdirs}")
                    valid_train = True
                else:
                    self.log_message(f"训练数据集 (分类) 验证错误: {train_images_dir} 中未找到类别子文件夹。")
            else:
                self.log_message(f"训练数据集 (分类) 验证错误: {train_images_dir} 不是一个有效的目录。")

            valid_val = True # Assume valid if not provided
            if val_images_dir:
                valid_val = False
                if os.path.isdir(val_images_dir):
                    subdirs_val = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
                    if subdirs_val:
                        self.log_message(f"验证数据集 (分类) 验证: 在 {val_images_dir} 中找到 {len(subdirs_val)} 个可能的类别子文件夹: {subdirs_val}")
                        valid_val = True
                    else:
                        self.log_message(f"验证数据集 (分类) 验证错误: {val_images_dir} 中未找到类别子文件夹。")
                else:
                    self.log_message(f"验证数据集 (分类) 验证错误: {val_images_dir} 不是一个有效的目录。")
            
            if valid_train and valid_val:
                QMessageBox.information(self, "验证成功", "图像分类数据集结构初步检查通过。")
            else:
                QMessageBox.warning(self, "验证问题", "图像分类数据集结构检查失败，请确保目录包含类别子文件夹，并检查日志。")
        
    def update_parameters_display(self):
        """Update UI parameters based on selected model."""
        model = self.model_combo.currentText()
        
        # Adjust batch size based on model size
        if model.endswith('n'):  # nano models
            self.batch_size_spin.setValue(16)
        elif model.endswith('s'):  # small models
            self.batch_size_spin.setValue(16)
        elif model.endswith('m'):  # medium models
            self.batch_size_spin.setValue(8)
        elif model.endswith('l'):  # large models
            self.batch_size_spin.setValue(8)
        elif model.endswith('x'):  # extra large models
            self.batch_size_spin.setValue(4)
        
        # Log model change # This log is now slightly different as self.model_type won't have .pt yet
        # self.log_message(f"已选择模型: {model}") # model here is raw from combobox
        
        # Update fine-tuning state in case model changed
        self.update_fine_tuning_state() 

    def on_model_selection_changed(self, model_name):
        self.model_type = model_name # e.g. yolov8n.pt or yolov8n (if from scratch)
        self.log_message(f"模型类型更改为: {model_name}")
        
        # 延迟检查模型状态，避免UI阻塞
        self.model_check_timer.stop()
        self.model_check_timer.start(500)  # 500ms延迟检查
        
        # No direct action here, state is handled by update_model_source_ui_state and start_training

    def update_model_list(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        
        yolo_versions = ["8", "9", "10", "11"]  # 恢复YOLO11支持，已确认存在
        yolo_sizes = ["n", "s", "m", "l", "x"]
        
        models = []

        if self.task_type == "detect":
            # Common detection models
            for v in yolo_versions:
                for s in yolo_sizes:
                    models.append(f"yolov{v}{s}")
            # Example: add OBB models if needed later
            # models.extend([f"yolov{v}{s}-obb" for v in yolo_versions for s in yolo_sizes])

        elif self.task_type == "classify":
            # Common classification models
            for v in yolo_versions:
                for s in yolo_sizes:
                    models.append(f"yolov{v}{s}-cls")
            models.extend(["resnet18", "resnet34", "resnet50", "resnet101"]) # Keep other common classification backbones
        else:
            models = [] # Should not happen if task_type is always 'detect' or 'classify'

        self.model_combo.addItems(models)
        if models:
             # Set default model based on task type
            if self.task_type == "detect":
                default_model_base = "yolov8n"
            elif self.task_type == "classify":
                default_model_base = "yolov8n-cls"
            else:
                default_model_base = models[0]

            if default_model_base in models:
                self.model_combo.setCurrentText(default_model_base)
                # self.model_type is updated by on_model_selection_changed via setCurrentText signal
            else: # Fallback if default_model_base is not in the list (e.g. empty models list)
                if models: # Ensure models list is not empty
                    self.model_combo.setCurrentIndex(0)
                # self.model_type will be updated by on_model_selection_changed

        self.model_combo.blockSignals(False)
        # Trigger on_model_selection_changed to set self.model_type correctly for the initially selected/default model
        if self.model_combo.count() > 0:
            # When list updates, on_model_selection_changed will be called by setCurrentText or currentIndex change.
            # Explicitly calling it here ensures self.model_type is set even if the first item doesn't change text.
            self.on_model_selection_changed(self.model_combo.currentText())
        else:
            self.model_type = "" # No models available
    
    def on_task_type_changed(self, index):
        self.task_type = "detect" if index == 0 else "classify"
        self.update_task_specific_ui()
        self.update_model_list()  # Update model list based on task type
        self.update_parameters_display() # Update displayed parameters for the new task
        # 弹窗提示
        if self.task_type == "classify":
            QMessageBox.information(self, "分类任务数据集结构说明",
                "分类任务数据集要求如下：\n\n"
                "1. 训练集根目录下，每个类别为一个子文件夹，子文件夹名即为类别名。\n"
                "2. 每个类别子文件夹内放置该类别的所有图片。\n"
                "3. 验证集同理。\n\n"
                "示例：\n"
                "train/\n  cat/\n    img1.jpg\n    img2.jpg\n  dog/\n    img3.jpg\n    img4.jpg\n"
            )
        else:
            QMessageBox.information(self, "目标检测任务数据集结构说明",
                "目标检测任务数据集要求如下：\n\n"
                "1. 训练/验证集分别有 images 和 labels 两个文件夹。\n"
                "2. images/ 下为图片，labels/ 下为同名 txt 文件（YOLO格式）。\n"
                "3. 根目录需有 data.yaml 配置文件。\n\n"
                "示例：\n"
                "dataset/\n  images/\n    train/\n      xxx.jpg\n    val/\n      yyy.jpg\n  labels/\n    train/\n      xxx.txt\n    val/\n      yyy.txt\n  data.yaml\n"
                "\n如未检测到 data.yaml，将自动为你生成。"
            )
            # 自动生成data.yaml
            self.try_create_data_yaml()

    def try_create_data_yaml(self):
        """在检测任务下，若未检测到data.yaml则自动生成一个模板。"""
        if self.task_type != "detect":
            return
        # 推断根目录
        train_dir = self.train_images_edit.text()
        val_dir = self.val_images_edit.text()
        if not train_dir or not val_dir:
            return
        # 推断根目录（假设train/val都在images/下）
        root_dir = os.path.commonpath([train_dir, val_dir])
        yaml_path = os.path.join(os.path.dirname(root_dir), "data.yaml")
        if os.path.exists(yaml_path):
            return
        # 优先查找classes.txt
        label_dir = self.train_labels_edit.text()
        names = []
        classes_txt_path = None
        # 1. labels目录下
        if label_dir and os.path.isdir(label_dir):
            possible = os.path.join(label_dir, "classes.txt")
            if os.path.isfile(possible):
                classes_txt_path = possible
        # 2. 根目录下
        if not classes_txt_path:
            possible = os.path.join(os.path.dirname(root_dir), "classes.txt")
            if os.path.isfile(possible):
                classes_txt_path = possible
        # 读取类别名
        if classes_txt_path:
            with open(classes_txt_path, 'r', encoding='utf-8') as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            # 遍历所有txt，收集类别id
            class_ids = set()
            if label_dir and os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.endswith('.txt'):
                        with open(os.path.join(label_dir, fname), 'r') as f:
                            for line in f:
                                if line.strip():
                                    class_id = line.split()[0]
                                    if class_id.isdigit():
                                        class_ids.add(int(class_id))
                if class_ids:
                    names = [f'class{i}' for i in sorted(class_ids)]
        data_yaml = {
            'train': train_dir.replace('\\', '/'),
            'val': val_dir.replace('\\', '/'),
            'nc': len(names) if names else 1,
            'names': names if names else ['class0']
        }
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(data_yaml, f, allow_unicode=True)
        self.data_yaml_path_edit.setText(yaml_path)
        QMessageBox.information(self, "已自动生成data.yaml", f"已在{os.path.dirname(root_dir)}生成data.yaml，类别名：{names if names else ['class0']}\n如需修改请手动编辑data.yaml或classes.txt。")

    def update_task_specific_ui(self):
        """Update UI elements specific to detection or classification tasks."""
        is_detection = self.task_type == "detect"
        self.data_group.setTitle("数据集 (目标检测)" if is_detection else "数据集 (图像分类)")
        self.train_images_label.setText("训练图像目录:" if is_detection else "训练集根目录 (包含类别子文件夹):")
        self.train_labels_label.setText("训练标签目录:" if is_detection else "训练标签目录 (自动从文件夹结构推断):")
        self.train_labels_edit.setEnabled(is_detection)
        self.train_labels_btn.setEnabled(is_detection)
        self.train_labels_open_btn.setEnabled(is_detection)
        
        self.val_images_label.setText("验证图像目录:" if is_detection else "验证集根目录 (包含类别子文件夹):")
        self.val_labels_label.setText("验证标签目录:" if is_detection else "验证标签目录 (自动从文件夹结构推断):")
        self.val_labels_edit.setEnabled(is_detection)
        self.val_labels_btn.setEnabled(is_detection)
        self.val_labels_open_btn.setEnabled(is_detection)
        
        # Enable fine-tuning for detection
        self.fine_tuning_mode.setEnabled(is_detection)
        self.fine_tuning_mode.setText("微调模式（冻结主干网络，仅训练检测头）" if is_detection else "微调模式（例如，冻结部分层，仅训练分类器）")

        # This will trigger an update to the model list based on the new task type
        self.update_model_list()

    def parse_hyperparameters(self, hyperparameters_str):
        """Parse hyperparameters from text input into a dictionary."""
        params = {}
        if not hyperparameters_str:
            return params
            
        for line in hyperparameters_str.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                try:
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    elif '.' in value:
                        value = float(value)
                    else:
                        value = int(value)
                except ValueError:
                    # Keep as string if conversion fails
                    pass
                    
                params[key] = value
            except ValueError:
                self.log_message(f"Warning: Invalid hyperparameter line: {line}")
                continue
                
        return params

    def check_selected_model_status(self):
        """检查选中模型的状态并更新UI"""
        if not self.model_combo.currentText():
            return
            
        model_name = self.model_combo.currentText()
        
        # 只在下载模式时显示状态
        if not self.download_model_radio.isChecked():
            self.model_status_label.setVisible(False)
            self.download_model_btn.setVisible(False)
            self.open_model_folder_btn.setVisible(False)
            return
            
        # 在下载模式时，总是显示"打开模型目录"按钮
        self.open_model_folder_btn.setVisible(True)
        
        # 检查模型是否存在
        model_exists = self.is_model_available(model_name)
        
        if model_exists:
            self.model_status_label.setText("✓ 模型可用")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.model_status_label.setVisible(True)
            self.download_model_btn.setVisible(False)
        else:
            # 检查版本兼容性
            model_name_with_ext = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            is_compatible, version_str, error_msg = check_ultralytics_version_compatibility(model_name_with_ext)
            
            if not is_compatible:
                if version_str == 'not_installed':
                    self.model_status_label.setText("❌ 需要安装ultralytics")
                    self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
                else:
                    # 获取版本信息
                    version_info = self.get_ultralytics_version_info(version_str)
                    self.model_status_label.setText(f"❌ 版本不兼容 (当前: {version_str})")
                    self.model_status_label.setStyleSheet("color: orange; font-weight: bold;")
                    self.model_status_label.setToolTip(f"版本信息:\n{version_info}")
                self.model_status_label.setVisible(True)
                self.download_model_btn.setVisible(False)
            else:
                self.model_status_label.setText("⬇ 模型需要下载")
                self.model_status_label.setStyleSheet("color: orange; font-weight: bold;")
                self.model_status_label.setVisible(True)
                self.download_model_btn.setVisible(True)

    def is_model_available(self, model_name):
        """检查模型是否在本地可用"""
        try:
            # 确保模型名有.pt扩展名
            model_name_with_ext = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
            
            # 检查本地缓存目录
            try:
                from utils.training_worker import DEFAULT_MODEL_CACHE_DIR
                cached_model_path = os.path.join(DEFAULT_MODEL_CACHE_DIR, model_name_with_ext)
                if os.path.exists(cached_model_path):
                    return True
            except ImportError:
                pass
                
            # 检查ultralytics默认缓存位置
            try:
                home_dir = os.path.expanduser("~")
                
                # 检查Ultralytics默认模型目录
                ultralytics_cache = os.path.join(home_dir, ".cache", "ultralytics")
                if os.path.exists(ultralytics_cache):
                    for file in os.listdir(ultralytics_cache):
                        if file == model_name_with_ext:
                            return True
                            
                # 检查torch hub缓存
                torch_cache = os.path.join(home_dir, ".cache", "torch", "hub")
                if os.path.exists(torch_cache):
                    for root, dirs, files in os.walk(torch_cache):
                        if model_name_with_ext in files:
                            return True
                            
                # 尝试直接用ultralytics检查模型是否可用（不触发下载）
                # 这里我们不实际创建YOLO实例，只检查缓存
                import ultralytics
                
                # 检查ultralytics的缓存目录结构
                possible_paths = [
                    os.path.join(home_dir, ".cache", "ultralytics", model_name_with_ext),
                    os.path.join(home_dir, ".ultralytics", "models", model_name_with_ext),
                    os.path.join(os.getcwd(), model_name_with_ext),
                ]
                
                for path in possible_paths:
                    if os.path.exists(path):
                        return True
                        
            except Exception as e:
                # 如果检查过程出错，假设模型不存在
                pass
                
            return False
            
        except Exception as e:
            # 如果检查过程出错，假设模型不存在
            return False

    def download_selected_model(self):
        """下载选中的模型"""
        if not self.model_combo.currentText():
            return
            
        model_name = self.model_combo.currentText()
        model_name_with_ext = model_name if model_name.endswith('.pt') else f"{model_name}.pt"
        
        # 禁用下载按钮并显示下载状态
        self.download_model_btn.setEnabled(False)
        self.download_model_btn.setText("下载中...")
        self.model_status_label.setText("⏳ 正在下载...")
        self.model_status_label.setStyleSheet("color: blue; font-weight: bold;")
        
        # 在子线程中下载模型
        self.start_model_download(model_name_with_ext)

    def start_model_download(self, model_name):
        """在子线程中开始模型下载"""
        from PyQt5.QtCore import QThread, QObject, pyqtSignal
        
        class ModelDownloadWorker(QObject):
            download_complete = pyqtSignal(bool, str)  # success, message
            log_update = pyqtSignal(str)  # 日志更新信号
            
            def __init__(self, model_name):
                super().__init__()
                self.model_name = model_name
                
            def get_ultralytics_version_info(self, current_version):
                """获取ultralytics版本信息和建议"""
                try:
                    import requests
                    import json
                    from packaging import version
                    
                    # 获取最新版本信息
                    try:
                        response = requests.get("https://pypi.org/pypi/ultralytics/json", timeout=5)
                        if response.status_code == 200:
                            data = response.json()
                            latest_version = data['info']['version']
                            
                            # 比较版本
                            try:
                                current_ver = version.parse(current_version)
                                latest_ver = version.parse(latest_version)
                                
                                if current_ver < latest_ver:
                                    version_status = f"⚠️ 有新版本可用: {latest_version}"
                                    version_suggestion = f"建议升级: pip install ultralytics=={latest_version}"
                                elif current_ver == latest_ver:
                                    version_status = f"✅ 已是最新版本: {latest_version}"
                                    version_suggestion = "版本是最新的，问题可能是网络或其他原因"
                                else:
                                    version_status = f"🚀 开发版本: {current_version} (最新发布: {latest_version})"
                                    version_suggestion = "使用的是开发版本"
                                    
                                # 检查YOLO11支持
                                yolo11_support = ""
                                if current_ver >= version.parse("8.3.0"):
                                    yolo11_support = "✅ 支持YOLO11"
                                else:
                                    yolo11_support = "❌ 不支持YOLO11 (需要>=8.3.0)"
                                    
                                return f"{version_status}\n最新发布版本: {latest_version}\n{yolo11_support}\n💡 {version_suggestion}"
                                
                            except Exception as ve:
                                return f"最新版本: {latest_version}\n⚠️ 版本比较失败: {str(ve)}"
                        else:
                            return f"⚠️ 无法获取最新版本信息 (HTTP {response.status_code})"
                            
                    except requests.RequestException as re:
                        return f"⚠️ 网络错误，无法获取最新版本: {str(re)[:100]}"
                        
                except ImportError:
                    # 如果没有requests或packaging库，提供基本信息
                    try:
                        from packaging import version
                        if version.parse(current_version) >= version.parse("8.3.0"):
                            return "✅ 当前版本应该支持YOLO11\n💡 建议检查网络连接或尝试手动下载"
                        else:
                            return "❌ 当前版本可能不支持YOLO11\n💡 建议升级: pip install --upgrade ultralytics"
                    except ImportError:
                        # 简单的字符串比较
                        if current_version.startswith('8.3') or current_version.startswith('8.4') or current_version.startswith('8.5'):
                            return "✅ 当前版本应该支持YOLO11\n💡 建议检查网络连接"
                        else:
                            return "⚠️ 建议升级到最新版本\n💡 运行: pip install --upgrade ultralytics"
                except Exception as e:
                    return f"⚠️ 版本检查失败: {str(e)}"
                
            def run(self):
                try:
                    self.log_update.emit(f"开始下载模型: {self.model_name}")
                    
                    # 导入必要的库
                    import ultralytics
                    from ultralytics import YOLO
                    import os
                    import tempfile
                    import requests
                    import shutil
                    
                    self.log_update.emit(f"使用ultralytics版本: {ultralytics.__version__}")
                    
                    # 确保模型名格式正确
                    if not self.model_name.endswith('.pt'):
                        model_name_with_ext = f"{self.model_name}.pt"
                    else:
                        model_name_with_ext = self.model_name
                    
                    self.log_update.emit(f"正在下载模型: {model_name_with_ext}")
                    
                    # 尝试多种下载方法
                    success = False
                    error_messages = []
                    
                    # 方法1: 直接使用ultralytics下载（推荐）
                    try:
                        self.log_update.emit("方法1: 尝试使用ultralytics自动下载...")
                        
                        # 临时禁用ultralytics的GitHub API检查，直接下载
                        os.environ['ULTRALYTICS_OFFLINE'] = '1'
                        
                        model = YOLO(model_name_with_ext)
                        
                        # 恢复环境变量
                        if 'ULTRALYTICS_OFFLINE' in os.environ:
                            del os.environ['ULTRALYTICS_OFFLINE']
                        
                        # 验证模型是否成功创建
                        if model is not None:
                            self.log_update.emit("✓ ultralytics自动下载成功!")
                            
                            # 尝试获取模型路径
                            model_path = None
                            if hasattr(model, 'ckpt_path') and model.ckpt_path:
                                model_path = model.ckpt_path
                            elif hasattr(model, 'model_path') and model.model_path:
                                model_path = model.model_path
                            
                            if model_path and os.path.exists(model_path):
                                self.download_complete.emit(True, f"模型已保存到: {model_path}")
                                return
                            else:
                                self.download_complete.emit(True, f"模型 {model_name_with_ext} 下载并缓存成功")
                                return
                                
                    except Exception as e:
                        # 清理环境变量
                        if 'ULTRALYTICS_OFFLINE' in os.environ:
                            del os.environ['ULTRALYTICS_OFFLINE']
                        error_msg = f"ultralytics自动下载失败: {str(e)}"
                        self.log_update.emit(f"✗ {error_msg}")
                        error_messages.append(error_msg)
                    
                    # 方法2: 手动下载（备用方案）
                    try:
                        self.log_update.emit("方法2: 尝试手动下载...")
                        
                        # 构建正确的下载URL - 根据实际GitHub releases结构
                        base_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
                        download_url = f"{base_url}/{model_name_with_ext}"
                        
                        self.log_update.emit(f"下载URL: {download_url}")
                        
                        # 检查模型是否真的存在
                        head_response = requests.head(download_url, timeout=10)
                        if head_response.status_code == 404:
                            # 尝试其他可能的URL
                            alternative_urls = [
                                f"https://github.com/ultralytics/assets/releases/latest/download/{model_name_with_ext}",
                                f"https://github.com/ultralytics/assets/releases/download/v8.2.0/{model_name_with_ext}",
                            ]
                            
                            for alt_url in alternative_urls:
                                self.log_update.emit(f"尝试备用URL: {alt_url}")
                                alt_head = requests.head(alt_url, timeout=10)
                                if alt_head.status_code == 200:
                                    download_url = alt_url
                                    self.log_update.emit(f"找到有效URL: {download_url}")
                                    break
                            else:
                                raise Exception(f"无法找到模型 {model_name_with_ext} 的有效下载链接")
                        
                        # 下载文件
                        self.log_update.emit("开始下载文件...")
                        response = requests.get(download_url, stream=True, timeout=60)
                        response.raise_for_status()
                        
                        # 保存到ultralytics缓存目录
                        home_dir = os.path.expanduser("~")
                        cache_dir = os.path.join(home_dir, ".cache", "ultralytics")
                        os.makedirs(cache_dir, exist_ok=True)
                        model_path = os.path.join(cache_dir, model_name_with_ext)
                        
                        # 下载进度
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded = 0
                        
                        with open(model_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded += len(chunk)
                                    if total_size > 0:
                                        progress = int(downloaded * 100 / total_size)
                                        if progress % 10 == 0:  # 每10%更新一次
                                            self.log_update.emit(f"下载进度: {progress}%")
                        
                        self.log_update.emit(f"✓ 手动下载成功: {model_path}")
                        
                        # 验证下载的模型
                        if os.path.exists(model_path) and os.path.getsize(model_path) > 1024:  # 至少1KB
                            # 尝试加载模型验证
                            try:
                                test_model = YOLO(model_path)
                                self.log_update.emit("✓ 模型验证成功!")
                                self.download_complete.emit(True, f"模型已下载到: {model_path}")
                                return
                            except Exception as ve:
                                self.log_update.emit(f"模型验证失败: {str(ve)}")
                                error_messages.append(f"模型验证失败: {str(ve)}")
                        else:
                            error_messages.append("下载的文件大小异常")
                            
                    except Exception as e:
                        error_msg = f"手动下载失败: {str(e)}"
                        self.log_update.emit(f"✗ {error_msg}")
                        error_messages.append(error_msg)
                    
                    # 如果所有方法都失败了
                    # 检查ultralytics版本信息
                    current_version = ultralytics.__version__
                    version_info = self.get_ultralytics_version_info(current_version)
                    
                    # 获取缓存目录信息
                    import os
                    home_dir = os.path.expanduser("~")
                    cache_dir = os.path.join(home_dir, ".cache", "ultralytics")
                    
                    combined_error = "所有下载方法都失败:\n" + "\n".join(f"{i+1}. {err}" for i, err in enumerate(error_messages))
                    combined_error += f"\n\n📊 版本信息:\n"
                    combined_error += f"当前ultralytics版本: {current_version}\n"
                    combined_error += version_info
                    combined_error += f"\n\n💡 解决方案:\n"
                    combined_error += "1. 检查网络连接\n"
                    combined_error += "2. 确保ultralytics版本>=8.3.0\n" 
                    combined_error += "3. 升级ultralytics: pip install --upgrade ultralytics\n"
                    combined_error += f"4. 手动下载模型文件到缓存目录:\n   {cache_dir}\n"
                    combined_error += f"   下载链接: https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name_with_ext}\n"
                    combined_error += "5. 使用VPN或代理，GitHub可能被墙\n"
                    combined_error += "6. 尝试使用'本地模型文件'选项"
                    
                    self.download_complete.emit(False, combined_error)
                        
                except ImportError as e:
                    error_msg = f"导入ultralytics失败: {str(e)}\n请安装ultralytics: pip install ultralytics"
                    self.log_update.emit(error_msg)
                    self.download_complete.emit(False, error_msg)
                except Exception as e:
                    error_msg = f"下载过程出错: {str(e)}"
                    self.log_update.emit(error_msg)
                    
                    # 获取缓存目录信息以供手动下载参考
                    try:
                        import os
                        home_dir = os.path.expanduser("~")
                        cache_dir = os.path.join(home_dir, ".cache", "ultralytics")
                    except:
                        cache_dir = "~/.cache/ultralytics"
                    
                    # 根据不同的错误类型提供不同的建议
                    suggestions = []
                    if "No such file or directory" in str(e):
                        suggestions.extend([
                            "1. 模型名称可能不正确",
                            "2. ultralytics版本不支持该模型", 
                            "3. 网络连接问题"
                        ])
                    elif "Permission" in str(e):
                        suggestions.extend([
                            "1. 没有写入权限",
                            "2. 防病毒软件阻止"
                        ])
                    elif "timeout" in str(e).lower():
                        suggestions.extend([
                            "1. 网络超时",
                            "2. 代理设置问题"
                        ])
                    
                    if suggestions:
                        error_msg += f"\n\n可能的原因:\n" + "\n".join(suggestions)
                    
                    # 添加手动下载建议
                    error_msg += f"\n\n💡 手动下载方案:\n"
                    error_msg += f"1. 创建缓存目录: {cache_dir}\n"
                    error_msg += f"2. 下载模型文件到该目录:\n"
                    error_msg += f"   https://github.com/ultralytics/assets/releases/download/v8.3.0/{self.model_name}\n"
                    error_msg += f"3. 或者尝试备用链接:\n"
                    error_msg += f"   https://github.com/ultralytics/assets/releases/latest/download/{self.model_name}"
                    
                    self.download_complete.emit(False, error_msg)
        
        # 创建下载线程
        self.download_thread = QThread()
        self.download_worker = ModelDownloadWorker(model_name)
        self.download_worker.moveToThread(self.download_thread)
        
        # 连接信号
        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.download_complete.connect(self.on_model_download_complete)
        self.download_worker.log_update.connect(self.log_message)  # 连接日志信号
        self.download_worker.download_complete.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self.download_thread.deleteLater)
        
        # 启动下载
        self.download_thread.start()

    def on_model_download_complete(self, success, message):
        """模型下载完成的回调"""
        self.download_model_btn.setEnabled(True)
        self.download_model_btn.setText("下载模型")
        
        if success:
            self.model_status_label.setText("✓ 模型已下载")
            self.model_status_label.setStyleSheet("color: green; font-weight: bold;")
            self.download_model_btn.setVisible(False)
            self.log_message(f"模型下载成功: {message}")
            
            # 重新检查模型状态以确保UI更新
            self.check_selected_model_status()
            
            QMessageBox.information(self, "下载成功", f"模型下载成功！\n{message}")
        else:
            self.model_status_label.setText("❌ 下载失败")
            self.model_status_label.setStyleSheet("color: red; font-weight: bold;")
            self.log_message(f"模型下载失败: {message}")
            
            # 显示详细的错误信息和解决建议
            error_details = f"模型下载失败！\n\n错误信息: {message}\n\n建议解决方案:\n"
            error_details += "1. 检查网络连接是否正常\n"
            error_details += "2. 确保ultralytics版本支持该模型\n"
            error_details += "3. 尝试手动下载模型文件\n"
            error_details += "4. 或使用'本地模型文件'选项"
            
            QMessageBox.warning(self, "下载失败", error_details) 

    def open_folder(self, folder_path):
        """跨平台打开文件夹"""
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "错误", f"文件夹不存在: {folder_path}")
            return
            
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(folder_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", folder_path])
            else:  # Linux and other Unix-like systems
                subprocess.run(["xdg-open", folder_path])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开文件夹: {str(e)}")

    def open_model_cache_folder(self):
        """打开模型缓存文件夹"""
        try:
            # 尝试多个可能的ultralytics模型存储位置
            home_dir = os.path.expanduser("~")
            possible_paths = []
            
            # 1. 标准ultralytics缓存目录
            standard_cache = os.path.join(home_dir, ".cache", "ultralytics")
            possible_paths.append(standard_cache)
            
            # 2. torch hub缓存目录中的ultralytics子目录
            torch_cache = os.path.join(home_dir, ".cache", "torch", "hub")
            possible_paths.append(torch_cache)
            
            # 3. 当前工作目录
            current_dir = os.getcwd()
            possible_paths.append(current_dir)
            
            # 4. 用户目录下的.ultralytics文件夹
            ultralytics_user_dir = os.path.join(home_dir, ".ultralytics")
            possible_paths.append(ultralytics_user_dir)
            
            self.log_message("正在搜索模型缓存目录...")
            
            # 寻找实际包含模型文件的目录
            model_dirs_with_files = []
            for path in possible_paths:
                if os.path.exists(path):
                    # 检查是否包含.pt文件
                    pt_files = []
                    try:
                        # 搜索当前目录和一级子目录
                        for item in os.listdir(path):
                            item_path = os.path.join(path, item)
                            if os.path.isfile(item_path) and item.endswith('.pt'):
                                pt_files.append(item_path)
                            elif os.path.isdir(item_path):
                                # 检查子目录中的.pt文件
                                try:
                                    for subitem in os.listdir(item_path):
                                        if subitem.endswith('.pt'):
                                            pt_files.append(os.path.join(item_path, subitem))
                                except:
                                    continue
                    except:
                        continue
                    
                    if pt_files:
                        model_dirs_with_files.append((path, len(pt_files), pt_files[:3]))  # 最多显示3个文件示例
            
            if model_dirs_with_files:
                # 找到了包含模型文件的目录，选择文件最多的那个
                best_dir = max(model_dirs_with_files, key=lambda x: x[1])
                cache_dir = best_dir[0]
                file_count = best_dir[1]
                example_files = best_dir[2]
                
                self.log_message(f"✅ 找到模型缓存目录: {cache_dir}")
                self.log_message(f"📁 包含 {file_count} 个模型文件")
                self.log_message(f"📄 示例文件: {[os.path.basename(f) for f in example_files]}")
                
                QMessageBox.information(
                    self, 
                    "找到模型缓存目录", 
                    f"模型缓存目录: {cache_dir}\n\n"
                    f"发现 {file_count} 个模型文件\n"
                    f"示例: {', '.join([os.path.basename(f) for f in example_files])}"
                )
                self.open_folder(cache_dir)
            else:
                # 没找到现有的模型文件，创建标准缓存目录
                cache_dir = standard_cache
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)
                    self.log_message(f"📁 创建新的模型缓存目录: {cache_dir}")
                else:
                    self.log_message(f"📁 打开空的模型缓存目录: {cache_dir}")
                
                # 显示详细的提示信息
                QMessageBox.information(
                    self, 
                    "模型缓存目录", 
                    f"模型缓存目录: {cache_dir}\n\n"
                    f"💡 提示：目录为空，可能是因为:\n"
                    f"• 还没有下载过任何模型\n"
                    f"• 模型存储在其他位置\n"
                    f"• ultralytics版本较新，存储位置发生了变化\n\n"
                    f"🔧 建议操作:\n"
                    f"• 先下载一个模型，再查看此目录\n"
                    f"• 检查其他可能的存储位置\n"
                    f"• 查看训练日志中的模型路径信息"
                )
                self.open_folder(cache_dir)
                
        except Exception as e:
            error_msg = f"无法搜索模型缓存目录: {str(e)}"
            self.log_message(f"❌ {error_msg}")
            QMessageBox.warning(self, "错误", error_msg)

    def get_ultralytics_version_info(self, current_version):
        """获取ultralytics版本信息和建议"""
        try:
            import requests
            import json
            from packaging import version
            
            # 获取最新版本信息
            try:
                response = requests.get("https://pypi.org/pypi/ultralytics/json", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    latest_version = data['info']['version']
                    
                    # 比较版本
                    try:
                        current_ver = version.parse(current_version)
                        latest_ver = version.parse(latest_version)
                        
                        if current_ver < latest_ver:
                            version_status = f"⚠️ 有新版本可用: {latest_version}"
                            version_suggestion = f"建议升级: pip install ultralytics=={latest_version}"
                        elif current_ver == latest_ver:
                            version_status = f"✅ 已是最新版本: {latest_version}"
                            version_suggestion = "版本是最新的，问题可能是网络或其他原因"
                        else:
                            version_status = f"🚀 开发版本: {current_version} (最新发布: {latest_version})"
                            version_suggestion = "使用的是开发版本"
                            
                        # 检查YOLO11支持
                        yolo11_support = ""
                        if current_ver >= version.parse("8.3.0"):
                            yolo11_support = "✅ 支持YOLO11"
                        else:
                            yolo11_support = "❌ 不支持YOLO11 (需要>=8.3.0)"
                            
                        return f"{version_status}\n最新发布版本: {latest_version}\n{yolo11_support}\n💡 {version_suggestion}"
                        
                    except Exception as ve:
                        return f"最新版本: {latest_version}\n⚠️ 版本比较失败: {str(ve)}"
                else:
                    return f"⚠️ 无法获取最新版本信息 (HTTP {response.status_code})"
                    
            except requests.RequestException as re:
                return f"⚠️ 网络错误，无法获取最新版本: {str(re)}"
                
        except ImportError:
            # 如果没有requests或packaging库，提供基本信息
            try:
                from packaging import version
                if version.parse(current_version) >= version.parse("8.3.0"):
                    return "✅ 当前版本应该支持YOLO11\n💡 建议检查网络连接或尝试手动下载"
                else:
                    return "❌ 当前版本可能不支持YOLO11\n💡 建议升级: pip install --upgrade ultralytics"
            except ImportError:
                # 简单的字符串比较
                if current_version.startswith('8.3') or current_version.startswith('8.4') or current_version.startswith('8.5'):
                    return "✅ 当前版本应该支持YOLO11\n💡 建议检查网络连接"
                else:
                    return "⚠️ 建议升级到最新版本\n💡 运行: pip install --upgrade ultralytics"
        except Exception as e:
            return f"⚠️ 版本检查失败: {str(e)}" 