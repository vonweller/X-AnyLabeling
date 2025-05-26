import os
import sys
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QFileDialog, QComboBox, QLineEdit, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QCheckBox, 
                            QMessageBox, QProgressBar, QTextEdit, QScrollArea,
                            QRadioButton, QButtonGroup, QFormLayout, QSlider,
                            QInputDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QColor, QFont, QDesktopServices

from utils.training_worker import TrainingWorker
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
        
        # Default paths (will be updated from settings if available)
        self.default_train_dir = ""
        self.default_val_dir = ""
        self.default_output_dir = ""
        self.default_model_path = ""  # Added for default model path
        
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
        self.update_fine_tuning_state()
        self.update_weights_path_state()
        
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
        self.data_group = QGroupBox("数据集") # Made data_group an instance variable
        data_layout = QFormLayout()
        data_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        
        # Training images and labels
        self.train_images_layout = QHBoxLayout()
        self.train_images_edit = QLineEdit()
        self.train_images_edit.setReadOnly(True)
        self.train_images_btn = QPushButton("浏览...")
        self.train_images_layout.addWidget(self.train_images_edit)
        self.train_images_layout.addWidget(self.train_images_btn)
        self.train_images_label = QLabel("训练图像目录:") # Keep a reference to the label
        data_layout.addRow(self.train_images_label, self.train_images_layout)

        self.train_labels_layout = QHBoxLayout()
        self.train_labels_edit = QLineEdit()
        self.train_labels_edit.setReadOnly(True)
        self.train_labels_btn = QPushButton("浏览...")
        self.train_labels_layout.addWidget(self.train_labels_edit)
        self.train_labels_layout.addWidget(self.train_labels_btn)
        self.train_labels_label = QLabel("训练标签目录:") # Keep a reference to the label
        data_layout.addRow(self.train_labels_label, self.train_labels_layout)

        # Validation images and labels
        self.val_images_layout = QHBoxLayout()
        self.val_images_edit = QLineEdit()
        self.val_images_edit.setReadOnly(True)
        self.val_images_btn = QPushButton("浏览...")
        self.val_images_layout.addWidget(self.val_images_edit)
        self.val_images_layout.addWidget(self.val_images_btn)
        self.val_images_label = QLabel("验证图像目录:") # Keep a reference to the label
        data_layout.addRow(self.val_images_label, self.val_images_layout)

        self.val_labels_layout = QHBoxLayout()
        self.val_labels_edit = QLineEdit()
        self.val_labels_edit.setReadOnly(True)
        self.val_labels_btn = QPushButton("浏览...")
        self.val_labels_layout.addWidget(self.val_labels_edit)
        self.val_labels_layout.addWidget(self.val_labels_btn)
        self.val_labels_label = QLabel("验证标签目录:") # Keep a reference to the label
        data_layout.addRow(self.val_labels_label, self.val_labels_layout)
        
        # Add widgets to form layout
        self.data_group.setLayout(data_layout) # Use self.data_group
        
        # Model section
        model_group = QGroupBox("模型配置")
        model_layout = QFormLayout()
        
        # Model type selection
        self.model_combo = QComboBox()
        self.model_combo.currentTextChanged.connect(self.on_model_selection_changed)
        
        # Model Initialization Options
        init_group_box = QGroupBox("模型初始化")
        init_layout = QVBoxLayout()
        
        # Radio buttons for initialization options
        self.model_init_group = QButtonGroup(self)
        
        self.use_pretrained_radio = QRadioButton("使用预训练权重")
        self.from_scratch_radio = QRadioButton("从头开始训练（不使用预训练权重）")
        self.custom_weights_radio = QRadioButton("使用自定义权重")
        
        self.model_init_group.addButton(self.use_pretrained_radio)
        self.model_init_group.addButton(self.from_scratch_radio)
        self.model_init_group.addButton(self.custom_weights_radio)
        
        init_layout.addWidget(self.use_pretrained_radio)
        init_layout.addWidget(self.from_scratch_radio)
        init_layout.addWidget(self.custom_weights_radio)
        
        # Custom weights path layout
        self.model_path_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_btn = QPushButton("浏览...")
        self.model_path_layout.addWidget(self.model_path_edit)
        self.model_path_layout.addWidget(self.model_path_btn)
        
        # Default to use pretrained
        self.use_pretrained_radio.setChecked(True)
        
        # Initially disable the model path controls since 'Use pretrained' is selected by default
        self.model_path_edit.setEnabled(False)
        self.model_path_btn.setEnabled(False)
        
        # Fine-tuning mode
        self.fine_tuning_mode = QCheckBox("微调模式（冻结骨干网络，仅训练检测头）")
        self.fine_tuning_mode.setChecked(False)
        
        init_layout.addWidget(self.fine_tuning_mode)
        init_group_box.setLayout(init_layout)
        
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
        model_layout.addRow("自定义权重:", self.model_path_layout)
        model_layout.addRow("批次大小:", self.batch_size_spin)
        model_layout.addRow("训练轮数:", self.epochs_spin)
        model_layout.addRow("图像尺寸:", self.img_size_spin)
        model_layout.addRow("学习率:", self.lr_spin)
        model_layout.addWidget(init_group_box)
        model_group.setLayout(model_layout)
        
        # Output section
        output_group = QGroupBox("输出")
        output_layout = QFormLayout()
        
        self.output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setReadOnly(True)
        self.output_dir_btn = QPushButton("浏览...")
        self.output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_layout.addWidget(self.output_dir_btn)
        
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
        """Connect UI signals to slots."""
        # Model selection change
        self.model_combo.currentIndexChanged.connect(self.update_parameters_display)
        
        # Directory selection
        self.train_images_btn.clicked.connect(lambda: self.select_directory("训练图像目录", self.train_images_edit))
        self.train_labels_btn.clicked.connect(lambda: self.select_directory("训练标签目录", self.train_labels_edit))
        self.val_images_btn.clicked.connect(lambda: self.select_directory("验证图像目录", self.val_images_edit))
        self.val_labels_btn.clicked.connect(lambda: self.select_directory("验证标签目录", self.val_labels_edit))
        self.output_dir_btn.clicked.connect(lambda: self.select_directory("输出目录", self.output_dir_edit))
        self.model_path_btn.clicked.connect(self.select_model_path)
        
        # Form controls
        self.use_pretrained_radio.toggled.connect(self.on_initialization_mode_changed)
        self.custom_weights_radio.toggled.connect(self.on_initialization_mode_changed)
        self.from_scratch_radio.toggled.connect(self.on_initialization_mode_changed)
        
        # Connect fine-tuning checkbox to ensure it works only with pretrained weights
        self.fine_tuning_mode.toggled.connect(self.update_fine_tuning_state)
        
        # Control buttons
        self.validate_btn.clicked.connect(self.validate_dataset)
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
    
    def on_initialization_mode_changed(self, checked=None):
        """Handle changes to initialization mode radio buttons"""
        # Update dependent UI states
        self.update_weights_path_state()
        self.update_fine_tuning_state()
        
        # If training from scratch is selected, disable fine-tuning
        if self.from_scratch_radio.isChecked() and self.fine_tuning_mode.isChecked():
            self.fine_tuning_mode.setChecked(False)
            self.log_message("从头开始训练不支持微调模式，已禁用微调")
    
    def select_train_dir(self):
        """Open dialog to select training data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Training Data Directory")
        if dir_path:
            self.train_dir_edit.setText(dir_path)
    
    def select_val_dir(self):
        """Open dialog to select validation data directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Validation Data Directory")
        if dir_path:
            self.val_dir_edit.setText(dir_path)
    
    def select_output_dir(self):
        """Open dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_dir_edit.setText(dir_path)
    
    def select_model_path(self):
        """Open dialog to select model weights file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weights", "", "Model Files (*.pt *.pth *.weights);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)
            # Update fine-tuning state since a model has been selected
            self.update_fine_tuning_state()
    
    def start_training(self):
        """Validate inputs and start training in a separate thread."""
        # Validate inputs
        if not self.validate_inputs():
            return
        
        # Disable UI elements during training
        self.set_ui_enabled(False)
        self.is_training = True
        
        # Clear log
        self.log_text.clear()
        self.log_message("正在准备训练任务...")
        
        # Determine which training mode to use
        if self.custom_weights_radio.isChecked():
            # Custom weights mode
            model_weights_path = self.model_path_edit.text()
            if not model_weights_path:
                self.log_message("警告: 选择了自定义权重，但未指定权重文件")
                QMessageBox.warning(self, "缺少输入", "使用自定义权重模式时，请选择模型权重文件。")
                self.set_ui_enabled(True)
                self.is_training = False
                return
            # For custom weights, pretrained is False as we are providing specific weights.
            # The model_type (e.g., yolov8n-cls) might still be relevant for architecture if weights are partial.
            # However, Ultralytics typically infers architecture from .pt file itself.
            # We pass the raw model_type, and the worker will use model_weights_path primarily.
            current_model_type_for_worker = self.model_type # e.g., "yolov8n-cls"
            use_pretrained_for_worker = False
            model_weights_for_worker = model_weights_path
            self.log_message(f"使用自定义权重: {model_weights_path} 为模型 {current_model_type_for_worker}")

        elif self.use_pretrained_radio.isChecked():
            # Pretrained weights mode
            current_model_type_for_worker = self.model_type # e.g., "yolov8n-cls"
            use_pretrained_for_worker = True
            model_weights_for_worker = None # Worker will handle forming "model_type.pt" for download/load
            self.log_message(f"使用预训练权重初始化模型: {current_model_type_for_worker}")
        else: # self.from_scratch_radio.isChecked()
            # Train from scratch mode
            current_model_type_for_worker = self.model_type # e.g., "yolov8n-cls"
            use_pretrained_for_worker = False
            model_weights_for_worker = None # Worker will handle forming "model_type.yaml" for loading config
            self.log_message(f"从头开始训练模型 (不使用预训练权重): {current_model_type_for_worker}")
        
        # Check if it's fine-tuning mode
        fine_tuning = self.fine_tuning_mode.isChecked()
        if fine_tuning and not (self.use_pretrained_radio.isChecked() or self.custom_weights_radio.isChecked()):
            self.log_message("警告: 微调模式需要预训练模型或自定义权重！已禁用微调。")
            fine_tuning = False
        
        # Create worker instance
        self.training_worker = TrainingWorker(
            model_type=current_model_type_for_worker, # Pass the base model type e.g., "yolov8n-cls"
            task_type=self.task_type,
            train_dir=self.train_images_edit.text(),
            val_dir=self.val_images_edit.text(), 
            output_dir=self.output_dir_edit.text(),
            project_name=self.project_name_edit.text(),
            batch_size=self.batch_size_spin.value(),
            epochs=self.epochs_spin.value(),
            img_size=self.img_size_spin.value(),
            learning_rate=self.lr_spin.value(),
            pretrained=use_pretrained_for_worker, # Explicitly pass the pretrained flag
            model_weights=model_weights_for_worker, # Pass the path to custom weights, or None
            fine_tuning=fine_tuning
        )
        
        self.training_thread = QThread()
        self.training_worker.moveToThread(self.training_thread)
        
        # Connect signals
        self.training_worker.progress_update.connect(self.update_progress)
        self.training_worker.log_update.connect(self.log_message)
        self.training_worker.training_complete.connect(self.on_training_complete)
        self.training_worker.training_error.connect(self.on_training_error)
        self.training_thread.started.connect(self.training_worker.run)
        
        # Start training
        self.training_thread.start()
    
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
        self.val_images_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        
        # Model path button should only be enabled if custom weights is checked
        model_path_enabled = enabled and self.custom_weights_radio.isChecked()
        self.model_path_btn.setEnabled(model_path_enabled)
        self.model_path_edit.setEnabled(model_path_enabled)
        
        self.model_combo.setEnabled(enabled)
        self.batch_size_spin.setEnabled(enabled)
        self.epochs_spin.setEnabled(enabled)
        self.img_size_spin.setEnabled(enabled)
        self.lr_spin.setEnabled(enabled)
        
        # Radio buttons for model initialization
        self.use_pretrained_radio.setEnabled(enabled)
        self.from_scratch_radio.setEnabled(enabled)
        self.custom_weights_radio.setEnabled(enabled)
        
        # Fine-tuning is only enabled if using pretrained or custom weights
        fine_tuning_enabled = enabled and (self.use_pretrained_radio.isChecked() or self.custom_weights_radio.isChecked())
        self.fine_tuning_mode.setEnabled(fine_tuning_enabled)
        
        self.project_name_edit.setEnabled(enabled)
    
    def validate_inputs(self):
        """Validate user inputs before starting training."""
        # 检查训练图像目录
        if not self.train_images_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择训练数据目录。")
            return False

        # 对于检测任务，标签目录是必需的
        if self.task_type == "detect":
            if not self.train_labels_edit.text():
                QMessageBox.warning(self, "缺少输入", "目标检测任务请选择训练标签目录。")
                return False
            if not self.val_labels_edit.text() and self.val_images_edit.text(): # If val images are provided, labels are also needed for detection
                QMessageBox.warning(self, "缺少输入", "目标检测任务请为验证集选择标签目录。")
                return False

        # 验证图像目录不是必须的，但如果提供了，其标签目录对于检测也是必须的
        if not self.val_images_edit.text() and self.task_type == "detect" and self.val_labels_edit.text():
            QMessageBox.warning(self, "输入不一致", "为验证集提供了标签目录但未提供图像目录。")
            return False

        # 检查输出目录
        if not self.output_dir_edit.text():
            QMessageBox.warning(self, "缺少输入", "请选择输出目录。")
            return False
        return True
    
    def update_settings(self, settings):
        """Update tab settings based on settings from settings tab."""
        if 'default_model' in settings:
            index = self.model_combo.findText(settings['default_model'])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
                self.model_type = settings['default_model']
        
        if 'default_batch_size' in settings:
            self.batch_size_spin.setValue(settings['default_batch_size'])
        
        if 'default_img_size' in settings:
            self.img_size_spin.setValue(settings['default_img_size'])
        
        # Update default paths for images/labels and set in UI if empty
        if 'default_train_images_dir' in settings:
            if settings['default_train_images_dir'] and not self.train_images_edit.text():
                self.train_images_edit.setText(settings['default_train_images_dir'])
        if 'default_train_labels_dir' in settings:
            if settings['default_train_labels_dir'] and not self.train_labels_edit.text():
                self.train_labels_edit.setText(settings['default_train_labels_dir'])
        if 'default_val_images_dir' in settings:
            if settings['default_val_images_dir'] and not self.val_images_edit.text():
                self.val_images_edit.setText(settings['default_val_images_dir'])
        if 'default_val_labels_dir' in settings:
            if settings['default_val_labels_dir'] and not self.val_labels_edit.text():
                self.val_labels_edit.setText(settings['default_val_labels_dir'])
        
        if 'default_output_dir' in settings:
            self.default_output_dir = settings['default_output_dir']
            if self.default_output_dir and not self.output_dir_edit.text():
                self.output_dir_edit.setText(self.default_output_dir)
            
        # Update default model path and set in UI if empty
        if 'default_train_model_path' in settings:
            self.default_model_path = settings['default_train_model_path']
            if self.default_model_path and not self.model_path_edit.text():
                self.model_path_edit.setText(self.default_model_path)
                # If a default model path is provided, select custom weights radio button
                self.custom_weights_radio.setChecked(True)
            
        # Update UI states
        self.update_weights_path_state()
        self.update_fine_tuning_state()
    
    def update_fine_tuning_state(self, checked=None):
        """Update UI state based on fine-tuning and model initialization options."""
        # Fine-tuning requires pretrained or custom weights
        using_pretrained_weights = self.use_pretrained_radio.isChecked() or self.custom_weights_radio.isChecked()
        using_custom_weights = self.custom_weights_radio.isChecked() and bool(self.model_path_edit.text())
        
        # Fine-tuning is only enabled if using pretrained or valid custom weights
        fine_tuning_enabled = using_pretrained_weights or using_custom_weights
        
        # Set the enabled state of fine_tuning_mode checkbox
        self.fine_tuning_mode.setEnabled(fine_tuning_enabled)
        
        # If fine-tuning is checked but not using pretrained or custom weights, uncheck it
        if self.fine_tuning_mode.isChecked() and not fine_tuning_enabled:
            self.fine_tuning_mode.setChecked(False)
        
        # Update the tooltip based on state
        if fine_tuning_enabled:
            self.fine_tuning_mode.setToolTip("冻结检测头之前的所有参数，仅更新检测头参数")
        else:
            self.fine_tuning_mode.setToolTip("微调模式需要预训练模型或指定的权重文件")
    
    def update_weights_path_state(self, checked=None):
        """Enable or disable custom weights path based on radio button state"""
        # Only enable the model path controls when custom weights is selected
        is_custom = self.custom_weights_radio.isChecked()
        self.model_path_edit.setEnabled(is_custom)
        self.model_path_btn.setEnabled(is_custom)
        
        # Clear the path if custom weights is not selected
        if not is_custom:
            self.model_path_edit.clear()
    
    def select_directory(self, title, line_edit):
        dir_path = QFileDialog.getExistingDirectory(self, title)
        if dir_path:
            line_edit.setText(dir_path)

            # Automatically create train/val subdirectories for classification task
            if self.task_type == "classify" and line_edit is self.train_images_edit:
                selected_path = dir_path
                train_subdir = os.path.join(selected_path, "train")
                val_subdir = os.path.join(selected_path, "val")
                created_train_val_folders = False

                # Check if 'train' and 'val' subdirectories are missing and the selected path is not itself 'train' or 'val'
                if not os.path.exists(train_subdir) and \
                   not os.path.exists(val_subdir) and \
                   os.path.basename(selected_path).lower() not in ["train", "val"]:
                    
                    reply_train_val = QMessageBox.question(self, '创建训练/验证子目录?',
                                                       f"您选择的目录 '{selected_path}' \n"
                                                       f"不包含 'train' 和 'val' 子文件夹。\n\n"
                                                       f"您希望自动创建它们以符合标准的分类数据集结构吗?",
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    
                    if reply_train_val == QMessageBox.Yes:
                        try:
                            os.makedirs(train_subdir, exist_ok=True)
                            os.makedirs(val_subdir, exist_ok=True)
                            self.log_message(f"已在 '{selected_path}' 中创建 'train' 和 'val' 子文件夹。")
                            created_train_val_folders = True
                        except Exception as e:
                            self.log_message(f"创建 train/val 子文件夹失败: {e}")
                            QMessageBox.critical(self, "错误", f"创建 train/val 子文件夹失败: {e}")
                elif os.path.exists(train_subdir) and os.path.exists(val_subdir):
                    created_train_val_folders = True # They already exist
                elif os.path.exists(train_subdir) and not os.path.exists(val_subdir):
                    # If only train exists, ask to create val
                    reply_val = QMessageBox.question(self, '创建验证子目录?', 
                                                   f"目录 '{train_subdir}' 已存在，但 'val' 子文件夹缺失。\n"
                                                   f"您希望创建 'val' 子文件夹吗?",
                                                   QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if reply_val == QMessageBox.Yes:
                        try:
                            os.makedirs(val_subdir, exist_ok=True)
                            self.log_message(f"已在 '{selected_path}' 中创建 'val' 子文件夹。")
                            created_train_val_folders = True # Now both should exist or train exists and val is created
                        except Exception as e:
                            self.log_message(f"创建 'val' 子文件夹失败: {e}")
                            QMessageBox.critical(self, "错误", f"创建 'val' 子文件夹失败: {e}")
                    else:
                         # User chose not to create val, proceed if train exists
                         created_train_val_folders = os.path.exists(train_subdir) 
                else: # train doesn't exist, but val might (unlikely scenario, but handle)
                    created_train_val_folders = False

                # If train (and optionally val) folders are now available, ask to create class subfolders
                if created_train_val_folders and os.path.exists(train_subdir):
                    reply_classes = QMessageBox.question(self, '创建类别子文件夹?',
                                                       f"您想在 'train' (以及 'val'，如果存在) 文件夹内\n"
                                                       f"根据您提供的类别名称创建子文件夹吗?",
                                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                    if reply_classes == QMessageBox.Yes:
                        class_names_str, ok = QInputDialog.getText(self, '输入类别名称',
                                                                      '请输入类别名称，用英文逗号分隔 (例如: 猫,狗,鸟):')
                        if ok and class_names_str:
                            class_names = [name.strip() for name in class_names_str.split(',') if name.strip()]
                            if class_names:
                                for class_name in class_names:
                                    try:
                                        os.makedirs(os.path.join(train_subdir, class_name), exist_ok=True)
                                        if os.path.exists(val_subdir): # Also create in val if it exists
                                            os.makedirs(os.path.join(val_subdir, class_name), exist_ok=True)
                                    except Exception as e:
                                        self.log_message(f"为类别 '{class_name}' 创建子文件夹失败: {e}")
                                        QMessageBox.warning(self, "创建错误", f"为类别 '{class_name}' 创建子文件夹时出错: {e}")
                                        break # Stop if one fails
                                else:
                                    self.log_message(f"已在 'train' (和 'val') 文件夹中为类别 {class_names} 创建了子文件夹。")
                                    QMessageBox.information(self, "完成", "类别子文件夹已创建。")
                            else:
                                QMessageBox.warning(self, "输入无效", "未提供有效的类别名称。")

            # 自动同步到设置页
            from ui.main_window import MainWindow
            main_window = self.parentWidget()
            while main_window and not isinstance(main_window, MainWindow):
                main_window = main_window.parentWidget()
            if main_window and hasattr(main_window, 'settings_tab'):
                settings_tab = main_window.settings_tab
                # 根据line_edit对象同步到对应设置项
                if line_edit is self.train_images_edit:
                    settings_tab.default_train_images_edit.setText(dir_path)
                elif line_edit is self.train_labels_edit:
                    settings_tab.default_train_labels_edit.setText(dir_path)
                elif line_edit is self.val_images_edit:
                    settings_tab.default_val_images_edit.setText(dir_path)
                elif line_edit is self.val_labels_edit:
                    settings_tab.default_val_labels_edit.setText(dir_path)
                # 实时保存
                settings_tab.save_settings()
    
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
        self.model_type = model_name # Store raw model name, e.g., "yolov8n-cls"
        self.log_message(f"已选择模型: {model_name}") # Log the raw name
        # print(f"Model selection changed to (raw): {self.model_type}") # Keep for debugging if needed
        self.update_fine_tuning_state() # Ensure fine-tuning state is updated based on new model selection

    def update_model_list(self):
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        
        yolo_versions = ["8", "9", "10", "11", "12"]
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
        if index == 0: # Detection
            self.task_type = "detect"
        elif index == 1: # Classification
            self.task_type = "classify"
        else:
            self.task_type = "detect" # Default or handle error
        
        self.update_model_list()
        self.update_task_specific_ui()

    def update_task_specific_ui(self):
        if self.task_type == "detect":
            self.data_group.setTitle("数据集 (目标检测)")
            self.train_images_label.setText("训练图像目录:")
            self.train_labels_label.setText("训练标签目录:")
            self.train_labels_edit.setEnabled(True)
            self.train_labels_btn.setEnabled(True)
            
            self.val_images_label.setText("验证图像目录:")
            self.val_labels_label.setText("验证标签目录:")
            self.val_labels_edit.setEnabled(True)
            self.val_labels_btn.setEnabled(True)
            
            # Enable fine-tuning for detection
            self.fine_tuning_mode.setEnabled(True)
            self.fine_tuning_mode.setText("微调模式（冻结骨干网络，仅训练检测头）")

        elif self.task_type == "classify":
            self.data_group.setTitle("数据集 (图像分类)")
            self.train_images_label.setText("训练集根目录 (包含类别子文件夹):")
            self.train_labels_label.setText("训练标签目录 (自动从文件夹结构推断):")
            self.train_labels_edit.setEnabled(False)
            self.train_labels_btn.setEnabled(False)
            self.train_labels_edit.setText("") # Clear if previously set
            
            self.val_images_label.setText("验证集根目录 (包含类别子文件夹):")
            self.val_labels_label.setText("验证标签目录 (自动从文件夹结构推断):")
            self.val_labels_edit.setEnabled(False)
            self.val_labels_btn.setEnabled(False)
            self.val_labels_edit.setText("") # Clear if previously set

            # Fine-tuning might have different meaning or not be applicable in the same way for classification
            # Or it might mean freezing feature extractor layers.
            # For now, let's make it more generic or disable if not directly translatable.
            self.fine_tuning_mode.setEnabled(True) # Or False, depending on how you want to handle it
            self.fine_tuning_mode.setText("微调模式（例如，冻结部分层，仅训练分类器）")
            # You might also want to adjust available models in self.model_combo here
            # or ensure it's called after self.task_type is set.
        
        # This will trigger an update to the model list based on the new task type
        self.update_model_list() 