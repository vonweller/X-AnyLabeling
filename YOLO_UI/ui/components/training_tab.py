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
        self.train_images_layout.addWidget(self.train_images_edit)
        self.train_images_layout.addWidget(self.train_images_btn)
        self.train_images_label = QLabel("训练图像目录:")
        data_layout.addRow(self.train_images_label, self.train_images_layout)
        
        self.train_labels_layout = QHBoxLayout()
        self.train_labels_edit = QLineEdit()
        self.train_labels_edit.setReadOnly(True)
        self.train_labels_btn = QPushButton("浏览...")
        self.train_labels_layout.addWidget(self.train_labels_edit)
        self.train_labels_layout.addWidget(self.train_labels_btn)
        self.train_labels_label = QLabel("训练标签目录:")
        data_layout.addRow(self.train_labels_label, self.train_labels_layout)
        
        # Validation data
        self.val_images_layout = QHBoxLayout()
        self.val_images_edit = QLineEdit()
        self.val_images_edit.setReadOnly(True)
        self.val_images_btn = QPushButton("浏览...")
        self.val_images_layout.addWidget(self.val_images_edit)
        self.val_images_layout.addWidget(self.val_images_btn)
        self.val_images_label = QLabel("验证图像目录:")
        data_layout.addRow(self.val_images_label, self.val_images_layout)
        
        self.val_labels_layout = QHBoxLayout()
        self.val_labels_edit = QLineEdit()
        self.val_labels_edit.setReadOnly(True)
        self.val_labels_btn = QPushButton("浏览...")
        self.val_labels_layout.addWidget(self.val_labels_edit)
        self.val_labels_layout.addWidget(self.val_labels_btn)
        self.val_labels_label = QLabel("验证标签目录:")
        data_layout.addRow(self.val_labels_label, self.val_labels_layout)
        
        # Data YAML path
        self.data_yaml_layout = QHBoxLayout()
        self.data_yaml_path_edit = QLineEdit()
        self.data_yaml_path_edit.setReadOnly(True)
        self.data_yaml_btn = QPushButton("浏览...")
        self.data_yaml_layout.addWidget(self.data_yaml_path_edit)
        self.data_yaml_layout.addWidget(self.data_yaml_btn)
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

        self.local_folder_model_radio = QRadioButton("从本地文件夹选择预训练模型")
        self.local_folder_model_radio.setToolTip("从您指定的本地文件夹中加载所选类型的预训练模型。")
        self.model_source_group.addButton(self.local_folder_model_radio)
        model_source_box_layout.addWidget(self.local_folder_model_radio)
        
        self.local_model_folder_layout = QHBoxLayout()
        self.local_model_folder_edit = QLineEdit()
        self.local_model_folder_edit.setPlaceholderText("选择包含模型的文件夹")
        self.local_model_folder_edit.setReadOnly(True)
        self.local_model_folder_btn = QPushButton("浏览...")
        self.local_model_folder_layout.addWidget(self.local_model_folder_edit)
        self.local_model_folder_layout.addWidget(self.local_model_folder_btn)
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
        self.custom_model_path_layout.addWidget(self.custom_model_path_edit)
        self.custom_model_path_layout.addWidget(self.custom_model_path_btn)
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

    def on_model_source_changed(self, checked=None):
        # If a radio button is checked, then process
        if checked is None or checked: # Ensure this is called correctly
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
        data_yaml_path = self.data_yaml_path_edit.text()
        epochs = self.epochs_spin.value()
        batch_size = self.batch_size_spin.value()
        img_size = self.img_size_spin.value()
        output_dir = self.output_dir_edit.text()
        device = self.device_combo.currentText().split('(')[0].strip().lower() if self.device_combo.currentText() else ''
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
        self.val_images_btn.setEnabled(enabled)
        self.output_dir_btn.setEnabled(enabled)
        
        # Model path button should only be enabled if custom weights is checked
        model_path_enabled = enabled and self.custom_weights_radio.isChecked()
        self.custom_model_path_btn.setEnabled(model_path_enabled)
        self.custom_model_path_edit.setEnabled(model_path_enabled)
        
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

    def update_model_source_ui_state(self):
        """Updates UI elements based on the selected model source and train mode."""
        is_download = self.download_model_radio.isChecked()
        is_local_folder = self.local_folder_model_radio.isChecked()
        is_custom_file = self.custom_weights_radio.isChecked()
        is_from_scratch = self.from_scratch_radio.isChecked()

        # Enable/Disable Local Model Folder selection
        self.local_model_folder_edit.setEnabled(is_local_folder and not is_from_scratch)
        self.local_model_folder_btn.setEnabled(is_local_folder and not is_from_scratch)
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
        # No direct action here, state is handled by update_model_source_ui_state and start_training

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
        self.task_type = "detect" if index == 0 else "classify"
        self.update_task_specific_ui()
        self.update_model_list()  # Update model list based on task type
        self.update_parameters_display() # Update displayed parameters for the new task

    def update_task_specific_ui(self):
        """Update UI elements specific to detection or classification tasks."""
        is_detection = self.task_type == "detect"
        self.data_group.setTitle("数据集 (目标检测)" if is_detection else "数据集 (图像分类)")
        self.train_images_label.setText("训练图像目录:" if is_detection else "训练集根目录 (包含类别子文件夹):")
        self.train_labels_label.setText("训练标签目录:" if is_detection else "训练标签目录 (自动从文件夹结构推断):")
        self.train_labels_edit.setEnabled(is_detection)
        self.train_labels_btn.setEnabled(is_detection)
        
        self.val_images_label.setText("验证图像目录:" if is_detection else "验证集根目录 (包含类别子文件夹):")
        self.val_labels_label.setText("验证标签目录:" if is_detection else "验证标签目录 (自动从文件夹结构推断):")
        self.val_labels_edit.setEnabled(is_detection)
        self.val_labels_btn.setEnabled(is_detection)
        
        # Enable fine-tuning for detection
        self.fine_tuning_mode.setEnabled(is_detection)
        self.fine_tuning_mode.setText("微调模式（冻结骨干网络，仅训练检测头）" if is_detection else "微调模式（例如，冻结部分层，仅训练分类器）")

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

    def on_model_source_changed(self):
        """Handles changes in model source selection (download vs. local)."""
        self.update_model_source_ui_state()

    def on_model_source_changed(self):
        """Handles changes in model source selection (download vs. local)."""
        self.update_model_source_ui_state() 