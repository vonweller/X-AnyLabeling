class DatasetTab(QWidget):
    """Tab for dataset conversion and management."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Create and arrange UI elements."""
        layout = QVBoxLayout()
        
        # 修改标题
        title_label = QLabel("模型与数据转换")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)
        
        # 模型转换部分
        model_conversion_group = QGroupBox("模型转换")
        model_layout = QFormLayout()
        
        # PT模型选择
        self.pt_model_layout = QHBoxLayout()
        self.pt_model_edit = QLineEdit()
        self.pt_model_edit.setReadOnly(True)
        self.pt_model_btn = QPushButton("浏览...")
        self.pt_model_layout.addWidget(self.pt_model_edit)
        self.pt_model_layout.addWidget(self.pt_model_btn)
        model_layout.addRow("PT模型文件:", self.pt_model_layout)
        
        # 输出目录选择
        self.onnx_output_layout = QHBoxLayout()
        self.onnx_output_edit = QLineEdit()
        self.onnx_output_edit.setReadOnly(True)
        self.onnx_output_btn = QPushButton("浏览...")
        self.onnx_output_layout.addWidget(self.onnx_output_edit)
        self.onnx_output_layout.addWidget(self.onnx_output_btn)
        model_layout.addRow("输出目录:", self.onnx_output_layout)
        
        # 转换按钮
        self.convert_btn = QPushButton("转换为ONNX")
        self.convert_btn.setMinimumHeight(40)
        model_layout.addRow(self.convert_btn)
        
        model_conversion_group.setLayout(model_layout)
        layout.addWidget(model_conversion_group)
        
        # 数据集转换部分
        dataset_group = QGroupBox("数据集转换")
        dataset_layout = QFormLayout()
        
        # ... 原有的数据集转换UI代码 ...
        
        # 连接信号
        self.pt_model_btn.clicked.connect(self.select_pt_model)
        self.onnx_output_btn.clicked.connect(self.select_onnx_output_dir)
        self.convert_btn.clicked.connect(self.convert_to_onnx)
        
        self.setLayout(layout)
    
    def select_pt_model(self):
        """选择PT模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择PT模型文件",
            "",
            "PyTorch Model Files (*.pt)"
        )
        if file_path:
            self.pt_model_edit.setText(file_path)
    
    def select_onnx_output_dir(self):
        """选择ONNX输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录"
        )
        if dir_path:
            self.onnx_output_edit.setText(dir_path)
    
    def convert_to_onnx(self):
        """将PT模型转换为ONNX格式"""
        pt_model_path = self.pt_model_edit.text()
        output_dir = self.onnx_output_edit.text()
        
        if not pt_model_path or not os.path.isfile(pt_model_path):
            QMessageBox.warning(self, "错误", "请选择有效的PT模型文件")
            return
            
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "错误", "请选择有效的输出目录")
            return
            
        try:
            # 获取模型文件名（不含扩展名）
            model_name = os.path.splitext(os.path.basename(pt_model_path))[0]
            onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
            
            # 导入模型
            from ultralytics import YOLO
            model = YOLO(pt_model_path)
            
            # 导出ONNX
            model.export(format="onnx", imgsz=640)  # 使用默认图像大小640
            
            # 移动生成的ONNX文件到指定目录
            if os.path.exists(f"{model_name}.onnx"):
                os.rename(f"{model_name}.onnx", onnx_path)
                QMessageBox.information(self, "转换成功", f"模型已成功转换为ONNX格式并保存到:\n{onnx_path}")
            else:
                QMessageBox.warning(self, "转换失败", "ONNX文件生成失败")
                
        except Exception as e:
            QMessageBox.critical(self, "转换错误", f"转换过程中发生错误:\n{str(e)}")