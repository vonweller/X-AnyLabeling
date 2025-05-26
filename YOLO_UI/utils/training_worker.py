import os
import sys
import time
import threading
import torch
from PyQt5.QtCore import QObject, pyqtSignal
import traceback

# Define a cache directory for downloaded models, relative to this utils.py file
UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_CACHE_DIR = os.path.join(UTILS_DIR, "..", "data", "models")
os.makedirs(DEFAULT_MODEL_CACHE_DIR, exist_ok=True)

class TrainingWorker(QObject):
    """Worker class to handle YOLO model training in a separate thread."""
    
    # Signal definitions
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, model_name, data_yaml_path, epochs, batch_size, img_size, output_dir, 
                 device, task, is_from_scratch, freeze_backbone, other_args,
                 model_source_option, local_model_search_dir=None, project_name="yolo_project"):
        """
        Initialize the training worker with parameters.
        
        Args:
            model_name (str): Base model name (e.g., 'yolov8n.pt' or 'yolov8n' if from scratch),
                              or full path if custom_file or resolved local_folder.
            data_yaml_path (str): Path to the data.yaml file.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            img_size (int): Image size for training.
            output_dir (str): Path to save output results (parent of project_name).
            device (str): Device to use (e.g., 'cpu', '0').
            task (str): Task type ('detect', 'classify', etc.).
            is_from_scratch (bool): Whether to train from scratch.
            freeze_backbone (bool): Whether to freeze backbone layers (or apply task-specific freeze).
            other_args (dict): Dictionary of other arguments for the trainer.
            model_source_option (str): 'download', 'local_folder', or 'custom_file'.
            local_model_search_dir (str, optional): Directory to search for local models if source is 'local_folder'.
            project_name (str): Project name for output organization.
        """
        super().__init__()
        # Renaming for clarity within worker to match original TrainingTab names somewhat
        self.model_type_or_path = model_name 
        self.data_yaml_path = data_yaml_path
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.output_dir = output_dir
        self.device = device
        self.task_type = task # Renamed from task to task_type for consistency
        self.is_from_scratch = is_from_scratch
        self.freeze_backbone = freeze_backbone # This might be passed as 'freeze' in other_args
        self.other_args = other_args if other_args is not None else {}
        
        self.model_source_option = model_source_option
        self.local_model_search_dir = local_model_search_dir
        self.project_name = project_name # Used for `project` in YOLO trainer

        # Original args that might be slightly different now:
        # self.model_type was more like a base (yolov8n), now model_name is more direct
        # self.pretrained is now covered by is_from_scratch and model_source_option
        # self.model_weights is now essentially self.model_type_or_path if custom_file, or resolved path
        # self.learning_rate is expected to be in other_args if specified (e.g., {'lr0': 0.01})
        
        self._stop_event = threading.Event()
        self._trainer_ref = None
        self._process_ref = None
    
    def _check_internet_connection(self):
        """
        Check for internet connectivity.
        Returns:
            bool: True if internet is available, False otherwise
        """
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            self.log_update.emit("Internet connection detected (Google DNS).")
            return True
        except OSError:
            try:
                import socket
                socket.create_connection(("223.5.5.5", 53), timeout=3) # Alibaba DNS
                self.log_update.emit("Internet connection detected (Alibaba DNS).")
                return True
            except OSError:
                self.log_update.emit("No internet connection detected.")
        return False
    
    def _check_gpu_environment(self):
        """检查GPU环境和CUDA版本"""
        try:
            import torch
            self.log_update.emit("正在检查GPU环境...")
            
            # 检查CUDA是否可用
            cuda_available = torch.cuda.is_available()
            self.log_update.emit(f"CUDA 是否可用: {'是' if cuda_available else '否'}")
            
            if cuda_available:
                # 获取CUDA版本
                cuda_version = torch.version.cuda
                self.log_update.emit(f"CUDA 版本: {cuda_version}")
                
                # 获取GPU数量
                gpu_count = torch.cuda.device_count()
                self.log_update.emit(f"检测到 {gpu_count} 个GPU设备:")
                
                # 列出所有GPU信息
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # 转换为GB
                    self.log_update.emit(f"GPU {i}: {gpu_name} (显存: {gpu_memory:.1f}GB)")
                
                # 显示当前使用的GPU
                current_device = torch.cuda.current_device()
                self.log_update.emit(f"当前使用的GPU: {current_device} ({torch.cuda.get_device_name(current_device)})")
            else:
                self.log_update.emit("警告: 未检测到可用的CUDA设备，将使用CPU进行训练")
                
        except Exception as e:
            self.log_update.emit(f"检查GPU环境时出错: {str(e)}")

    def _download_model_if_needed(self, model_name_to_download):
        """
        Downloads the model if it's not in the local cache.
        Uses ultralytics.YOLO to handle the download.
        Args:
            model_name_to_download (str): The model name (e.g., 'yolov8n.pt').
        Returns:
            str: Path to the (potentially downloaded) model file, or None on failure.
        """
        try:
            from ultralytics import YOLO
            # 检查本地缓存
            cached_model_path = os.path.join(DEFAULT_MODEL_CACHE_DIR, model_name_to_download)
            
            if os.path.exists(cached_model_path):
                self.log_update.emit(f"在本地缓存中找到模型 {model_name_to_download}: {cached_model_path}")
                return cached_model_path

            self.log_update.emit(f"模型 {model_name_to_download} 不在本地缓存中，准备下载...")
            
            if not self._check_internet_connection():
                self.log_update.emit(f"无法下载 {model_name_to_download}: 无网络连接")
                return None

            # 设置下载进度回调
            def download_progress_callback(current, total):
                progress = int(current * 100 / total)
                self.log_update.emit(f"下载进度: {progress}% ({current}/{total} bytes)")
                self.progress_update.emit(progress)

            # 使用YOLO下载模型
            self.log_update.emit(f"开始下载模型 {model_name_to_download}...")
            model_instance = YOLO(model_name_to_download)  # 这会触发下载
            
            # 获取下载后的模型路径，兼容ckpt为dict的情况
            downloaded_path = None
            ckpt = getattr(model_instance, 'ckpt', None)
            if isinstance(ckpt, str) and os.path.exists(ckpt):
                downloaded_path = ckpt
            elif hasattr(model_instance, 'ckpt_path') and isinstance(model_instance.ckpt_path, str) and os.path.exists(model_instance.ckpt_path):
                downloaded_path = model_instance.ckpt_path
            elif hasattr(model_instance, 'model') and hasattr(model_instance.model, 'pt_path') and isinstance(model_instance.model.pt_path, str) and os.path.exists(model_instance.model.pt_path):
                downloaded_path = model_instance.model.pt_path
            else:
                if hasattr(model_instance, 'predictor') and hasattr(model_instance.predictor, 'model') and hasattr(model_instance.predictor.model, 'ckpt_path'):
                    ckpt_path = model_instance.predictor.model.ckpt_path
                    if isinstance(ckpt_path, str) and os.path.exists(ckpt_path):
                        downloaded_path = ckpt_path

            if downloaded_path and os.path.exists(downloaded_path):
                self.log_update.emit(f"模型下载完成: {downloaded_path}")
                # 复制到本地缓存
                if downloaded_path != cached_model_path:
                    try:
                        import shutil
                        shutil.copy2(downloaded_path, cached_model_path)
                        self.log_update.emit(f"已将模型复制到本地缓存: {cached_model_path}")
                        return cached_model_path
                    except Exception as e:
                        self.log_update.emit(f"警告: 复制模型到本地缓存失败: {e}，将使用Ultralytics的路径: {downloaded_path}")
                        return downloaded_path
                return downloaded_path
            else:
                self.log_update.emit(f"下载后无法获取有效的模型路径: {model_name_to_download}")
                if os.path.exists(model_name_to_download):
                    self.log_update.emit(f"使用直接存在的模型文件: {model_name_to_download}")
                    return model_name_to_download
                return None

        except Exception as e:
            self.log_update.emit(f"下载模型时出错: {traceback.format_exc()}")
            self.training_error.emit(f"下载模型失败: {model_name_to_download}, 错误: {e}")
            return None

    def _resolve_model_path(self):
        """
        Resolves the actual model name or path to be used for training.
        Handles download, local folder, or custom file logic.
        Returns:
            str: The model string (name or path) for YOLO trainer, or None on error.
        """
        if self.is_from_scratch:
            # For from_scratch, model_type_or_path is like 'yolov8n' (implies .yaml config)
            model_name_no_ext = self.model_type_or_path.replace(".pt", "").replace(".pth", "")
            self.log_update.emit(f"Training from scratch using model configuration: {model_name_no_ext}.yaml (assumed)")
            return model_name_no_ext # Ultralytics will look for <model_name_no_ext>.yaml

        if self.model_source_option == "download":
            # model_type_or_path should be like 'yolov8n.pt'
            model_to_download = self.model_type_or_path
            if not model_to_download.endswith((".pt", ".pth")):
                self.log_update.emit(f"Warning: Model name {model_to_download} for download doesn't end with .pt or .pth. Appending .pt")
                model_to_download += ".pt"
            return self._download_model_if_needed(model_to_download)
            
        elif self.model_source_option == "local_folder":
            # model_type_or_path is 'yolov8n.pt', local_model_search_dir is the folder
            if not self.local_model_search_dir:
                self.log_update.emit("Error: Local model folder selected but no directory provided.")
                self.training_error.emit("本地模型文件夹未指定。")
                return None
            model_filename = self.model_type_or_path
            if not model_filename.endswith((".pt", ".pth")):
                 model_filename += ".pt" # Ensure it's a file name
            
            potential_path = os.path.join(self.local_model_search_dir, model_filename)
            if os.path.isfile(potential_path):
                self.log_update.emit(f"Using local model: {potential_path}")
                return potential_path
            else:
                self.log_update.emit(f"Error: Model file '{model_filename}' not found in local folder: {self.local_model_search_dir}")
                self.training_error.emit(f"在文件夹 {self.local_model_search_dir} 中未找到模型 {model_filename}。")
                return None
                
        elif self.model_source_option == "custom_file":
            # model_type_or_path is already the full path to the .pt file
            custom_path = self.model_type_or_path 
            if not custom_path or not os.path.isfile(custom_path):
                self.log_update.emit(f"Error: Custom weights file path invalid or not found: {custom_path}")
                self.training_error.emit(f"自定义权重文件路径无效: {custom_path}")
                return None
            self.log_update.emit(f"Using custom weights file: {custom_path}")
            return custom_path
            
        else:
            self.log_update.emit(f"Error: Unknown model source option: {self.model_source_option}")
            self.training_error.emit(f"未知的模型来源选项: {self.model_source_option}")
            return None

    def run(self):
        """Run the training process."""
        try:
            self.log_update.emit("开始初始化训练进程...")
            
            # 检查GPU环境
            self._check_gpu_environment()
            
            from ultralytics import YOLO
            import ultralytics
            ultralytics_version = getattr(ultralytics, '__version__', 'unknown')
            self.log_update.emit(f"Ultralytics YOLO 导入成功 (版本: {ultralytics_version})")
            
            self.log_update.emit("正在解析模型路径...")
            actual_model_to_load = self._resolve_model_path()
            if not actual_model_to_load:
                self.log_update.emit("模型路径解析失败，中止训练。")
                self._emit_training_complete(success=False, message="模型路径解析失败。")
                return
            
            self.log_update.emit(f"已解析训练模型: {actual_model_to_load}")
            self.log_update.emit(f"开始 {self.task_type} 训练...")
            self.log_update.emit(f"数据配置: {self.data_yaml_path}")
            self.log_update.emit(f"训练参数: Epochs={self.epochs}, Batch={self.batch_size}, ImgSz={self.img_size}")
            self.log_update.emit(f"输出目录: {self.output_dir}, 项目名称: {self.project_name}")
            self.log_update.emit(f"设备: {self.device if self.device else '自动选择'}")
            self.log_update.emit(f"训练模式: {'从头开始' if self.is_from_scratch else '使用预训练模型'}")
            self.log_update.emit(f"冻结骨干网络: {self.freeze_backbone}")
            if self.other_args:
                self.log_update.emit(f"其他参数: {self.other_args}")
            
            # 初始化模型
            self.log_update.emit("正在初始化模型...")
            try:
                model = YOLO(actual_model_to_load)
                self.log_update.emit(f"模型 '{actual_model_to_load}' 加载/初始化成功。")
            except Exception as e:
                error_msg = f"模型初始化失败: {str(e)}"
                self.log_update.emit(error_msg)
                self._emit_training_complete(success=False, message=error_msg)
                return
            
            # 准备训练参数
            self.log_update.emit("正在准备训练参数...")
            training_args = {
                'data': self.data_yaml_path,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'project': self.output_dir,
                'name': self.project_name,
                'device': self.device if self.device else None,
                'exist_ok': True,
            }
            
            # 处理冻结参数
            if not self.is_from_scratch and self.freeze_backbone:
                if self.task_type == "detect":
                    training_args['freeze'] = 10
                    self.log_update.emit("检测任务微调: 设置 'freeze=10' 冻结骨干网络。")
                elif self.task_type == "classify":
                    training_args['freeze'] = True
                    self.log_update.emit("分类任务微调: 设置 'freeze=True'。")
            
            # 添加其他参数
            for key, value in self.other_args.items():
                if key not in training_args:
                    training_args[key] = value
                else:
                    self.log_update.emit(f"警告: 超参数 '{key}' 与现有训练参数冲突。使用显式设置的值。")
            
            self.log_update.emit(f"最终训练参数: {training_args}")

            # 设置回调
            self.log_update.emit("正在设置训练回调...")
            self._trainer_ref = None

            def on_train_epoch_end_callback(trainer): 
                self._trainer_ref = trainer
                if self._stop_event.is_set():
                    self.log_update.emit("收到停止信号，正在停止训练...")
                    trainer.stop = True 
                    self.log_update.emit("已设置 trainer.stop = True")
                
                current_epoch = trainer.epoch + 1
                progress = int((current_epoch / self.epochs) * 100)
                self.progress_update.emit(progress)
                
                log_str = f"完成第 {current_epoch}/{self.epochs} 轮训练"
                if hasattr(trainer, 'metrics') and trainer.metrics:
                    metrics_str = ", ".join([f"{k.split('/')[-1]}: {v:.4f}" for k, v in trainer.metrics.items()])
                    log_str += f" 指标: {metrics_str}"
                self.log_update.emit(log_str)

            def on_train_end_callback(trainer):
                self._trainer_ref = trainer
                self.log_update.emit("训练结束回调触发")
            
            def on_fit_epoch_end_callback(trainer):
                on_train_epoch_end_callback(trainer)

            # 添加回调
            self.log_update.emit("正在添加回调函数...")
            model.add_callback("on_train_epoch_end", on_train_epoch_end_callback)
            model.add_callback("on_fit_epoch_end", on_fit_epoch_end_callback)
            model.add_callback("on_train_end", on_train_end_callback)
            
            # 开始训练
            self.log_update.emit("开始训练...")
            try:
                results = model.train(**training_args)
                self.log_update.emit("训练调用完成")
            except Exception as e:
                error_msg = f"训练过程出错: {str(e)}"
                self.log_update.emit(error_msg)
                self._emit_training_complete(success=False, message=error_msg)
                return
            
            if self._stop_event.is_set() or (self._trainer_ref and getattr(self._trainer_ref, 'stop', False)):
                self.log_update.emit("训练被用户停止")
                self._emit_training_complete(success=False, message="训练被用户停止。")
            else:
                self.log_update.emit("训练成功完成")
                self._emit_training_complete(success=True)

        except KeyboardInterrupt:
            self.log_update.emit("训练被用户中断 (KeyboardInterrupt)")
            self._emit_training_complete(success=False, message="训练被手动中断。")
        except Exception as e:
            error_msg = traceback.format_exc()
            self.log_update.emit(f"训练过程中发生严重错误: {error_msg}")
            self._emit_training_complete(success=False, message=f"训练中发生严重错误: {str(e)}")
        finally:
            self._trainer_ref = None
            self._stop_event.clear()
    
    def stop(self):
        """Signal the training process to stop."""
        self.log_update.emit("Stop requested for training worker.")
        self._stop_event.set()
        # The callback on_train_epoch_end will check this event and set trainer.stop = True
        # If trainer reference is available and has a stop, it can be called, but event is primary
        if self._trainer_ref and hasattr(self._trainer_ref, 'stop_training'): # Some trainers might have explicit stop_training
            try:
                self._trainer_ref.stop_training = True
                self.log_update.emit("Set trainer.stop_training = True")
            except Exception as e:
                self.log_update.emit(f"Error setting trainer.stop_training: {e}")
        elif self._trainer_ref and hasattr(self._trainer_ref, 'stop'):
             self._trainer_ref.stop = True # This is usually the flag ultralytics checks
             self.log_update.emit("Set trainer.stop = True via stop method direct access")

    def _emit_training_complete(self, success=True, message=None):
        if success:
            self.training_complete.emit()
        else:
            final_message = message if message else "训练未成功完成。"
            self.training_error.emit(final_message)
    
    # GPU check can remain simple
    def _check_gpu(self):
        """Check GPU availability and return device string."""
        # This method is not strictly needed if device is passed from UI
        # and YOLO handles device='' or device=None as auto-select.
        # However, logging it can be useful.
        if self.device and self.device.lower() != 'auto': # If a specific device is requested
            if "cuda" in self.device.lower() or self.device.isdigit():
                if not torch.cuda.is_available():
                    self.log_update.emit(f"Warning: Device {self.device} requested, but CUDA not available. Ultralytics will likely fall back to CPU.")
                else:
                    self.log_update.emit(f"Requested device: {self.device}. CUDA available.")
            return self.device
        
        # Auto-detection if no specific device or 'auto'
        if torch.cuda.is_available():
            self.log_update.emit(f"CUDA available. GPU: {torch.cuda.get_device_name(0)}. Using CUDA.")
            return None # Let Ultralytics auto-select GPU, typically '0'
        else:
            self.log_update.emit("CUDA not available. Using CPU.")
            return "cpu"

# Example of how TrainingTab might call this (from previous TrainingTab logic)
# self.training_worker = TrainingWorker(
# model_name=model_weights, # This is self.model_type_or_path in worker
# data_yaml_path=data_yaml_path,
# epochs=epochs,
# batch_size=batch_size,
# img_size=img_size,
# output_dir=output_dir, # This is parent for project
# device=device,
# task=task, # This is self.task_type in worker
# is_from_scratch=train_from_scratch,
# freeze_backbone=freeze_backbone, # This can be part of other_args or a direct param
# other_args=other_args,
# model_source_option=self.model_source_option,
# local_model_search_dir=self.local_model_folder_edit.text() if self.local_folder_model_radio.isChecked() else None,
# project_name=self.project_name_edit.text()
# )
# self.training_thread = QThread()
# self.training_worker.moveToThread(self.training_thread)
# self.training_worker.progress_update.connect(self.update_progress)
# self.training_worker.log_update.connect(self.log_message) # Connect to log_message in TrainingTab
# self.training_worker.training_complete.connect(self.on_training_complete)
# self.training_worker.training_error.connect(self.on_training_error)
# self.training_thread.started.connect(self.training_worker.run)
# self.training_thread.start() 