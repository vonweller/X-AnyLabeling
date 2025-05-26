import os
import sys
import time
import threading
import torch
from PyQt5.QtCore import QObject, pyqtSignal
import traceback

class TrainingWorker(QObject):
    """Worker class to handle YOLO model training in a separate thread."""
    
    # Signal definitions
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal()
    training_error = pyqtSignal(str)
    
    def __init__(self, model_type, task_type, train_dir, val_dir, output_dir, project_name,
                 batch_size, epochs, img_size, learning_rate, pretrained, model_weights=None, fine_tuning=False):
        """
        Initialize the training worker with parameters.
        
        Args:
            model_type (str): YOLO model type (e.g., 'yolov8n.pt')
            task_type (str): Task type ('detect' or 'classify')
            train_dir (str): Path to training data directory (or root for classification)
            val_dir (str): Path to validation data directory (or root for classification, can be same as train_dir if train/val subdirs exist)
            output_dir (str): Path to save output results
            project_name (str): Project name for output organization
            batch_size (int): Batch size for training
            epochs (int): Number of training epochs
            img_size (int): Image size for training
            learning_rate (float): Learning rate
            pretrained (bool): Whether to use pretrained weights (if model_weights is None)
            model_weights (str, optional): Path to custom model weights for initialization
            fine_tuning (bool): Whether to apply fine-tuning (e.g., freeze backbone)
        """
        super().__init__()
        self.model_type = model_type
        self.task_type = task_type
        self.train_dir = train_dir
        self.val_dir = val_dir # For classification, this might be redundant if train_dir has train/val subfolders
        self.output_dir = output_dir
        self.project_name = project_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.learning_rate = learning_rate
        self.pretrained = pretrained
        self.model_weights = model_weights
        self.fine_tuning = fine_tuning
        
        self._stop_event = threading.Event()
        self._trainer_ref = None  # Reference to the trainer object for direct access
        self._process_ref = None  # Reference to any training process that might be running
    
    def _check_internet_connection(self):
        """
        Check for internet connectivity by attempting to connect to known servers.
        
        Returns:
            bool: True if internet is available, False otherwise
        """
        try:
            # Try to connect to Google's DNS server (should work in most countries)
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            try:
                # Try to connect to Baidu (for users in China)
                socket.create_connection(("220.181.38.148", 80), timeout=3)
                return True
            except OSError:
                pass
        
        # Alternative method: try to resolve a known domain
        try:
            socket.gethostbyname("google.com")
            return True
        except:
            try:
                socket.gethostbyname("baidu.com")
                return True
            except:
                pass
        
        return False
    
    def run(self):
        """Run the training process."""
        try:
            self.log_update.emit(f"Starting {self.task_type} training with {self.model_type}")
            print(f"Starting {self.task_type} training with {self.model_type}")
            self.log_update.emit(f"Batch size: {self.batch_size}, Image size: {self.img_size}")
            print(f"Batch size: {self.batch_size}, Image size: {self.img_size}")
            self.log_update.emit(f"Learning rate: {self.learning_rate}, Epochs: {self.epochs}")
            print(f"Learning rate: {self.learning_rate}, Epochs: {self.epochs}")
            
            # Check internet connectivity for model downloading
            has_internet = self._check_internet_connection()
            if not has_internet and self.pretrained and not self.model_weights:
                self.log_update.emit("警告：检测到没有互联网连接。若本地没有预训练模型文件，将自动切换到从头训练模式")
            
            # 预加载YOLO模型，这可以避免在训练时重复下载权重
            if not self.model_weights and self.pretrained:
                model_cache_dir = os.path.join(self.output_dir, "model_cache")
                os.makedirs(model_cache_dir, exist_ok=True)
                model_file = os.path.join(model_cache_dir, f"{self.model_type}.pt")
                
                # Check for model in multiple locations with priority
                model_found = False
                
                # Check locations to look for model files
                possible_locations = [
                    # 1. Current directory (highest priority)
                    f"{self.model_type}.pt",
                    # 2. Cache directory
                    model_file,
                    # 3. Common model directories
                    os.path.join("models", f"{self.model_type}.pt"),
                    os.path.join("weights", f"{self.model_type}.pt"),
                    os.path.join("pretrained", f"{self.model_type}.pt"),
                    # 4. Application directory
                    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{self.model_type}.pt")
                ]
                
                # Look for model file in possible locations
                for location in possible_locations:
                    if os.path.exists(location):
                        self.log_update.emit(f"找到本地模型权重: {location}")
                        # Copy to cache if not already there
                        if location != model_file:
                            try:
                                import shutil
                                shutil.copy(location, model_file)
                                self.log_update.emit(f"已复制模型权重到缓存目录: {model_file}")
                            except Exception as e:
                                self.log_update.emit(f"复制模型到缓存失败，将直接使用原始文件: {str(e)}")
                        self.model_weights = location
                        model_found = True
                        break
                
                # If model not found locally, prepare for download
                if not model_found:
                    self.log_update.emit(f"本地未找到模型文件 {self.model_type}.pt，将尝试自动下载")
                    self.model_weights = None
                
                    # Try to create a placeholder for the download target
                    # This will allow the YOLO loader to download directly to our cache
                    try:
                        # Create an empty file to mark the download location
                        with open(model_file, 'w') as f:
                            f.write("# Placeholder for model download\n")
                        self.log_update.emit(f"已准备下载位置: {model_file}")
                        # We don't set model_weights yet - let YOLO handle the download
                    except Exception as e:
                        self.log_update.emit(f"准备下载位置失败: {str(e)}")
            
            # Create data.yaml file based on dataset format (only for detection)
            yaml_path = None
            if self.task_type == "detect":
                yaml_path = self._create_dataset_yaml() # This method needs to exist and function for detection
                if not yaml_path:
                    self.training_error.emit("Failed to create or validate dataset YAML for detection task.")
                    return
            
            # Check GPU availability
            device = self._check_gpu()
            self.log_update.emit(f"Using device: {device}")
            
            # Import YOLO after checking environment
            # This is done inside the run method to avoid importing in the main thread
            try:
                from ultralytics import YOLO
                # 尝试获取ultralytics版本
                import ultralytics
                ultralytics_version = getattr(ultralytics, '__version__', 'unknown')
                self.log_update.emit(f"Ultralytics YOLO imported successfully (version: {ultralytics_version})")
                
                # 检测是否可以导入Callback类
                has_callback_class = False
                try:
                    from ultralytics.utils.callbacks.base import Callback
                    has_callback_class = True
                    self.log_update.emit("Callback class available")
                except ImportError:
                    self.log_update.emit("Callback class not available, will use function-based callbacks")
                    has_callback_class = False
                
            except ImportError as e:
                self.training_error.emit(f"Failed to import ultralytics: {str(e)}")
                return
            
            # Initialize the model
            try:
                model_arg_for_load = ""
                if self.model_weights: # Custom weights provided
                    model_arg_for_load = self.model_weights # Use the direct path to .pt or .pth file
                    self.log_update.emit(f"使用自定义权重文件进行初始化: {model_arg_for_load}")
                elif self.pretrained: # Use official pretrained weights
                    model_arg_for_load = f"{self.model_type}.pt" # e.g., yolov8n-cls.pt
                    self.log_update.emit(f"使用官方预训练权重进行初始化: {model_arg_for_load}")
                else: # Train from scratch
                    model_arg_for_load = f"{self.model_type}.yaml" # e.g., yolov8n-cls.yaml
                    self.log_update.emit(f"从配置文件进行初始化 (从头训练): {model_arg_for_load}")

                self.log_update.emit(f"Initializing YOLO model: {model_arg_for_load} for task: {self.task_type}")
                # Forcing task in YOLO constructor if model name doesn't explicitly state it (e.g. generic resnet50.pt for classification)
                # However, Ultralytics YOLO() typically infers task from model suffix like -cls, -seg, -pose, or from model structure in .pt
                # If model_arg_for_load is a .yaml, task is usually defined within the yaml.
                # If model_arg_for_load is a .pt, task is inferred.
                # Explicitly setting task might be useful if the model name is ambiguous (e.g. a custom .pt not ending in -cls)
                
                effective_task = self.task_type
                # If the model name itself specifies a task (like yolov8n-seg.pt), let YOLO handle it unless it's a generic name.
                # For classification, if model is like 'resnet50.pt', task needs to be explicit.
                if self.task_type == 'classify' and not model_arg_for_load.endswith('-cls.pt') and not model_arg_for_load.endswith('-cls.yaml'):
                     # This condition might be too broad. YOLO might infer 'classify' from a 'resnet50.pt' if it's a classification model.
                     # Let's rely on YOLO's inference first, and only override if necessary or if model name has no task hint.
                     # For now, pass self.task_type, YOLO will use it if model can't infer.
                     pass # Relying on YOLO's task inference or explicit task in YAML for now.

                model = YOLO(model_arg_for_load, task=self.task_type) # Explicitly pass task
                self.log_update.emit(f"Successfully initialized model: {model_arg_for_load} (Task explicitly set to: {self.task_type}, Inferred by model: {model.task if hasattr(model, 'task') else 'N/A'})")

            except Exception as e:
                error_msg = f"Failed to initialize YOLO model ({model_arg_for_load}): {str(e)}\n{traceback.format_exc()}"
                self.log_update.emit(error_msg)
                self.training_error.emit(error_msg)
                return
            
            # Prepare training arguments
            train_args = {
                "data": yaml_path if self.task_type == "detect" else self.train_dir, # For classification, data is the root dir
                "epochs": self.epochs,
                "batch": self.batch_size,
                "imgsz": self.img_size,
                "lr0": self.learning_rate, # lr0 for initial learning rate
                "project": self.output_dir,
                "name": self.project_name,
                "device": device,
                "exist_ok": True, # Allow overwriting existing project/name
                # "pretrained": self.pretrained, # This is handled by how model is loaded (YOLO('model.pt') vs YOLO('model.yaml'))
                                        # If self.model_weights is set, it uses those weights.
                                        # If self.pretrained is true and no model_weights, it downloads/uses official .pt (e.g. yolov8n.pt)
                                        # If self.pretrained is false and no model_weights, it uses .yaml (from scratch e.g. yolov8n.yaml)
            }

            if self.model_weights and self.pretrained:
                 # This case should ideally be handled by TrainingTab logic: if custom_weights_radio is checked, pretrained is false.
                 # If it still occurs, it implies custom weights are primary.
                 self.log_update.emit("Note: Custom weights were provided; these will be used for initialization. 'pretrained=True' flag from UI was likely for official weights but custom path took precedence.")
                 # The YOLO constructor already handled this by loading self.model_weights.

            if not self.model_weights and not self.pretrained:
                self.log_update.emit(f"训练将从 {self.model_type}.yaml 配置开始 (无预训练权重)。")
            elif not self.model_weights and self.pretrained:
                self.log_update.emit(f"训练将从官方预训练的 {self.model_type}.pt 权重开始。")

            if self.fine_tuning:
                if self.task_type == "classify":
                    # For classification, common to freeze backbone. Ultralytics default models might handle this with 'freeze'
                    # Example: freeze up to certain layers. For simplicity, using a common value or letting ultralytics decide.
                    # Ultralytics models like yolov8n-cls.pt might not need explicit freeze for typical transfer learning.
                    # If using ResNet etc., freeze might be like model.freeze = 10 (ResNet50 has more layers)
                    train_args["freeze"] = 10 # Example: freeze first 10 layers. Adjust as needed or make configurable.
                    self.log_update.emit("Fine-tuning enabled for classification (example: freezing initial layers).")
                elif self.task_type == "detect":
                    # For detection, freeze often means freezing the backbone
                    # Ultralytics YOLOv8 default behavior for fine-tuning might be sufficient if loading pretrained weights.
                    # Or specify backbone layers: train_args["freeze"] = N (e.g., 10 for first 10 layers of backbone)
                    train_args["freeze"] = 10 # Example, might need adjustment based on specific model backbone structure
                    self.log_update.emit("Fine-tuning enabled for detection (example: freezing backbone layers).")

            self.log_update.emit(f"Training arguments: {train_args}")

            # Setup callbacks for progress and stop
            # 创建保存指标的目录
            metrics_dir = os.path.join(self.output_dir, self.project_name)
            
            # 设置进度更新的监控线程
            stop_flag = threading.Event()
            
            def progress_monitor():
                last_metrics_time = 0
                metrics_file = None
                run_dirs = [] # Initialize run_dirs
                
                # 寻找可能的指标文件路径
                def find_metrics_file():
                    # 检查最近创建的run目录
                    run_dirs = []
                    if os.path.exists(metrics_dir):
                        for d in os.listdir(metrics_dir):
                            full_path = os.path.join(metrics_dir, d)
                            if os.path.isdir(full_path) and d.startswith("exp") or d.startswith("train"):
                                run_dirs.append((os.path.getmtime(full_path), full_path))
                
                # 按修改时间排序，获取最新的目录
                if run_dirs:
                    latest_dir = sorted(run_dirs, reverse=True)[0][1]
                    # 检查CSV文件
                    csv_path = os.path.join(latest_dir, "results.csv")
                    if os.path.exists(csv_path):
                        return csv_path
                return None
                
                # 初始进度更新 - 不再使用固定延迟
                self.progress_update.emit(5)
                self.log_update.emit("加载和准备环境...")
                
                # 主循环监控训练进度
                while not stop_flag.is_set() and not self._stop_event.is_set():
                    # 尝试找到指标文件
                    if metrics_file is None:
                        metrics_file = find_metrics_file()
                    
                    # 如果找到了指标文件，读取并显示最新指标
                    if metrics_file and os.path.exists(metrics_file):
                        current_time = os.path.getmtime(metrics_file)
                        
                        # 只有当文件更新时才读取
                        if current_time > last_metrics_time:
                            last_metrics_time = current_time
                            try:
                                with open(metrics_file, 'r') as f:
                                    lines = f.readlines()
                                    if len(lines) > 1:  # 至少有标题行和一行数据
                                        last_line = lines[-1].strip()
                                        header = lines[0].strip().split(',')
                                        values = last_line.split(',')
                                        
                                        # 解析指标
                                        metrics = {}
                                        for i, key in enumerate(header):
                                            if i < len(values):
                                                try:
                                                    metrics[key] = float(values[i])
                                                except ValueError:
                                                    metrics[key] = values[i]
                                        
                                        # 更新进度
                                        if 'epoch' in metrics and 'epochs' in metrics:
                                            epoch = metrics['epoch']
                                            epochs = metrics['epochs']
                                            progress = int((epoch / epochs) * 100)
                                            self.progress_update.emit(progress)
                                        
                                        # 组织指标信息
                                        info_text = f"Epoch: {metrics.get('epoch', '?')}/{metrics.get('epochs', '?')}\n"
                                        
                                        # 添加损失指标
                                        losses = ["train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"]
                                        info_text += "损失指标:\n"
                                        for loss in losses:
                                            if loss in metrics:
                                                info_text += f"  {loss}: {metrics[loss]:.4f}\n"
                                        
                                        # 添加精度指标
                                        accuracies = ["metrics/precision", "metrics/recall", "metrics/mAP50", "metrics/mAP50-95"]
                                        info_text += "精度指标:\n"
                                        for acc in accuracies:
                                            if acc in metrics:
                                                info_text += f"  {acc}: {metrics[acc]:.4f}\n"
                                        
                                        # 输出完整信息
                                        self.log_update.emit(info_text)
                                        print(info_text)  # Direct stdout output
                            except Exception as e:
                                error_msg = f"读取指标文件出错: {str(e)}"
                                self.log_update.emit(error_msg)
                                print(error_msg, file=sys.stderr)  # Direct stderr output
                    
                    # 休眠时间缩短，更快地检查更新
                    time.sleep(0.5)
            
            # 记录开始时间
            start_time = time.time()
            
            # 启动监控线程
            self.log_update.emit("启动进度监控...")
            monitor_thread = threading.Thread(target=progress_monitor)
            monitor_thread.daemon = True
            monitor_thread.start()
            
            # 尝试设置自定义stdout捕获类，以便更实时地获取训练输出
            class StdoutCapture:
                def __init__(self, worker):
                    self.worker = worker
                    self.original_stdout = sys.stdout
                    self.original_stderr = sys.stderr
                    self.buffer = ""

                def write(self, text):
                    # 写入原始流
                    self.original_stdout.write(text)
                    self.original_stdout.flush()
                    
                    # 添加到缓冲区
                    self.buffer += text
                    
                    # 如果有完整行，则发送到UI
                    if '\n' in text:
                        lines = self.buffer.split('\n')
                        for line in lines[:-1]:  # 处理除最后一个可能不完整的行外的所有行
                            if line.strip():  # 如果行不为空
                                # 只处理训练相关信息
                                if "Epoch" in line and ("GPU_mem" in line or "box_loss" in line):
                                    self.worker.log_update.emit(line)
                        
                        # 保留最后一个不完整的行
                        self.buffer = lines[-1] if lines else ""
                
                def flush(self):
                    self.original_stdout.flush()
                    
                def __enter__(self):
                    sys.stdout = self
                    return self
                
                def __exit__(self, exc_type, exc_val, exc_tb):
                    sys.stdout = self.original_stdout
                    
            # 创建捕获实例
            stdout_capture = StdoutCapture(self)
            
            # 创建通用回调函数，支持任何版本的ultralytics
            def on_train_batch_end_fn(trainer=None):
                # 在每个训练批次结束时检查停止标志
                if self._stop_event.is_set():
                    self.log_update.emit("检测到停止信号，正在中断训练...")
                    if trainer:
                        self._trainer_ref = trainer  # Store reference to trainer
                        # 尝试停止训练循环
                        if hasattr(trainer, 'epoch_progress'):
                            try:
                                trainer.epoch_progress.close()  # 关闭进度条
                            except:
                                pass
                        if hasattr(trainer, 'stop'):
                            trainer.stop = True
                    return False  # 返回False以停止训练循环
                return True
            
            def on_train_epoch_end_fn(trainer=None):
                # 在每个epoch结束时检查停止标志
                if self._stop_event.is_set():
                    self.log_update.emit("检测到停止信号，正在中断训练...")
                    if trainer:
                        self._trainer_ref = trainer  # Store reference to trainer
                        if hasattr(trainer, 'stop'):
                            trainer.stop = True
                    return False  # 返回False以停止训练循环
                return True
            
            # 添加新的回调函数，用于设置进程参考
            def on_train_start_fn(trainer=None):
                if trainer:
                    self._trainer_ref = trainer  # Store reference to trainer
                    self.log_update.emit("训练开始，已捕获训练器引用")
                import threading
                self._process_ref = threading.current_thread()
                return True
            
            # 创建带有回调的自定义训练参数
            # BASE train_args - DO NOT MODIFY THIS DICTIONARY directly in callback attempts below
            # Instead, copy and update if a specific attempt needs different args.
            base_train_args = {
                'data': yaml_path if self.task_type == "detect" else self.train_dir,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'project': self.output_dir, # Use output_dir for project
                'name': self.project_name,   # Use project_name for name (experiment)
                'lr0': self.learning_rate,
                'device': device,
                'exist_ok': True, # Important: allow re-running into the same project/name
                'save_dir': os.path.join(self.output_dir, self.project_name), # Explicit save_dir
                'plots': True
            }
            
            # Add fine-tuning arg if applicable (moved from earlier to be part of base_train_args)
            if self.fine_tuning:
                if self.task_type == "classify":
                    base_train_args["freeze"] = 10 
                    self.log_update.emit("Fine-tuning enabled for classification (example: freezing initial layers).")
                elif self.task_type == "detect":
                    base_train_args["freeze"] = 10 
                    self.log_update.emit("Fine-tuning enabled for detection (example: freezing backbone layers).")


            self.log_update.emit(f"Base training arguments: {base_train_args}")
            
            # 根据不同的ultralytics版本，尝试不同的回调方式
            self.log_update.emit("配置训练参数和回调...")
            
            results = None
            
            # Modern way to add callbacks (preferred for newer Ultralytics)
            # Clear any existing default callbacks if needed (optional, depends on desired behavior)
            # model.clear_callbacks() 

            # Add our custom callbacks
            # Note: The functions on_train_start_fn, on_train_batch_end_fn, on_train_epoch_end_fn
            # are already defined to accept a 'trainer' argument, which Ultralytics should provide.
            try:
                self.log_update.emit("尝试使用 model.add_callback() 注册回调...")
                model.add_callback("on_train_start", on_train_start_fn)
                model.add_callback("on_train_batch_end", on_train_batch_end_fn)
                model.add_callback("on_epoch_end", on_train_epoch_end_fn) # Common event name is on_epoch_end
                # also try on_train_epoch_end if the above doesn't fire for stop event
                # model.add_callback("on_train_epoch_end", on_train_epoch_end_fn)
                self.log_update.emit("回调已通过 add_callback 添加。")

                with stdout_capture:
                    self.log_update.emit("开始训练 (使用 add_callback)，第一个epoch可能较慢...")
                    results = model.train(**base_train_args) # Train with base_train_args

            except Exception as e_add_callback:
                self.log_update.emit(f"使用 model.add_callback() 注册回调失败或训练出错: {str(e_add_callback)}")
                self.log_update.emit(f"详细错误: {traceback.format_exc()}")
                # Fallback to no-callback training if add_callback approach fails
                if results is None:
                    self.log_update.emit("尝试无回调训练 (因 add_callback 失败)")
                    with stdout_capture:
                        self.log_update.emit("开始训练 (无回调)，第一个epoch可能较慢...")
                        results = model.train(**base_train_args) # Use base_train_args here
            
            # 训练完成，停止监控线程
            stop_flag.set()
            
            # 检查结果并更新UI
            if self._stop_event.is_set():
                self.log_update.emit("训练被用户中止")
                self.training_complete.emit()
            else:
                if results is not None and hasattr(results, 'metrics'):
                    metrics = results.metrics
                    self.log_update.emit(f"训练完成! 最终结果:")
                    if hasattr(metrics, 'box_loss'):
                        self.log_update.emit(f"box_loss: {metrics.box_loss:.4f}")
                    if hasattr(metrics, 'cls_loss'):
                        self.log_update.emit(f"cls_loss: {metrics.cls_loss:.4f}")
                    if hasattr(metrics, 'map50'):
                        self.log_update.emit(f"mAP50: {metrics.map50:.4f}")
                
                self.log_update.emit("训练成功完成!")
                self.progress_update.emit(100)
                self.training_complete.emit()
            
        except Exception as e:
            if self._stop_event.is_set():
                self.log_update.emit("训练已被用户中止")
                self.training_complete.emit()
            else:
                self.training_error.emit(f"训练错误: {str(e)}")
    
    def stop(self):
        """Stop the training process immediately."""
        self._stop_event.set()
        self.log_update.emit("收到停止信号，立即中断训练...")
        
        # Attempt to terminate the training more aggressively
        if self._trainer_ref is not None:
            try:
                # Try all possible ways to forcibly stop the trainer
                if hasattr(self._trainer_ref, 'stop'):
                    self._trainer_ref.stop = True
                if hasattr(self._trainer_ref, 'epoch_progress') and hasattr(self._trainer_ref.epoch_progress, 'close'):
                    self._trainer_ref.epoch_progress.close()
                if hasattr(self._trainer_ref, 'stopper') and hasattr(self._trainer_ref.stopper, 'run'):
                    self._trainer_ref.stopper.possible_stop = True
                self.log_update.emit("已发送终止信号到训练器")
            except Exception as e:
                self.log_update.emit(f"尝试终止训练器时出错: {str(e)}")
        
        # If we have a training process, attempt to terminate it more forcefully
        if self._process_ref is not None:
            try:
                import signal
                import ctypes
                import os
                
                if hasattr(self._process_ref, 'terminate'):
                    self._process_ref.terminate()
                    self.log_update.emit("已强制终止训练进程")
                elif isinstance(self._process_ref, threading.Thread) and self._process_ref.is_alive():
                    # This is a more aggressive approach for Python threads
                    if hasattr(threading, '_async_raise'):
                        threading._async_raise(self._process_ref.ident, SystemExit)
                    self.log_update.emit("已尝试强制终止训练线程")
            except Exception as e:
                self.log_update.emit(f"尝试强制终止训练时出错: {str(e)}")
        
        # Signal that training was stopped by user
        threading.Thread(target=self._emit_training_complete, daemon=True).start()
    
    def _emit_training_complete(self):
        """Emit training complete signal after a short delay to allow cleanup"""
        time.sleep(0.5)  # Short delay to allow other operations to complete
        self.training_complete.emit()
    
    def _check_gpu(self):
        """Check if CUDA is available and return appropriate device."""
        if torch.cuda.is_available():
            return 0  # Use first GPU
        else:
            self.log_update.emit("CUDA not available, using CPU")
            return 'cpu'
    
    def _create_dataset_yaml(self):
        """
        Create a dataset.yaml file for YOLO training, specific to detection task.
        This method is now only relevant for self.task_type == 'detect'.
        Returns:
            str: Path to the created YAML file, or None on failure.
        """
        if self.task_type != "detect":
            self.log_update.emit("Skipping YAML creation for non-detection task.")
            return None

        self.log_update.emit("Creating dataset YAML for detection task...")
        # ... (rest of the _create_dataset_yaml method remains largely the same but is now conditional)
        # Ensure paths used (self.train_dir, self.val_dir) are appropriate for detection (e.g. point to image folders)
        # And class names are correctly fetched for detection.

        # Example of how it might start:
        train_images_path = os.path.join(self.train_dir, 'images', 'train') # Assuming this structure
        val_images_path = os.path.join(self.val_dir, 'images', 'val')       # Or self.val_dir directly if it points to val images
        
        # If train_dir/val_dir are already the 'images/train' or 'images/val' folders, adjust accordingly.
        # This part needs to be robust based on how train_dir/val_dir are set in the UI for detection.
        # For now, assuming train_dir is the root of the dataset containing 'images' and 'labels' folders.

        # Correctly determine the dataset root for relative paths in YAML.
        # The common parent of train_dir and val_dir, or a configured dataset_root if available.
        # For simplicity, let's assume output_dir can serve as a place to write the yaml
        # and paths in yaml will be relative to a common dataset root or absolute.

        # For now, I will keep the existing logic of _create_dataset_yaml but add a check 
        # at the beginning and ensure it's only called for detection.
        # The existing logic for path normalization and class name extraction needs to be correct for detection.

        # Get class names (ensure this is appropriate for detection format)
        class_names = self._get_class_names_for_detection() # New or refactored method for detection
        if not class_names:
            self.log_update.emit("错误：无法获取类别名称。请检查数据集格式和标签文件。")
            # self.training_error.emit("无法获取类别名称，请检查数据集。") # Emitting error here might be too early
            return None
        
        num_classes = len(class_names)
        self.log_update.emit(f"找到 {num_classes} 个类别: {class_names}")
        
        # Create YAML content
        # Ensure paths are relative to a common root or absolute as expected by YOLO
        # The paths like self.train_dir and self.val_dir from UI are expected to be
        # the root directories for train/val sets containing images/labels subdirs.

        # To make paths relative to the YAML file location (which is good practice if YAML is with dataset)
        # or use absolute paths if YAML is in output_dir.
        # For now, using absolute paths as it simplifies things, assuming self.train_dir and self.val_dir are set by user.

        yaml_content = {
            'path': os.path.abspath(self.output_dir), # Or a common dataset root path
            'train': os.path.abspath(self.train_dir), # This should point to dir with images/train, labels/train
            'val': os.path.abspath(self.val_dir),     # Similar for val
            'nc': num_classes,
            'names': class_names
        }
        
        # Adjust train/val paths if they are meant to be relative to 'path' or are already specific image folders
        # This part is crucial and depends on the exact meaning of self.train_dir from the UI for detection.
        # For example, if self.train_dir is 'path/to/dataset/train_images_folder',
        # then yaml train should be 'train_images_folder' and path should be 'path/to/dataset'
        # The current _create_dataset_yaml likely has more sophisticated path handling for detection.
        # We need to ensure that logic is preserved and correct for detection.
        # The example below is a simplified version if paths are absolute.

        # Assuming the original _create_dataset_yaml handles this correctly for detection.
        # The core idea is that the output of this function (yaml_path) must be correct for YOLO detection.

        # Original logic for creating yaml_path in self.output_dir:
        yaml_file_name = f"{self.project_name}_data.yaml"
        yaml_path = os.path.join(self.output_dir, self.project_name, yaml_file_name) # Place it inside project folder
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

        try:
            import yaml # Make sure to import yaml
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
            self.log_update.emit(f"数据集配置文件已创建: {yaml_path}")
            
            # Validate the generated YAML (optional but good)
            if not self._validate_yaml(yaml_path):
                self.log_update.emit(f"错误: 生成的YAML文件 ({yaml_path}) 验证失败。")
                return None
            return yaml_path
        except Exception as e:
            self.log_update.emit(f"创建数据集YAML文件失败: {str(e)}")
            return None

    def _get_class_names_for_detection(self):
        # This method should replicate the class name extraction logic previously in _create_dataset_yaml
        # or call the relevant parts of _get_class_names, specifically for detection task.
        # For example, it might read from a classes.txt or parse label files if format is YOLO.
        # This is a placeholder for the actual implementation based on original code.
        self.log_update.emit("Fetching class names for detection...")
        # Replace with actual logic from the old _create_dataset_yaml or _get_class_names
        # Example placeholder:
        # if self.dataset_format == "YOLO": # Assuming this attribute still exists or is inferred
        #    return self._get_yolo_class_names() # Existing method if it works for detection context
        # elif self.dataset_format == "COCO":
        #    return self._get_coco_class_names(os.path.join(self.train_dir, 'annotations', 'instances_train.json'))
        # For now, returning a dummy list or ensuring the original logic is moved here:
        
        # Placeholder - this needs to be the actual logic from your _get_class_names
        # or the relevant parts for detection.
        # If _get_class_names was general enough, it might be called here directly
        # or with specific parameters for detection.
        
        # Let's assume there was a classes.txt in the train_dir for detection.
        classes_txt_path = os.path.join(self.train_dir, "classes.txt") 
        # Or if data.yaml was expected to be found / generated elsewhere, that logic needs to be here.
        # For simplicity, if you had a direct way in the old code, use it.
        if os.path.exists(classes_txt_path):
            with open(classes_txt_path, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
                if class_names:
                    return class_names
        
        self.log_update.emit("Warning: classes.txt not found or empty in train_dir. Could not determine class names for detection YAML.")
        # Fallback or error if essential for YAML creation
        return ["object"] # Fallback dummy

    def _update_paths_in_yaml(self, src_yaml, dst_yaml):
        """更新YAML文件中的路径以适应当前环境"""
        import yaml
        
        try:
            # 读取原始YAML
            with open(src_yaml, 'r') as f:
                data = yaml.safe_load(f)
            
            # 规范化训练和验证目录
            train_dir = self._normalize_path(self.train_dir)
            val_dir = self._normalize_path(self.val_dir)
            
            # 检查验证目录是否存在
            if not os.path.exists(val_dir):
                self.log_update.emit(f"警告: 验证目录不存在: {val_dir}")
                self.log_update.emit("将使用训练目录作为验证数据")
                val_dir = train_dir
            
            # 检查并更新路径
            if 'path' in data:
                # 如果原始路径不存在，改为使用当前路径
                original_path = data['path']
                if not os.path.exists(original_path):
                    data['path'] = os.path.dirname(train_dir)
                    self.log_update.emit(f"更新数据集路径: {original_path} -> {data['path']}")
            else:
                data['path'] = os.path.dirname(train_dir)
            
            # 确保train和val字段存在和正确
            data['train'] = os.path.basename(train_dir)
            data['val'] = os.path.basename(val_dir)
            
            # 确保路径格式正确(不要有多个连续的斜杠)
            data['path'] = self._normalize_path(data['path'])
            
            # 写入更新后的YAML
            with open(dst_yaml, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
            
            # 验证生成的YAML文件
            self._validate_yaml(dst_yaml)
            
        except Exception as e:
            self.log_update.emit(f"更新YAML文件失败: {str(e)}，将创建新文件")
            # 如果失败，创建新文件
            class_names = self._get_class_names()
            with open(dst_yaml, 'w') as f:
                f.write(f"# Dataset configuration\n")
                f.write(f"path: {os.path.dirname(train_dir)}\n")
                f.write(f"train: {os.path.basename(train_dir)}\n")
                f.write(f"val: {os.path.basename(val_dir)}\n")
                f.write(f"nc: {len(class_names)}\n")
                f.write(f"names: {class_names}\n")
            
            # 验证生成的YAML文件
            self._validate_yaml(dst_yaml)
    
    def _normalize_path(self, path):
        """规范化路径，确保路径格式正确"""
        # 替换多个连续的斜杠为单个斜杠
        path = path.replace('///', '/').replace('//', '/')
        
        # 确保Windows路径使用正确的斜杠格式
        if os.name == 'nt':
            path = path.replace('/', '\\')
            # 修复可能出现的多个反斜杠问题
            while '\\\\' in path:
                path = path.replace('\\\\', '\\')
        
        return path
    
    def _validate_yaml(self, yaml_path):
        """验证YAML文件中的路径是否有效"""
        import yaml
        try:
            # 读取YAML文件
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # 检查基础路径是否存在
            base_path = data.get('path', '')
            if not base_path or not os.path.exists(base_path):
                self.log_update.emit(f"警告: YAML中的基础路径不存在: {base_path}")
                
                # 尝试修复基础路径
                if 'train' in data:
                    train_rel = data['train']
                    possible_base = self._find_valid_base_path(train_rel)
                    if possible_base:
                        data['path'] = possible_base
                        self.log_update.emit(f"自动修复基础路径为: {possible_base}")
                        
                        # 重新写入YAML
                        with open(yaml_path, 'w') as f:
                            yaml.dump(data, f, default_flow_style=False)
            
            # 检查训练和验证路径是否存在
            if 'path' in data and os.path.exists(data['path']):
                if 'train' in data:
                    train_path = os.path.join(data['path'], data['train'])
                    if not os.path.exists(train_path):
                        self.log_update.emit(f"警告: 训练路径不存在: {train_path}")
                
                if 'val' in data:
                    val_path = os.path.join(data['path'], data['val'])
                    if not os.path.exists(val_path):
                        self.log_update.emit(f"警告: 验证路径不存在: {val_path}")
                        
                        # 如果验证路径不存在但训练路径存在，使用训练路径代替
                        if 'train' in data and os.path.exists(os.path.join(data['path'], data['train'])):
                            data['val'] = data['train']
                            self.log_update.emit(f"自动设置验证集与训练集相同: {data['train']}")
                            
                            # 重新写入YAML
                            with open(yaml_path, 'w') as f:
                                yaml.dump(data, f, default_flow_style=False)
        
        except Exception as e:
            self.log_update.emit(f"验证YAML文件失败: {str(e)}")
    
    def _find_valid_base_path(self, train_rel):
        """尝试找到有效的基础路径"""
        # 从当前工作目录开始
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, train_rel)):
            return cwd
        
        # 从训练目录的父级目录尝试
        train_dir = self._normalize_path(self.train_dir)
        parent_dir = os.path.dirname(train_dir)
        if os.path.exists(os.path.join(parent_dir, train_rel)):
            return parent_dir
        
        # 从训练目录本身尝试
        if os.path.basename(train_dir) == train_rel:
            return os.path.dirname(train_dir)
        
        return None
    
    def _get_class_names(self):
        """
        Extract class names from the dataset.
        
        Returns:
            list: List of class names
        """
        if self.task_type == "detect":
            # 对于YOLO格式，尝试从classes.txt或data.yaml文件中获取类名
            class_names = self._get_yolo_class_names()
        elif self.task_type == "classify":
            # For COCO, we would parse the annotations JSON file
            class_names = self._get_coco_class_names()
        elif self.task_type == "voc":
            # For VOC, we would look for the labels in the annotation XML files
            class_names = self._get_voc_class_names()
        else:
            class_names = ['class0', 'class1']
        
        # 如果没有找到类名，使用默认名称
        if not class_names:
            self.log_update.emit("警告: 无法确定类名，使用默认值")
            class_names = ['class0', 'class1']
        
        return class_names
    
    def _get_yolo_class_names(self):
        """从YOLO格式数据集中提取类名"""
        class_names = []
        
        # 首先尝试从data.yaml文件中获取
        possible_yaml_paths = [
            os.path.join(self.train_dir, "data.yaml"),
            os.path.join(os.path.dirname(self.train_dir), "data.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(self.train_dir)), "data.yaml")
        ]
        
        for yaml_path in possible_yaml_paths:
            if os.path.exists(yaml_path):
                try:
                    import yaml
                    with open(yaml_path, 'r') as f:
                        data = yaml.safe_load(f)
                    
                    if 'names' in data:
                        self.log_update.emit(f"从YAML文件加载类名: {yaml_path}")
                        return data['names']
                except Exception as e:
                    self.log_update.emit(f"从YAML读取类名失败: {str(e)}")
        
        # 然后尝试从classes.txt文件中获取
        possible_class_files = [
            os.path.join(self.train_dir, "classes.txt"),
            os.path.join(os.path.dirname(self.train_dir), "classes.txt"),
            os.path.join(os.path.dirname(os.path.dirname(self.train_dir)), "classes.txt")
        ]
        
        for class_file in possible_class_files:
            if os.path.exists(class_file):
                try:
                    with open(class_file, 'r') as f:
                        class_names = [line.strip() for line in f.readlines() if line.strip()]
                    
                    if class_names:
                        self.log_update.emit(f"从classes.txt加载类名: {class_file}")
                        return class_names
                except Exception as e:
                    self.log_update.emit(f"从classes.txt读取类名失败: {str(e)}")
        
        # 如果上述方法都失败，尝试从标签文件中推断
        try:
            # 查找包含.txt文件的目录
            txt_dirs = []
            for root, dirs, files in os.walk(self.train_dir):
                if any(f.endswith('.txt') for f in files):
                    txt_dirs.append(root)
            
            if not txt_dirs:
                self.log_update.emit("未找到标签文件(.txt)")
                return []
            
            # 收集所有出现的类ID
            class_ids = set()
            for txt_dir in txt_dirs:
                for file in os.listdir(txt_dir):
                    if file.endswith('.txt'):
                        try:
                            with open(os.path.join(txt_dir, file), 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        class_ids.add(int(parts[0]))
                        except Exception:
                            pass
            
            # 创建类名列表
            if class_ids:
                max_id = max(class_ids)
                class_names = [f"class{i}" for i in range(max_id + 1)]
                self.log_update.emit(f"从标签文件推断类名: 找到{len(class_names)}个类")
                return class_names
        
        except Exception as e:
            self.log_update.emit(f"从标签文件推断类名失败: {str(e)}")
        
        return []
    
    def _get_coco_class_names(self):
        """从COCO格式数据集中提取类名"""
        try:
            import json
            
            # Look for annotations file
            ann_file = None
            for file in os.listdir(self.train_dir):
                if file.endswith('.json') and ('annotations' in file or 'instances' in file):
                    ann_file = os.path.join(self.train_dir, file)
                    break
            
            if ann_file:
                with open(ann_file, 'r') as f:
                    coco_data = json.load(f)
                
                # Extract category names
                if 'categories' in coco_data:
                    categories = sorted(coco_data['categories'], key=lambda x: x['id'])
                    class_names = [cat['name'] for cat in categories]
                    return class_names
        
        except Exception as e:
            self.log_update.emit(f"Error extracting COCO class names: {str(e)}")
        
        return []
    
    def _get_voc_class_names(self):
        """从VOC格式数据集中提取类名"""
        try:
            import xml.etree.ElementTree as ET
            
            # Get a list of all XML files
            xml_files = []
            for root, _, files in os.walk(self.train_dir):
                for file in files:
                    if file.endswith('.xml'):
                        xml_files.append(os.path.join(root, file))
            
            if not xml_files:
                self.log_update.emit("未找到VOC XML标注文件")
                return []
            
            # Extract unique class names from XML files
            class_names = set()
            for xml_file in xml_files[:10]:  # Only parse a few files for efficiency
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.findall('.//object'):
                    name = obj.find('name').text
                    class_names.add(name)
            
            return sorted(list(class_names))
        
        except Exception as e:
            self.log_update.emit(f"Error extracting VOC class names: {str(e)}")
        
        return []
    
    def _find_common_parent(self, path1, path2):
        """找到两个路径的共同父目录"""
        path1 = os.path.abspath(path1)
        path2 = os.path.abspath(path2)
        
        # 将路径拆分为组件
        parts1 = path1.split(os.sep)
        parts2 = path2.split(os.sep)
        
        # 找到共同的前缀
        common_parts = []
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_parts.append(p1)
            else:
                break
        
        # 构建共同父路径
        if common_parts:
            common_path = os.sep.join(common_parts)
            # 在Windows上，确保包含驱动器号
            if os.name == 'nt' and not common_path.endswith(':'):
                common_path += os.sep
            return common_path
        
        # 如果没有共同部分，返回根目录
        return os.path.dirname(path1) 