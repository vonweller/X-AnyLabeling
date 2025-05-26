import subprocess
import re

def get_local_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version'], encoding='utf-8')
        match = re.search(r'release ([\\d.]+)', output)
        if match:
            return match.group(1)
    except Exception as e:
        return None

version = get_local_cuda_version()
print("本地 CUDA 版本:", version if version else "未检测到 CUDA")