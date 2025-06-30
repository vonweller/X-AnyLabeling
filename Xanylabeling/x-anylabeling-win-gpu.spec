# -*- mode: python -*-
# vim: ft=python

import sys
import os
from pathlib import Path

sys.setrecursionlimit(5000)  # required on Windows

# Get the site-packages directory
import site
site_packages = site.getsitepackages()[0]

# Find onnxruntime DLLs
onnxruntime_path = os.path.join(site_packages, 'onnxruntime', 'capi')
cuda_dll = os.path.join(onnxruntime_path, 'onnxruntime_providers_cuda.dll')
shared_dll = os.path.join(onnxruntime_path, 'onnxruntime_providers_shared.dll')

# Verify DLLs exist
if not os.path.exists(cuda_dll):
    print(f"Warning: CUDA DLL not found at {cuda_dll}")
    cuda_dll = None
if not os.path.exists(shared_dll):
    print(f"Warning: Shared DLL not found at {shared_dll}")
    shared_dll = None

# Prepare datas list
datas = [
    ('anylabeling/configs/auto_labeling/*.yaml', 'anylabeling/configs/auto_labeling'),
    ('anylabeling/configs/*.yaml', 'anylabeling/configs'),
    ('anylabeling/views/labeling/widgets/auto_labeling/auto_labeling.ui', 'anylabeling/views/labeling/widgets/auto_labeling'),
    ('anylabeling/services/auto_labeling/configs/bert/*', 'anylabeling/services/auto_labeling/configs/bert'),
    ('anylabeling/services/auto_labeling/configs/clip/*', 'anylabeling/services/auto_labeling/configs/clip'),
    ('anylabeling/services/auto_labeling/configs/ppocr/*', 'anylabeling/services/auto_labeling/configs/ppocr'),
    ('anylabeling/services/auto_labeling/configs/ram/*', 'anylabeling/services/auto_labeling/configs/ram'),
    ('YOLO_UI', 'YOLO_UI'),
]

# Add DLLs if they exist
if cuda_dll:
    datas.append((cuda_dll, 'onnxruntime/capi'))
if shared_dll:
    datas.append((shared_dll, 'onnxruntime/capi'))

a = Analysis(
    ['anylabeling/app.py'],
    pathex=['anylabeling'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'torch',
        'torchvision',
        'onnxruntime',
        'numpy',
        'cv2',
        'PyQt5',
        'yaml'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
)
pyz = PYZ(a.pure, a.zipped_data)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='X-AnyLabeling-GPU',
    debug=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,  # Changed to True temporarily for debugging
    icon='anylabeling/resources/images/icon.icns',
)
app = BUNDLE(
    exe,
    name='X-AnyLabeling.app',
    icon='anylabeling/resources/images/icon.icns',
    bundle_identifier=None,
    info_plist={'NSHighResolutionCapable': 'True'},
)