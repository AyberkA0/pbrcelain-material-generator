import os
import sys

import glob
import importlib.util

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

APP_NAME = "PBRCELAIN"
VERSION = os.environ.get("PBRCELAIN_VERSION", "1.0.0")
IS_MAC = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

hiddenimports = []
for package in (
    "transformers.models.depth_anything",
    "transformers.models.dpt",
    "transformers.models.clip",
    "diffusers.pipelines.marigold",
    "diffusers.schedulers",
    "diffusers.models.unets",
    "diffusers.models.autoencoders",
):
    hiddenimports += collect_submodules(package)
hiddenimports += ["OpenGL.platform.darwin" if IS_MAC else "OpenGL.platform.win32", "PyQt6.QtSvg"]

datas = [("ui/assets", "ui/assets")]
datas += collect_data_files("transformers", include_py_files=False)
datas += collect_data_files("diffusers", include_py_files=False)
for dist in (
    "torch", "torchvision", "transformers", "diffusers", "accelerate", "huggingface_hub",
    "tokenizers", "safetensors", "tqdm", "regex", "requests", "packaging", "filelock",
    "numpy", "pyyaml", "pillow", "opencv-python-headless", "PyOpenGL",
):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

torchvision_dir = os.path.dirname(importlib.util.find_spec("torchvision").origin)
binaries = [
    (path, "torchvision")
    for pattern in ("*.so", "*.pyd", "*.dll", "*.dylib")
    for path in glob.glob(os.path.join(torchvision_dir, pattern))
]

a = Analysis(
    ["main.py"],
    pathex=[os.path.abspath(".")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "tensorflow", "jax", "flax"],
    noarchive=False,
)
pyz = PYZ(a.pure)

icon = "build/PBRCELAIN.icns" if IS_MAC else "ui/assets/app_icon.ico"

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    console=False,
    icon=icon,
    upx=False,
    target_arch="arm64" if IS_MAC else None,
)

coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME, upx=False)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier="com.pbrcelain.PBRCELAIN",
        version=VERSION,
        info_plist={
            "CFBundleName": APP_NAME,
            "CFBundleDisplayName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "LSMinimumSystemVersion": "12.0",
            "LSApplicationCategoryType": "public.app-category.graphics-design",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            "CFBundleDocumentTypes": [{
                "CFBundleTypeName": "PBRCELAIN Project",
                "CFBundleTypeRole": "Editor",
                "CFBundleTypeIconFile": "PBRCELAIN.icns",
                "LSHandlerRank": "Owner",
                "LSItemContentTypes": ["com.pbrcelain.project"],
            }],
            "UTExportedTypeDeclarations": [{
                "UTTypeIdentifier": "com.pbrcelain.project",
                "UTTypeDescription": "PBRCELAIN Project",
                "UTTypeConformsTo": ["public.data", "public.composite-content"],
                "UTTypeIconFile": "PBRCELAIN.icns",
                "UTTypeTagSpecification": {"public.filename-extension": ["pcln"]},
            }],
        },
    )
