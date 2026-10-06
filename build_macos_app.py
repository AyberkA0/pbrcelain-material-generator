"""Build a PBRCELAIN.app launcher on macOS and associate .pcln files with it.

The bundle is a thin launcher: it runs this project's main.py with the venv
interpreter (or the one running this script), so the source stays where it
is and edits take effect on the next launch. It gives the app its own Dock
name and icon, a Launchpad/Spotlight entry, and makes double-clicking a
.pcln file in Finder open it in PBRCELAIN.

    venv/bin/python build_macos_app.py               # install to ~/Applications
    venv/bin/python build_macos_app.py --dest DIR    # install somewhere else
    venv/bin/python build_macos_app.py --uninstall   # remove it again

Re-run after moving the project folder or recreating the venv, since the
launcher stores absolute paths. Output is logged to
~/Library/Logs/PBRCELAIN/pbrcelain.log.
"""
from __future__ import annotations

import json
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
import tempfile

if sys.platform != "darwin":
    sys.exit("This script builds a macOS .app bundle and only runs on macOS.")

APP_NAME = "PBRCELAIN"
BUNDLE_ID = "com.pbrcelain.PBRCELAIN"
PROJECT_UTI = "com.pbrcelain.project"
EXTENSION = "pcln"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_DIR, "main.py")
ICON_PNG = os.path.join(PROJECT_DIR, "ui", "assets", "mac", "app_icon.png")
LSREGISTER = (
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)


def _find_python() -> str:
    candidates = [
        os.path.join(PROJECT_DIR, "venv", "bin", "python"),
        os.path.join(PROJECT_DIR, ".venv", "bin", "python"),
        sys.executable,
    ]
    for path in candidates:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    sys.exit("No Python interpreter found - create the venv first (see README).")


def _version() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", PROJECT_DIR, "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"1.0.{out}"
    except (OSError, subprocess.CalledProcessError):
        return "1.0"


def _info_plist() -> dict:
    version = _version()
    return {
        "CFBundleName": APP_NAME,
        "CFBundleDisplayName": APP_NAME,
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleExecutable": APP_NAME,
        "CFBundleIconFile": APP_NAME,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "CFBundleInfoDictionaryVersion": "6.0",
        "LSMinimumSystemVersion": "12.0",
        "LSApplicationCategoryType": "public.app-category.graphics-design",
        "NSHighResolutionCapable": True,
        "NSSupportsAutomaticGraphicsSwitching": True,
        "NSRequiresAquaSystemAppearance": False,
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "PBRCELAIN Project",
                "CFBundleTypeRole": "Editor",
                "CFBundleTypeIconFile": APP_NAME,
                "LSHandlerRank": "Owner",
                "LSItemContentTypes": [PROJECT_UTI],
            }
        ],
        "UTExportedTypeDeclarations": [
            {
                "UTTypeIdentifier": PROJECT_UTI,
                "UTTypeDescription": "PBRCELAIN Project",
                "UTTypeConformsTo": ["public.data", "public.composite-content"],
                "UTTypeIconFile": APP_NAME,
                "UTTypeTagSpecification": {"public.filename-extension": [EXTENSION]},
            }
        ],
    }


def _interpreter_info(python: str) -> dict:
    """Ask `python` where its framework app binary and site-packages live."""
    code = (
        "import json, os, site, sys\n"
        "app = os.path.join(sys.base_prefix, 'Resources', 'Python.app', 'Contents', 'MacOS', 'Python')\n"
        "print(json.dumps({'framework_app': app if os.path.isfile(app) else None,"
        " 'in_venv': sys.prefix != sys.base_prefix,"
        " 'site_packages': site.getsitepackages()}))"
    )
    out = subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _launcher_script(interpreter: str, site_packages: list[str]) -> str:
    # `exec` keeps the launcher's process. Because the interpreter binary
    # itself lives inside this bundle, macOS identifies the running app as
    # PBRCELAIN.app (Dock name/icon), and Finder's "open document" events
    # reach it. A venv's site-packages are added explicitly, since the copied
    # binary only knows its base installation.
    if site_packages:
        boot = (
            "import runpy, site, sys\n"
            f"for p in {site_packages!r}: site.addsitedir(p)\n"
            f"sys.argv = [{MAIN_SCRIPT!r}] + sys.argv[1:]\n"
            f"runpy.run_path({MAIN_SCRIPT!r}, run_name='__main__')\n"
        )
        run = f'exec {shlex.quote(interpreter)} -c {shlex.quote(boot)} "$@"'
    else:
        run = f'exec {shlex.quote(interpreter)} {shlex.quote(MAIN_SCRIPT)} "$@"'
    return f"""#!/bin/bash
LOG_DIR="$HOME/Library/Logs/{APP_NAME}"
mkdir -p "$LOG_DIR"
cd {shlex.quote(PROJECT_DIR)} || exit 1
export PYTORCH_ENABLE_MPS_FALLBACK=1
{run} >>"$LOG_DIR/pbrcelain.log" 2>&1
"""


def _build_icns(dest_icns: str) -> None:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        iconset = os.path.join(tmp, f"{APP_NAME}.iconset")
        os.makedirs(iconset)
        with Image.open(ICON_PNG) as src:
            src = src.convert("RGBA")
            for size in (16, 32, 128, 256, 512):
                src.resize((size, size), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}.png"))
                src.resize((size * 2, size * 2), Image.LANCZOS).save(os.path.join(iconset, f"icon_{size}x{size}@2x.png"))
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", dest_icns], check=True)


def install(dest_dir: str) -> None:
    python = _find_python()
    app = os.path.join(dest_dir, f"{APP_NAME}.app")
    if os.path.exists(app):
        shutil.rmtree(app)

    macos_dir = os.path.join(app, "Contents", "MacOS")
    res_dir = os.path.join(app, "Contents", "Resources")
    os.makedirs(macos_dir)
    os.makedirs(res_dir)

    with open(os.path.join(app, "Contents", "Info.plist"), "wb") as f:
        plistlib.dump(_info_plist(), f)
    with open(os.path.join(app, "Contents", "PkgInfo"), "w") as f:
        f.write("APPL????")

    info = _interpreter_info(python)
    if info["framework_app"]:
        interpreter = os.path.join(macos_dir, "pbrcelain-python")
        shutil.copy2(info["framework_app"], interpreter)
        subprocess.run(
            ["codesign", "--force", "--sign", "-", "--identifier", f"{BUNDLE_ID}.python", interpreter],
            check=True, capture_output=True,
        )
    else:
        # Non-framework Python (e.g. conda): run it in place. Works the same,
        # but the Dock may show it as "python" instead of PBRCELAIN.
        interpreter = python
    site_packages = info["site_packages"] if info["in_venv"] and info["framework_app"] else []

    launcher = os.path.join(macos_dir, APP_NAME)
    with open(launcher, "w") as f:
        f.write(_launcher_script(interpreter, site_packages))
    os.chmod(launcher, 0o755)

    _build_icns(os.path.join(res_dir, f"{APP_NAME}.icns"))

    subprocess.run([LSREGISTER, "-f", app], check=False)
    subprocess.run(["touch", app], check=False)
    print(f"Installed {app}")
    print(f"  python : {python}")
    print(f"  project: {PROJECT_DIR}")
    print(f".{EXTENSION} files now open in {APP_NAME} (Finder may need a moment to refresh icons).")


def uninstall(dest_dir: str) -> None:
    app = os.path.join(dest_dir, f"{APP_NAME}.app")
    if not os.path.exists(app):
        print(f"Nothing to remove at {app}")
        return
    subprocess.run([LSREGISTER, "-u", app], check=False)
    shutil.rmtree(app)
    print(f"Removed {app}")


if __name__ == "__main__":
    args = sys.argv[1:]
    dest = os.path.expanduser("~/Applications")
    if "--dest" in args:
        i = args.index("--dest")
        if i + 1 >= len(args):
            sys.exit("--dest needs a directory")
        dest = os.path.abspath(os.path.expanduser(args[i + 1]))
    os.makedirs(dest, exist_ok=True)
    if "--uninstall" in args:
        uninstall(dest)
    else:
        install(dest)
