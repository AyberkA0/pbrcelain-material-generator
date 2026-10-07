"""Associate .pcln project files with PBRCELAIN on Windows (current user only)
and add a PBRCELAIN shortcut to the Start menu.

Double-clicking a .pcln file then launches this project's venv interpreter
(pythonw.exe, no console window) with main.py and the file path.

    venv\\Scripts\\python register_file_association.py              # register
    venv\\Scripts\\python register_file_association.py --unregister # remove

Writes only under HKEY_CURRENT_USER\\Software\\Classes, so no admin rights
are needed and other users are unaffected. Re-run after moving the project
folder, since the registered command stores absolute paths.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys

if sys.platform != "win32":
    sys.exit("File association registration is only supported on Windows.")

import winreg

EXTENSION = ".pcln"
PROG_ID = "PBRCELAIN.Project"
DESCRIPTION = "PBRCELAIN Project"
CLASSES_ROOT = r"Software\Classes"

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(PROJECT_DIR, "main.py")
ICON_FILE = os.path.join(PROJECT_DIR, "ui", "assets", "app_icon.ico")
START_MENU_SHORTCUT = os.path.join(
    os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs", "PBRCELAIN.lnk"
)


def _find_pythonw() -> str:
    candidates = [
        os.path.join(PROJECT_DIR, "venv", "Scripts", "pythonw.exe"),
        os.path.join(os.path.dirname(sys.executable), "pythonw.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    sys.exit("pythonw.exe not found - create the venv first (see README).")


def _set_value(subkey: str, value: str, name: str = "") -> None:
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, rf"{CLASSES_ROOT}\{subkey}", 0, winreg.KEY_WRITE) as key:
        winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)


def _delete_tree(subkey: str) -> None:
    full = rf"{CLASSES_ROOT}\{subkey}"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, full, 0, winreg.KEY_ALL_ACCESS) as key:
            while True:
                try:
                    child = winreg.EnumKey(key, 0)
                except OSError:
                    break
                _delete_tree(rf"{subkey}\{child}")
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, full)
    except FileNotFoundError:
        pass


def _notify_shell() -> None:
    SHCNE_ASSOCCHANGED = 0x08000000
    SHCNF_IDLIST = 0x0000
    ctypes.windll.shell32.SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, None, None)


def _create_start_menu_shortcut(pythonw: str) -> None:
    script = (
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut($env:PBR_LNK);"
        "$s.TargetPath = $env:PBR_TARGET; $s.Arguments = $env:PBR_ARGS;"
        "$s.WorkingDirectory = $env:PBR_DIR; $s.IconLocation = $env:PBR_ICON;"
        "$s.Description = 'PBRCELAIN material generator'; $s.Save()"
    )
    env = dict(
        os.environ,
        PBR_LNK=START_MENU_SHORTCUT, PBR_TARGET=pythonw, PBR_ARGS=f'"{MAIN_SCRIPT}"',
        PBR_DIR=PROJECT_DIR, PBR_ICON=f"{ICON_FILE},0",
    )
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], env=env, check=True)


def register() -> None:
    pythonw = _find_pythonw()
    command = f'"{pythonw}" "{MAIN_SCRIPT}" "%1"'

    _set_value(EXTENSION, PROG_ID)
    _set_value(rf"{EXTENSION}\OpenWithProgids", "", name=PROG_ID)
    _set_value(PROG_ID, DESCRIPTION)
    _set_value(rf"{PROG_ID}\DefaultIcon", f'"{ICON_FILE}",0')
    _set_value(rf"{PROG_ID}\shell", "open")
    _set_value(rf"{PROG_ID}\shell\open", "Open with PBRCELAIN")
    _set_value(rf"{PROG_ID}\shell\open\command", command)
    _notify_shell()
    print(f"Registered {EXTENSION} -> {command}")
    try:
        _create_start_menu_shortcut(pythonw)
        print(f"Added Start menu shortcut: {START_MENU_SHORTCUT}")
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Could not create the Start menu shortcut: {exc}")


def unregister() -> None:
    _delete_tree(PROG_ID)
    _delete_tree(EXTENSION)
    _notify_shell()
    print(f"Removed {EXTENSION} association.")
    if os.path.isfile(START_MENU_SHORTCUT):
        os.remove(START_MENU_SHORTCUT)
        print("Removed Start menu shortcut.")


if __name__ == "__main__":
    if "--unregister" in sys.argv[1:]:
        unregister()
    else:
        register()
