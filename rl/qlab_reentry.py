"""DPI-aware QLAB Plane re-entry helper.

This does not restart QLAB and does not move or resize the window. It only
clicks the same on-screen UI flow documented in qlab_reenter_plane.md.
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)


EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
user32.SetProcessDPIAware.restype = wintypes.BOOL
user32.EnumWindows.argtypes = [EnumWindowsProc, wintypes.LPARAM]
user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int
user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.GetWindowRect.restype = wintypes.BOOL
user32.SetForegroundWindow.argtypes = [wintypes.HWND]
user32.SetForegroundWindow.restype = wintypes.BOOL
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wintypes.BOOL
user32.mouse_event.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
]

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def _find_qlab_window() -> wintypes.HWND:
    user32.SetProcessDPIAware()
    found = []

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        title = buffer.value.strip()
        if "Quanser Interactive Labs" in title:
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    if not found:
        raise RuntimeError("QLAB window not found for Plane re-entry.")
    return found[0]


def _window_rect(hwnd: wintypes.HWND) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise RuntimeError("Could not read QLAB window rect.")
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    return rect.left, rect.top, width, height


def _click(hwnd: wintypes.HWND, rel_x: float, rel_y: float, sleep_seconds: float) -> None:
    left, top, width, height = _window_rect(hwnd)
    x = int(left + width * rel_x)
    y = int(top + height * rel_y)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.12)
    user32.SetCursorPos(x, y)
    time.sleep(0.08)
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, None)
    time.sleep(0.06)
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, None)
    time.sleep(sleep_seconds)


def reenter_plane() -> None:
    """Exit the current QLAB scene and enter Self-Driving -> Plane."""
    hwnd = _find_qlab_window()

    # Relative positions were calibrated on the documented QLAB window layout.
    # They scale with the current window rect and never move or resize it.
    _click(hwnd, 0.934, 0.138, 2.2)  # scene exit icon
    _click(hwnd, 0.710, 0.471, 2.2)  # Self-Driving Car Studio card
    for _ in range(4):
        _click(hwnd, 0.951, 0.468, 0.9)  # right carousel arrow to Plane
    _click(hwnd, 0.778, 0.464, 3.5)  # Plane card

