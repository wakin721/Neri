#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>

#include "flutter_window.h"
#include "utils.h"

namespace {

constexpr const wchar_t kSingleInstanceMutexName[] =
    L"Local\\Neri.Desktop.SingleInstance";
constexpr const wchar_t kNeriWindowTitle[] = L"Neri";

UINT GetNeriDuplicateLaunchMessage() {
  static const UINT message =
      ::RegisterWindowMessageW(L"Neri.Desktop.DuplicateLaunch");
  return message;
}

HWND FindExistingNeriWindow() {
  return ::FindWindowW(Win32Window::GetWindowClassName(), kNeriWindowTitle);
}

HWND WaitForExistingNeriWindow() {
  constexpr DWORD kWindowWaitIntervalMs = 100;
  constexpr int kWindowWaitAttempts = 50;
  for (int attempt = 0; attempt < kWindowWaitAttempts; ++attempt) {
    if (HWND window = FindExistingNeriWindow()) {
      return window;
    }
    ::Sleep(kWindowWaitIntervalMs);
  }
  return nullptr;
}

void RestoreExistingNeriWindow(HWND window) {
  if (window == nullptr) {
    return;
  }
  if (::IsIconic(window)) {
    ::ShowWindowAsync(window, SW_RESTORE);
  } else if (!::IsWindowVisible(window)) {
    ::ShowWindowAsync(window, SW_SHOW);
  }
  ::BringWindowToTop(window);
  if (!::SetForegroundWindow(window)) {
    FLASHWINFO flash_info{};
    flash_info.cbSize = sizeof(flash_info);
    flash_info.hwnd = window;
    flash_info.dwFlags = FLASHW_TRAY | FLASHW_TIMERNOFG;
    flash_info.uCount = 3;
    flash_info.dwTimeout = 0;
    ::FlashWindowEx(&flash_info);
  }
}

int ExitWhenNeriIsAlreadyRunning(HANDLE instance_mutex) {
  HWND existing_window = WaitForExistingNeriWindow();
  RestoreExistingNeriWindow(existing_window);
  const UINT duplicate_launch_message = GetNeriDuplicateLaunchMessage();
  const bool notified_existing_window =
      existing_window != nullptr && duplicate_launch_message != 0 &&
      ::PostMessageW(existing_window, duplicate_launch_message, 0, 0);
  if (!notified_existing_window) {
    ::MessageBoxW(
        nullptr,
        L"Neri \u5df2\u7ecf\u6253\u5f00\uff0c"
        L"\u6b63\u5728\u542f\u52a8\u4e2d\uff0c\u8bf7\u7a0d\u5019\u3002",
        kNeriWindowTitle,
        MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND);
  }
  ::CloseHandle(instance_mutex);
  return EXIT_SUCCESS;
}

}  // namespace

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t* command_line, _In_ int show_command) {
  HANDLE instance_mutex =
      ::CreateMutexW(nullptr, FALSE, kSingleInstanceMutexName);
  if (instance_mutex == nullptr) {
    ::MessageBoxW(
        nullptr,
        L"\u65e0\u6cd5\u521d\u59cb\u5316 Neri "
        L"\u5355\u5b9e\u4f8b\u4fdd\u62a4\u3002",
        kNeriWindowTitle,
        MB_OK | MB_ICONERROR | MB_SETFOREGROUND);
    return EXIT_FAILURE;
  }
  const DWORD mutex_error = ::GetLastError();
  if (mutex_error == ERROR_ALREADY_EXISTS) {
    return ExitWhenNeriIsAlreadyRunning(instance_mutex);
  }

  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(1280, 720);
  if (!window.Create(kNeriWindowTitle, origin, size)) {
    ::CloseHandle(instance_mutex);
    ::CoUninitialize();
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  ::CloseHandle(instance_mutex);
  return EXIT_SUCCESS;
}
