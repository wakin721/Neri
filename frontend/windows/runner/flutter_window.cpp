#include "flutter_window.h"

#include <flutter/standard_method_codec.h>
#include <shobjidl.h>

#include <optional>
#include <string>

#include "flutter/generated_plugin_registrant.h"
#include "utils.h"

namespace {

UINT GetNeriDuplicateLaunchMessage() {
  static const UINT message =
      ::RegisterWindowMessageW(L"Neri.Desktop.DuplicateLaunch");
  return message;
}

std::wstring Utf16FromUtf8(const std::string& utf8_string) {
  if (utf8_string.empty()) {
    return std::wstring();
  }
  int target_length = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, utf8_string.data(),
      static_cast<int>(utf8_string.length()), nullptr, 0);
  if (target_length == 0) {
    return std::wstring();
  }
  std::wstring utf16_string(target_length, L'\0');
  int converted_length = ::MultiByteToWideChar(
      CP_UTF8, MB_ERR_INVALID_CHARS, utf8_string.data(),
      static_cast<int>(utf8_string.length()), utf16_string.data(),
      target_length);
  if (converted_length == 0) {
    return std::wstring();
  }
  return utf16_string;
}

std::string InitialDirectoryFromCall(
    const flutter::MethodCall<flutter::EncodableValue>& call) {
  const auto* arguments =
      std::get_if<flutter::EncodableMap>(call.arguments());
  if (!arguments) {
    return std::string();
  }
  auto entry = arguments->find(flutter::EncodableValue("initialDirectory"));
  if (entry == arguments->end()) {
    return std::string();
  }
  const auto* value = std::get_if<std::string>(&entry->second);
  return value == nullptr ? std::string() : *value;
}

template <typename T>
std::optional<T> MapValue(const flutter::EncodableMap& arguments,
                          const char* key) {
  auto entry = arguments.find(flutter::EncodableValue(key));
  if (entry == arguments.end()) {
    return std::nullopt;
  }
  const auto* value = std::get_if<T>(&entry->second);
  return value == nullptr ? std::nullopt : std::optional<T>(*value);
}

HRESULT ShowChooseDirectoryDialog(HWND owner, const std::string& initial_path,
                                  std::string* selected_path) {
  IFileDialog* dialog = nullptr;
  HRESULT hr = ::CoCreateInstance(CLSID_FileOpenDialog, nullptr,
                                  CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&dialog));
  if (FAILED(hr)) {
    return hr;
  }

  DWORD options = 0;
  if (SUCCEEDED(dialog->GetOptions(&options))) {
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM |
                       FOS_PATHMUSTEXIST);
  }
  dialog->SetTitle(L"Choose input folder");

  const std::wstring initial_path_utf16 = Utf16FromUtf8(initial_path);
  if (!initial_path_utf16.empty()) {
    IShellItem* initial_folder = nullptr;
    if (SUCCEEDED(::SHCreateItemFromParsingName(
            initial_path_utf16.c_str(), nullptr, IID_PPV_ARGS(&initial_folder)))) {
      dialog->SetDefaultFolder(initial_folder);
      initial_folder->Release();
    }
  }

  hr = dialog->Show(owner);
  if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
    dialog->Release();
    return hr;
  }
  if (SUCCEEDED(hr)) {
    IShellItem* result = nullptr;
    hr = dialog->GetResult(&result);
    if (SUCCEEDED(hr)) {
      PWSTR path = nullptr;
      hr = result->GetDisplayName(SIGDN_FILESYSPATH, &path);
      if (SUCCEEDED(hr) && path != nullptr) {
        *selected_path = Utf8FromUtf16(path);
        ::CoTaskMemFree(path);
      }
      result->Release();
    }
  }

  dialog->Release();
  return hr;
}

}  // namespace

FlutterWindow::FlutterWindow(const flutter::DartProject& project)
    : project_(project) {}

FlutterWindow::~FlutterWindow() {}

bool FlutterWindow::OnCreate() {
  if (!Win32Window::OnCreate()) {
    return false;
  }

  RECT frame = GetClientArea();

  // The size here must match the window dimensions to avoid unnecessary surface
  // creation / destruction in the startup path.
  flutter_controller_ = std::make_unique<flutter::FlutterViewController>(
      frame.right - frame.left, frame.bottom - frame.top, project_);
  // Ensure that basic setup of the controller was successful.
  if (!flutter_controller_->engine() || !flutter_controller_->view()) {
    return false;
  }
  RegisterPlugins(flutter_controller_->engine());
  dialogs_channel_ =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          flutter_controller_->engine()->messenger(), "neri/dialogs",
          &flutter::StandardMethodCodec::GetInstance());
  dialogs_channel_->SetMethodCallHandler(
      [this](const auto& call, auto result) {
        if (call.method_name() != "chooseDirectory") {
          result->NotImplemented();
          return;
        }

        std::string selected_path;
        const HRESULT hr = ShowChooseDirectoryDialog(
            GetHandle(), InitialDirectoryFromCall(call), &selected_path);
        if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED)) {
          result->Success();
          return;
        }
        if (FAILED(hr)) {
      result->Error("choose_directory_failed", "Unable to open folder picker");
          return;
        }
        result->Success(flutter::EncodableValue(selected_path));
      });
  windows_shell_channel_ =
      std::make_unique<flutter::MethodChannel<flutter::EncodableValue>>(
          flutter_controller_->engine()->messenger(), "neri/windows_shell",
          &flutter::StandardMethodCodec::GetInstance());
  windows_shell_channel_->SetMethodCallHandler(
      [this](const auto& call, auto result) {
        if (call.method_name() != "updateStatus") {
          result->NotImplemented();
          return;
        }

        const auto* arguments =
            std::get_if<flutter::EncodableMap>(call.arguments());
        if (arguments == nullptr) {
          result->Error("invalid_arguments", "Expected a status map");
          return;
        }

        const std::string tooltip =
            MapValue<std::string>(*arguments, "tooltip").value_or("Neri");
        const bool task_action_enabled =
            MapValue<bool>(*arguments, "taskActionEnabled").value_or(false);
        const bool task_is_running =
            MapValue<bool>(*arguments, "taskIsRunning").value_or(false);
        const std::string progress_state =
            MapValue<std::string>(*arguments, "progressState")
                .value_or("none");
        double progress =
            MapValue<double>(*arguments, "progress").value_or(0.0);

        TaskbarProgressState state = TaskbarProgressState::kNone;
        if (progress_state == "indeterminate") {
          state = TaskbarProgressState::kIndeterminate;
        } else if (progress_state == "normal") {
          state = TaskbarProgressState::kNormal;
        } else if (progress_state == "paused") {
          state = TaskbarProgressState::kPaused;
        } else if (progress_state == "error") {
          state = TaskbarProgressState::kError;
        }

        SetTrayStatus(Utf16FromUtf8(tooltip), task_action_enabled,
                      task_is_running);
        SetTaskbarProgress(state, progress);
        result->Success();
        dart_ready_ = true;
        if (duplicate_launch_pending_) {
          NotifyDuplicateLaunch();
        }
      });
  SetChildContent(flutter_controller_->view()->GetNativeWindow());

  flutter_controller_->engine()->SetNextFrameCallback([&]() {
    this->Show();
  });

  // Flutter can complete the first frame before the "show window" callback is
  // registered. The following call ensures a frame is pending to ensure the
  // window is shown. It is a no-op if the first frame hasn't completed yet.
  flutter_controller_->ForceRedraw();

  return true;
}

void FlutterWindow::OnDestroy() {
  if (flutter_controller_) {
    dialogs_channel_ = nullptr;
    windows_shell_channel_ = nullptr;
    dart_ready_ = false;
    duplicate_launch_pending_ = false;
    flutter_controller_ = nullptr;
  }

  Win32Window::OnDestroy();
}

void FlutterWindow::OnTrayCommand(TrayCommand command) {
  if (!windows_shell_channel_) {
    return;
  }

  const char* action = nullptr;
  switch (command) {
    case TrayCommand::kToggleProcessing:
      action = "toggleProcessing";
      break;
    case TrayCommand::kOpenSettings:
      action = "openSettings";
      break;
    case TrayCommand::kExit:
      action = "exit";
      break;
  }
  windows_shell_channel_->InvokeMethod(
      "trayAction", std::make_unique<flutter::EncodableValue>(action));
}

void FlutterWindow::NotifyDuplicateLaunch() {
  if (!dart_ready_ || !windows_shell_channel_) {
    duplicate_launch_pending_ = true;
    return;
  }
  duplicate_launch_pending_ = false;
  windows_shell_channel_->InvokeMethod(
      "duplicateLaunch", std::make_unique<flutter::EncodableValue>());
}

LRESULT
FlutterWindow::MessageHandler(HWND hwnd, UINT const message,
                              WPARAM const wparam,
                              LPARAM const lparam) noexcept {
  if (message == GetNeriDuplicateLaunchMessage()) {
    NotifyDuplicateLaunch();
    return 0;
  }

  // Give Flutter, including plugins, an opportunity to handle window messages.
  if (flutter_controller_) {
    std::optional<LRESULT> result =
        flutter_controller_->HandleTopLevelWindowProc(hwnd, message, wparam,
                                                      lparam);
    if (result) {
      return *result;
    }
  }

  switch (message) {
    case WM_FONTCHANGE:
      flutter_controller_->engine()->ReloadSystemFonts();
      break;
  }

  return Win32Window::MessageHandler(hwnd, message, wparam, lparam);
}
