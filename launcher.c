#include <windows.h>
#include <Python.h>  // 引入 Python C API

#pragma comment(linker, "/SUBSYSTEM:windows /ENTRY:mainCRTStartup")

// 线程函数：用于运行 Python 后端
DWORD WINAPI BackendThread(LPVOID lpParam) {

    // ================= 1. 配置 Python 环境 =================
    // 告诉 Python 它的运行环境（标准库等）在哪个相对目录
    // 假设你把官方的 Embeddable Python 放在了 toolkit 目录下
    Py_SetPythonHome(L"toolkit");

    // 初始化 Python 解释器
    Py_Initialize();

    if (!Py_IsInitialized()) {
        MessageBoxA(NULL, "Python 引擎初始化失败！", "错误", MB_ICONERROR);
        return 1;
    }

    // ================= 2. 配置 Python 路径 =================
    // 将当前工作目录添加到 sys.path 中，否则它找不到 system.backend.main
    PyRun_SimpleString(
        "import sys\n"
        "import os\n"
        "sys.path.append(os.getcwd())\n"
    );

    // ================= 3. 启动后端 (会在此处阻塞) =================
    // 这段等效于你在命令行敲的 python -m uvicorn ...
    PyRun_SimpleString(
        "import uvicorn\n"
        "uvicorn.run('system.backend.main:app', host='127.0.0.1', port=721, app_dir='.')\n"
    );

    // 当 uvicorn 停止时，才会执行到这里，清理 Python 环境
    Py_Finalize();
    return 0;
}


int main() {
    SetConsoleOutputCP(65001);

    // 1. 创建独立线程，启动 Python 后端
    // 这样不会阻塞主进程去启动前端 UI
    HANDLE hBackendThread = CreateThread(NULL, 0, BackendThread, NULL, 0, NULL);
    if (hBackendThread == NULL) {
        MessageBoxA(NULL, "后端线程创建失败！", "错误", MB_ICONERROR);
        return 1;
    }

    // 2. 等待后端服务就绪 (给 uvicorn 3秒钟的启动时间)
    Sleep(3000);

    // 3. 启动前端进程
    char frontendCmd[] = "app\\Neri_Frontend.exe";
    STARTUPINFO siFrontend = { sizeof(siFrontend) };
    PROCESS_INFORMATION piFrontend = { 0 };

    if (!CreateProcessA(NULL, frontendCmd, NULL, NULL, FALSE, 0, NULL, NULL, &siFrontend, &piFrontend)) {
        // 前端启动失败，直接干掉整个进程（自然也会带走后端的 Python 线程）
        ExitProcess(1);
    }

    // 4. 挂起主线程，一直等待前端 UI 被用户关闭
    WaitForSingleObject(piFrontend.hProcess, INFINITE);

    // 5. 前端已关闭，清理句柄
    CloseHandle(piFrontend.hProcess);
    CloseHandle(piFrontend.hThread);

    // 6. 完美谢幕
    // 直接调用 ExitProcess(0) 结束当前的 Launcher 进程。
    // 这会自动且瞬间杀死正在后台运行 Uvicorn 的 BackendThread 线程，
    // 不需要像写 Python 脚本那样费劲地去发信号优雅停止 Uvicorn，非常干净利落。
    ExitProcess(0);

    return 0;
}