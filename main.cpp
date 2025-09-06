// main.cpp
// Minimal CEF + DirectX 12 sample with D3D11-on-12 interop.
// - Accelerated paint (GPU) path using shared D3D11 textures when available.
// - Fallback to software paint upload when GPU path is not available.

#define NOMINMAX
#include <windows.h>
#include <wrl.h>
#include <d3d12.h>
#include "d3dx12.h"  // <-- Add this
#include <dxgi1_6.h>
#include <d3d11on12.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>

#include "include/cef_app.h"
#include "include/cef_browser.h"
#include "include/cef_client.h"
#include "include/cef_render_handler.h"
#include "include/cef_command_line.h"
#include "include/wrapper/cef_helpers.h"
#include <include/cef_version_info.h>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

#include "DX12Context.h"

// ---------------------------- Window ---------------------------------

static const wchar_t* kClassName = L"CEF_DX12_OSR_Window";

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ----------------------------- CEF -----------------------------------

// Global access to DX for render handler (simple for demo)
static DX12Context* gDX = nullptr;
static HWND gHwnd = nullptr;





class RenderHandler : public CefRenderHandler {
public:
    RenderHandler(int w, int h) : w_(w), h_(h) {}

    void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override
    {
        rect = CefRect(0, 0, w_, h_);
    }

    void OnPaint(CefRefPtr<CefBrowser>, PaintElementType type, const CefRenderHandler::RectList& dirtyRects, const void* buffer, int width, int height) override
    {
        CEF_REQUIRE_UI_THREAD();
        if (type != PET_VIEW) return;
        // Software fallback path: buffer is BGRA
        if ((UINT)width == gDX->browserW && (UINT)height == gDX->browserH)
        {
            gDX->UploadSoftwareBitmap(buffer, width * 4);
        }
    }

    void OnAcceleratedPaint(CefRefPtr<CefBrowser> browser, PaintElementType type, const RectList& dirtyRects, const CefAcceleratedPaintInfo& info) override
    {
        CEF_REQUIRE_UI_THREAD();
        if (type != PET_VIEW || !info.shared_texture_handle)
        {
            return;
        }
        // Open the shared handle as an ID3D11Texture2D on our D3D11 device
        HANDLE h = reinterpret_cast<HANDLE>(info.shared_texture_handle);
        ComPtr<ID3D11Texture2D> src;
        HRESULT hr = gDX->d3d11Device->OpenSharedResource(h, __uuidof(ID3D11Texture2D), (void**)src.GetAddressOf());
        if (SUCCEEDED(hr) && src)
        {
            gDX->CopyFromD3D11Shared(src.Get());
        }
    }

    bool GetScreenInfo(CefRefPtr<CefBrowser>, CefScreenInfo& info) override
    {
        info.device_scale_factor = 1.0f;
        info.depth = 24; info.depth_per_component = 8; info.is_monochrome = false;
        info.rect = CefRect(0, 0, w_, h_);
        info.available_rect = info.rect;
        return true;
    }

    void UpdateSize(int w, int h) { w_ = w; h_ = h; }

    IMPLEMENT_REFCOUNTING(RenderHandler);
private:
    int w_, h_;
};





class CefOSRClient : public CefClient, public CefLifeSpanHandler
{
public:
    explicit CefOSRClient(CefRefPtr<RenderHandler> rh) : render_handler_(rh) {}
    CefRefPtr<CefRenderHandler> GetRenderHandler() override { return render_handler_; }
    CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override { return this; }

    void OnAfterCreated(CefRefPtr<CefBrowser> browser) override {
        browser_ = browser;
    }

    bool DoClose(CefRefPtr<CefBrowser>) override { return false; }

    void OnBeforeClose(CefRefPtr<CefBrowser>) override { browser_ = nullptr; }

    CefRefPtr<CefBrowser> browser() const { return browser_; }

    IMPLEMENT_REFCOUNTING(CefOSRClient);
private:
    CefRefPtr<RenderHandler> render_handler_;
    CefRefPtr<CefBrowser> browser_;
};






class MyApp : public CefApp, public CefBrowserProcessHandler {
public:
    CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override { return this; }
    void OnBeforeCommandLineProcessing(const CefString&, CefRefPtr<CefCommandLine> cmd) override {
        // Enable GPU compositing for OSR
        cmd->AppendSwitch("enable-gpu");
        cmd->AppendSwitch("enable-begin-frame-scheduling");
        cmd->AppendSwitch("enable-native-gpu-memory-buffers");
    }
    IMPLEMENT_REFCOUNTING(MyApp);
};



#pragma region ClientAppOther

// Client app implementation for other process types.
class ClientAppOther : public CefApp {
public:
    ClientAppOther();

private:
    IMPLEMENT_REFCOUNTING(ClientAppOther);
    DISALLOW_COPY_AND_ASSIGN(ClientAppOther);
};


ClientAppOther::ClientAppOther() = default;

#pragma endregion  // ClientAppOther


// -------------------------- App globals ------------------------------

static std::unique_ptr<DX12Context> gDxCtx;
static CefRefPtr<CefOSRClient> gClient;
static CefRefPtr<RenderHandler> gRenderHandler;

// ------------------------ WinMain and loop ---------------------------

HANDLE hLogFile;
void AppendToLog(const char* textToAppend)
{
    DWORD bytesWritten = 0;

    SYSTEMTIME local;
    GetLocalTime(&local);  // Local time

    char buf[1024];
    sprintf_s(buf, "[%02d:%02d:%02d.%03d] - %s\r\n", local.wHour, local.wMinute, local.wSecond, local.wMilliseconds, textToAppend);

    BOOL ok = WriteFile(hLogFile, buf, static_cast<DWORD>(strlen(buf)), &bytesWritten, nullptr);
    FlushFileBuffers(hLogFile);
}


int APIENTRY wWinMain(HINSTANCE hInst, HINSTANCE, LPWSTR, int)
{
    // log file to log the processes activity.
    // Created in the ProjectDir foilder
    const WCHAR* logFileName = L"myceflog.txt";
    hLogFile = CreateFile(logFileName, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
    auto er = GetLastError();

    // Parse command-line arguments.
    CefRefPtr<CefCommandLine> command_line = CefCommandLine::CreateCommandLine();
    command_line->InitFromString(::GetCommandLineW());
    int processType = -1;

    if (!command_line->HasSwitch("type"))
    {
        processType = 0;
    }
    else
    {
        const CefString process_type = command_line->GetSwitchValue("type");
        if (process_type == "renderer")
        {
            processType = 1;
        }
        else
        {
            processType = 3;
        }
    }

    // store debug info about the process started
    char buf[1024];
    DWORD pid = GetCurrentProcessId();
    DWORD tid = GetCurrentThreadId();
    sprintf_s(buf, "NewProcess (%i), PID=%u, TID=%u", processType, pid, tid);
    AppendToLog(buf);

    CefRefPtr<CefApp> app;
    if (processType == 0)  // client::ClientApp::BrowserProcess
    {
    //    app = new client::ClientAppBrowser();
    }
    else if (processType == 1) //client::ClientApp::RendererProcess
    {
    //    app = new client::ClientAppRenderer();
    }
    else if (processType == 3) //client::ClientApp::OtherProcess
    {
        app = new ClientAppOther();
    }


    CefMainArgs main_args(hInst);

    void* sandbox_info = nullptr;

#if defined(CEF_USE_SANDBOX)
    // Manage the life span of the sandbox information object. This is necessary
    // for sandbox support on Windows. See cef_sandbox_win.h for complete details.
    CefScopedSandboxInfo scoped_sandbox;
    sandbox_info = scoped_sandbox.sandbox_info();
#endif

    cef_version_info_t version_info = {};
    CEF_POPULATE_VERSION_INFO(&version_info);

    if (app)
    {
        // Execute the secondary process, if any.
        int exit_code = CefExecuteProcess(main_args, app, sandbox_info);
        if (exit_code >= 0) {
            return exit_code;
        }
    }


    // Window
    WNDCLASS wc{};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = kClassName;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    RegisterClass(&wc);
    RECT rc{ 0,0,DX12Context::gClientWidth,DX12Context::gClientHeight };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);
    gHwnd = CreateWindowEx(0, kClassName, L"CEF + DX12 (OSR)",
        WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left, rc.bottom - rc.top, nullptr, nullptr, hInst, nullptr);

    WCHAR msgString[1024];
    swprintf_s(msgString, sizeof(msgString) / sizeof(msgString[0]), L"winmain proc(%i)", processType);
    SetWindowText(gHwnd, msgString);

    // DX12 init
    gDxCtx = std::make_unique<DX12Context>();
    gDX = gDxCtx.get();
    gDxCtx->Init(gHwnd, DX12Context::gClientWidth, DX12Context::gClientHeight);

    //// CEF init
    //CefMainArgs main_args(GetModuleHandle(nullptr));
    //CefRefPtr<MyApp> app(new MyApp());

    CefSettings settings;
    settings.no_sandbox = true;
    settings.windowless_rendering_enabled = true;
    settings.external_message_pump = true;
    settings.multi_threaded_message_loop = false; // We'll pump it
    CefInitialize(main_args, settings, app.get(), nullptr);

    // Create OSR browser
//CEF    gRenderHandler = new RenderHandler((int)gDxCtx->browserW, (int)gDxCtx->browserH);
//CEF    gClient = new CefOSRClient(gRenderHandler);

    CefWindowInfo wi;
    wi.SetAsWindowless(gHwnd);
    CefBrowserSettings bs;
    bs.windowless_frame_rate = 60;

//CEF    CefBrowserHost::CreateBrowser(wi, gClient.get(), "https://example.org", bs, nullptr, nullptr);

    // Main loop
    MSG msg{};
    bool running = true;

    while (running)
    {
        // Process all pending messages
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
                break;
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        // Pump CEF work
//CEF        CefDoMessageLoopWork();

        // mouse position
        POINT pt;
        float mouseX, mouseY;
        if (GetCursorPos(&pt))
        {
            // Convert to client coordinates if needed
            ScreenToClient(gHwnd, &pt);
            // Now pt.x, pt.y are relative to your window
            // normalize -1..+1
            mouseX = (2.0f * float(pt.x)) / float(DX12Context::gClientWidth) - 1.0f;
            mouseY = (2.0f * float(pt.y)) / float(DX12Context::gClientHeight) - 1.0f;
        }

        // Render frame
        std::chrono::steady_clock::time_point last = std::chrono::high_resolution_clock::now();
        gDxCtx->Begin(last, mouseX, mouseY);
        gDxCtx->End();
    }

    gDxCtx->Finalize();

    CloseHandle(hLogFile);

    // Shutdown CEF
    if (gClient && gClient->browser())
    {
//CEF        gClient->browser()->GetHost()->CloseBrowser(true);
    }
    CefShutdown();

    return 0;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) 
    {
    case WM_SIZE:
        if (gDxCtx) {
            DX12Context::gClientWidth = LOWORD(lParam);
            DX12Context::gClientHeight = HIWORD(lParam);
            gDxCtx->Resize(DX12Context::gClientWidth, DX12Context::gClientHeight);
        }
        return 0;
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MOUSEMOVE:
    case WM_MOUSEWHEEL:
    case WM_KEYDOWN:
    case WM_KEYUP:
        //if (gClient && gClient->browser()) {
        //    CefMouseEvent me;
        //    POINT p; GetCursorPos(&p); ScreenToClient(hWnd, &p);
        //    me.x = p.x; me.y = p.y;
        //    bool mouseUp = (msg == WM_LBUTTONUP);
        //    bool mouseDown = (msg == WM_LBUTTONDOWN);
        //    if (mouseDown || mouseUp) {
        //        gClient->browser()->GetHost()->SendMouseClickEvent(
        //            me, MBT_LEFT, mouseUp, 1);
        //    }
        //    else if (msg == WM_MOUSEMOVE) {
        //        gClient->browser()->GetHost()->SendMouseMoveEvent(me, false);
        //    }
        //}
        break;
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProc(hWnd, msg, wParam, lParam);
}
