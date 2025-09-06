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
#include <optional>

#include "include/cef_app.h"
#include "include/cef_browser.h"
#include "include/cef_client.h"
#include "include/cef_render_handler.h"
#include "include/cef_command_line.h"
#include "include/wrapper/cef_helpers.h"
#include <include/cef_version_info.h>
#include <include/cef_crash_util.h>

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



// Client app implementation for the browser process.
class ClientAppBrowser : public CefApp, public CefBrowserProcessHandler {
public:
    // Interface for browser delegates. All Delegates must be returned via
    // CreateDelegates. Do not perform work in the Delegate
    // constructor. See CefBrowserProcessHandler for documentation.
    //class Delegate : public virtual CefBaseRefCounted {
    //public:
    //    virtual void OnBeforeCommandLineProcessing(
    //        CefRefPtr<ClientAppBrowser> app,
    //        CefRefPtr<CefCommandLine> command_line) {}

    //    virtual void OnRegisterCustomPreferences(
    //        CefRefPtr<ClientAppBrowser> app,
    //        cef_preferences_type_t type,
    //        CefRawPtr<CefPreferenceRegistrar> registrar) {}

    //    virtual void OnContextInitialized(CefRefPtr<ClientAppBrowser> app) {}

    //    virtual bool OnAlreadyRunningAppRelaunch(
    //        CefRefPtr<ClientAppBrowser> app,
    //        CefRefPtr<CefCommandLine> command_line,
    //        const CefString& current_directory) {
    //        return false;
    //    }

    //    virtual CefRefPtr<CefClient> GetDefaultClient(
    //        CefRefPtr<ClientAppBrowser> app) {
    //        return nullptr;
    //    }
    //};

//    typedef std::set<CefRefPtr<Delegate>> DelegateSet;

    ClientAppBrowser();

    // Called to populate |settings| based on |command_line| and other global
    // state.
    static void PopulateSettings(CefRefPtr<CefCommandLine> command_line,
        CefSettings& settings);

private:
    //// Registers cookieable schemes. Implemented by cefclient in
    //// client_app_delegates_browser.cc
    //static void RegisterCookieableSchemes(
    //    std::vector<std::string>& cookieable_schemes);

    //// Creates all of the Delegate objects. Implemented by cefclient in
    //// client_app_delegates_browser.cc
    //static void CreateDelegates(DelegateSet& delegates);

    // CefApp methods.
    void OnBeforeCommandLineProcessing(const CefString& process_type, CefRefPtr<CefCommandLine> command_line) override;

    CefRefPtr<CefBrowserProcessHandler> GetBrowserProcessHandler() override
    {
        return this;
    }

    // CefBrowserProcessHandler methods.
    void OnRegisterCustomPreferences(
        cef_preferences_type_t type,
        CefRawPtr<CefPreferenceRegistrar> registrar) override;
    void OnContextInitialized() override;
    bool OnAlreadyRunningAppRelaunch(CefRefPtr<CefCommandLine> command_line,
        const CefString& current_directory) override;
    void OnScheduleMessagePumpWork(int64_t delay) override;
//INVESTIG!!!    CefRefPtr<CefClient> GetDefaultClient() override;

    //// Set of supported Delegates.
    //DelegateSet delegates_;

    IMPLEMENT_REFCOUNTING(ClientAppBrowser);
    DISALLOW_COPY_AND_ASSIGN(ClientAppBrowser);
};



// Default client handler for unmanaged browser windows. Used with Chrome
// style only.
class DefaultClientHandler : public CefClient,
    public CefFocusHandler,
    public CefLifeSpanHandler,
    public CefLoadHandler,
    public CefRequestHandler,
    public CefResourceRequestHandler {
public:
    // If |use_alloy_style| is nullopt the global default will be used.
    explicit DefaultClientHandler(
        std::optional<bool> use_alloy_style = std::nullopt);

    // Returns the DefaultClientHandler for |client|, or nullptr if |client| is
    // not a DefaultClientHandler.
    static CefRefPtr<DefaultClientHandler> GetForClient(
        CefRefPtr<CefClient> client);

protected:
    bool OnBeforePopup(
        CefRefPtr<CefBrowser> browser,
        CefRefPtr<CefFrame> frame,
        int popup_id,
        const CefString& target_url,
        const CefString& target_frame_name,
        CefLifeSpanHandler::WindowOpenDisposition target_disposition,
        bool user_gesture,
        const CefPopupFeatures& popupFeatures,
        CefWindowInfo& windowInfo,
        CefRefPtr<CefClient>& client,
        CefBrowserSettings& settings,
        CefRefPtr<CefDictionaryValue>& extra_info,
        bool* no_javascript_access) override;
    void OnBeforePopupAborted(CefRefPtr<CefBrowser> browser,
        int popup_id) override;
    void OnBeforeClose(CefRefPtr<CefBrowser> browser) override;

private:
    // Used to determine the object type.
    //virtual const void* GetTypeKey() const override { return &kTypeKey; }
    //static constexpr int kTypeKey = 0;

    const bool use_alloy_style_;

    IMPLEMENT_REFCOUNTING(DefaultClientHandler);
    DISALLOW_COPY_AND_ASSIGN(DefaultClientHandler);
};


ClientAppBrowser::ClientAppBrowser() = default;

void ClientAppBrowser::OnRegisterCustomPreferences(
    cef_preferences_type_t type,
    CefRawPtr<CefPreferenceRegistrar> registrar)
{
    //if (type == CEF_PREFERENCES_TYPE_GLOBAL) {
    //    // Register global preferences with default values.
    //    prefs::RegisterGlobalPreferences(registrar);
    //}
}

void ClientAppBrowser::OnContextInitialized()
{
    if (CefCrashReportingEnabled()) {
        // Set some crash keys for testing purposes. Keys must be defined in the
        // "crash_reporter.cfg" file. See cef_crash_util.h for details.
        CefSetCrashKeyValue("testkey_small1", "value1_small_browser");
        CefSetCrashKeyValue("testkey_small2", "value2_small_browser");
        CefSetCrashKeyValue("testkey_medium1", "value1_medium_browser");
        CefSetCrashKeyValue("testkey_medium2", "value2_medium_browser");
        CefSetCrashKeyValue("testkey_large1", "value1_large_browser");
        CefSetCrashKeyValue("testkey_large2", "value2_large_browser");
    }

    //const std::string& crl_sets_path = CefCommandLine::GetGlobalCommandLine()->GetSwitchValue(switches::kCRLSetsPath);
    //if (!crl_sets_path.empty())
    //{
    //    // Load the CRLSets file from the specified path.
    //    CefLoadCRLSetsFile(crl_sets_path);
    //}
}

void ClientAppBrowser::OnBeforeCommandLineProcessing(const CefString& process_type, CefRefPtr<CefCommandLine> command_line)
{

    if (process_type.empty())
    {
        // Pass additional command-line flags when off-screen rendering is enabled.
        if (command_line->HasSwitch("off-screen-rendering-enabled") &&
            !command_line->HasSwitch("shared-texture-enabled"))
        {
            //// Use software rendering and compositing (disable GPU) for increased FPS
            //// and decreased CPU usage. This will also disable WebGL so remove these
            //// switches if you need that capability.
            //// See https://github.com/chromiumembedded/cef/issues/1257 for details.
            //if (!command_line->HasSwitch(switches::kEnableGPU))
            //{
            //    command_line->AppendSwitch("disable-gpu");
            //    command_line->AppendSwitch("disable-gpu-compositing");
            //}
        }

        //// Append Chromium command line parameters if touch events are enabled
        //if (client::MainContext::Get()->TouchEventsEnabled()) {
        //    command_line->AppendSwitchWithValue("touch-events", "enabled");
        //}
    }
}

bool ClientAppBrowser::OnAlreadyRunningAppRelaunch(CefRefPtr<CefCommandLine> command_line, const CefString& current_directory)
{
    //// Add logging for some common switches that the user may attempt to use.
    //static const char* kIgnoredSwitches[] = {
    //    switches::kMultiThreadedMessageLoop,
    //    switches::kOffScreenRenderingEnabled,
    //    switches::kUseViews,
    //};
    //for (auto& kIgnoredSwitche : kIgnoredSwitches) {
    //    if (command_line->HasSwitch(kIgnoredSwitche)) {
    //        LOG(WARNING) << "The --" << kIgnoredSwitche
    //            << " command-line switch is ignored on app relaunch.";
    //    }
    //}

    //// Create a new root window based on |command_line|.
    //auto config = std::make_unique<RootWindowConfig>(command_line->Copy());

    //MainContext::Get()->GetRootWindowManager()->CreateRootWindow(
    //    std::move(config));

    // Relaunch was handled.
    return true;
}

//CefRefPtr<CefClient> ClientAppBrowser::GetDefaultClient()
//{
//    // Default client handler for unmanaged browser windows. Used with
//    // Chrome style only.
//    LOG(INFO) << "Creating a chrome browser with the default client";
//    return new DefaultClientHandler();
//}


void ClientAppBrowser::OnScheduleMessagePumpWork(int64_t delay) {
    // Only used when `--external-message-pump` is passed via the command-line.
    //MainMessageLoopExternalPump* message_pump = MainMessageLoopExternalPump::Get();
    //if (message_pump)
    //{
    //    message_pump->OnScheduleMessagePumpWork(delay);
    //}
}




DefaultClientHandler::DefaultClientHandler(std::optional<bool> use_alloy_style)
    : use_alloy_style_(false)//use_alloy_style.value_or(MainContext::Get()->UseAlloyStyleGlobal()))
{
}

// static
//CefRefPtr<DefaultClientHandler> DefaultClientHandler::GetForClient(CefRefPtr<CefClient> client)
//{
//    auto base = BaseClientHandler::GetForClient(client);
//    if (base && base->GetTypeKey() == &kTypeKey)
//    {
//        return static_cast<DefaultClientHandler*>(base.get());
//    }
//    return nullptr;
//}

bool DefaultClientHandler::OnBeforePopup(
    CefRefPtr<CefBrowser> browser,
    CefRefPtr<CefFrame> frame,
    int popup_id,
    const CefString& target_url,
    const CefString& target_frame_name,
    CefLifeSpanHandler::WindowOpenDisposition target_disposition,
    bool user_gesture,
    const CefPopupFeatures& popupFeatures,
    CefWindowInfo& windowInfo,
    CefRefPtr<CefClient>& client,
    CefBrowserSettings& settings,
    CefRefPtr<CefDictionaryValue>& extra_info,
    bool* no_javascript_access) {
    CEF_REQUIRE_UI_THREAD();

    if (target_disposition == CEF_WOD_NEW_PICTURE_IN_PICTURE) {
        // Use default handling for document picture-in-picture popups.
        client = nullptr;
        return false;
    }

    // Allow popup creation.
    return false;
}

void DefaultClientHandler::OnBeforePopupAborted(CefRefPtr<CefBrowser> browser, int popup_id)
{
    CEF_REQUIRE_UI_THREAD();
}

void DefaultClientHandler::OnBeforeClose(CefRefPtr<CefBrowser> browser)
{
    CEF_REQUIRE_UI_THREAD();

    // Close all popups that have this browser as the opener.
    OnBeforePopupAborted(browser, /*popup_id=*/-1);

//    BaseClientHandler::OnBeforeClose(browser);
}









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
        app = new ClientAppBrowser();
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
    gRenderHandler = new RenderHandler((int)gDxCtx->browserW, (int)gDxCtx->browserH);
    gClient = new CefOSRClient(gRenderHandler);

    CefWindowInfo wi;
    wi.SetAsWindowless(gHwnd);
    CefBrowserSettings bs;
    bs.windowless_frame_rate = 60;

    CefBrowserHost::CreateBrowser(wi, gClient.get(), "https://example.org", bs, nullptr, nullptr);

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
