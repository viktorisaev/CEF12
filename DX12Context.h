#pragma once

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

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;


struct Vertex
{
    float pos[3];
    float uv[2];
};

struct Vector2D 
{
    float x;
    float y;
};

struct DX12Context {

    static int gClientWidth;
    static int gClientHeight;

    // Core
    ComPtr<IDXGIFactory6> factory;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<IDXGISwapChain3> swapChain;
    static const UINT kBackBufferCount = 2;
    UINT frameIndex = 0;
    ComPtr<ID3D12Resource> backBuffers[kBackBufferCount];
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    UINT rtvDescriptorSize = 0;
    ComPtr<ID3D12Resource> msaaRenderTargets[kBackBufferCount];     //MSAA
    ComPtr<ID3D12DescriptorHeap> msaaHeap;
    UINT msaaDescriptorSize = 0;
    ComPtr<ID3D12CommandAllocator> alloc[kBackBufferCount];
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceValue = 0;
    HANDLE fenceEvent = nullptr;

    // Pipeline
    ComPtr<ID3D12RootSignature> rootSig;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12DescriptorHeap> srvHeap;
    ComPtr<ID3D12Resource> vb;
    D3D12_VERTEX_BUFFER_VIEW vbView{};
    ComPtr<ID3D12Resource> cb;
    UINT8* cbCpu = nullptr;

    // Browser texture (DX12) + SRV
    UINT browserW = 1024;
    UINT browserH = 1024;
    UINT rowPitch = ((browserW * 4 + 255) & ~255u);
    UINT uploadSize = rowPitch * browserH;
    ComPtr<ID3D12Resource> uploadTexResource;
    ComPtr<ID3D12Resource> browserTex;
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu{};
    D3D12_GPU_DESCRIPTOR_HANDLE srvGpu{};

    // Upload heap for software path
//    ComPtr<ID3D12Resource> uploadHeap;

    // D3D11-on-12
    ComPtr<ID3D11Device> d3d11Device;
//    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11Texture2D> uploadTex;  // resource to populate data from CPU to GPU so later it could be copied to sharedBrowserTex
    ComPtr<ID3D11Texture2D> sharedBrowserTex;
    ComPtr<IDXGIResource1> dxgiRes;

    HANDLE sharedBrowserHandle = nullptr;
    ComPtr<ID3D12Resource> sharedTex12;


    ComPtr<ID3D11DeviceContext> d3d11ctx;
    ComPtr<ID3D11On12Device> d3d11on12;
//    D3D11_RESOURCE_FLAGS d3d11Flags{ /*D3D11_BIND_RENDER_TARGET |*/ D3D11_BIND_SHADER_RESOURCE };

    D3D11_QUERY_DESC queryDesc = { D3D11_QUERY_EVENT, 0 };
    ComPtr<ID3D11Query> query;

    // Synchronization for copy
    std::mutex mtx;

    // imgui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap>        g_pd3dSrvDescHeap;


    void WaitGPU();
    void Init(HWND hwnd, UINT width, UINT height);
    void Finalize();

    void UpdateTexture12(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds);
    void UpdateTexture11to12(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds);    // we create DirectX11 texture to be used in DirectX12 (like CEF does)
    void UpdateTexture11to12WithMouse(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds, float mouseX, float mouseY, bool mouseValid);

    void Resize(UINT w, UINT h);
    void CreateMSAATextures(UINT newWidth, UINT newHeight);  // could be called multipla times, when the window size changed
    Vector2D Begin(std::chrono::steady_clock::time_point timeStamp, float mouseX, float mouseY);    // returns mouse pos 0..1 both axes
    void End();
    void UploadSoftwareBitmap(const void* srcBGRA, UINT srcStride);
    void CopyFromD3D11Shared(HANDLE sharedBrowserHandle);

    static void ThrowIfFailed(HRESULT hr) {
        if (FAILED(hr)) { throw std::runtime_error("HRESULT failed"); }
    }
};
