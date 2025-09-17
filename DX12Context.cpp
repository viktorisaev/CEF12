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
#include "corecrt_math_defines.h"
#include <DirectXMath.h>

#include "DX12Context.h"

#include <imgui.h>
#include <imgui_impl_dx12.h>


#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

// ------------------------- DX12 helpers ------------------------------

static const float RotationStrength = 8.0f;   // how strong the window reacts on mouse. Less the value - stronger the feedback.

/*static*/ int DX12Context::gClientWidth = 1280;
/*static*/ int DX12Context::gClientHeight = 720;
int gSampleDesc = 4;    // sample desc used for MSAA


static const char* g_vs = R"(
cbuffer CB : register(b0)
{
  float4x4 mvp;
  float2   mouse;
};
struct VSIn { float3 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };
VSOut main(VSIn i) {
  VSOut o;
  o.pos = mul(mvp, float4(i.pos, 1));
  o.uv = i.uv;
  return o;
}
)";

static const char* g_ps = R"(
cbuffer CB : register(b0)
{
  float4x4 mvp;
  float2   mouse;
};
Texture2D tex0 : register(t0);
SamplerState samp0 : register(s0);

float4 main(float4 pos:SV_POSITION, float2 uv:TEXCOORD0) : SV_Target
{
// mouse pos
    float d = abs(length(uv - mouse) - 0.012);
    float c = smoothstep(0.001, 0.002, d);
    float4 texSmpl = tex0.Sample(samp0, uv);
    float4 finalColor = texSmpl * c + float4(1.0, 0.419, 0.419, 1.0) * (1.0 - c);
  return finalColor;
//  return float4(uv.x, uv.y, 1.0f, 1.0f);
}
)";


void DX12Context::WaitGPU() {
    const UINT64 fv = ++fenceValue;
    queue->Signal(fence.Get(), fv);
    if (fence->GetCompletedValue() < fv)
    {
        fence->SetEventOnCompletion(fv, fenceEvent);
        WaitForSingleObject(fenceEvent, INFINITE);
    }
}


void DX12Context::Init(HWND hwnd, UINT width, UINT height)
{
    browserW = 1024;
    browserH = 1024;

    UINT flags = 0;

    CD3DX12_HEAP_PROPERTIES uploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

#if defined(_DEBUG)
    {
        ComPtr<ID3D12Debug> dbg;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
        flags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif
    ThrowIfFailed(CreateDXGIFactory2(flags, IID_PPV_ARGS(&factory)));
    ThrowIfFailed(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ThrowIfFailed(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)));

    DXGI_SWAP_CHAIN_DESC1 sd{};
    sd.Width = width;
    sd.Height = height;
    sd.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    sd.SampleDesc.Count = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.BufferCount = kBackBufferCount;
    sd.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;

    ComPtr<IDXGISwapChain1> sc1;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(queue.Get(), hwnd, &sd, nullptr, nullptr, &sc1));
    ThrowIfFailed(sc1.As(&swapChain));
    frameIndex = swapChain->GetCurrentBackBufferIndex();


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// RENDER TARGET
    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvDesc{};
    rtvDesc.NumDescriptors = kBackBufferCount;
    rtvDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&rtvDesc, IID_PPV_ARGS(&rtvHeap)));
    rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    //// MSAA heap
    D3D12_DESCRIPTOR_HEAP_DESC msaaHeapDesc{};
    msaaHeapDesc.NumDescriptors = kBackBufferCount;
    msaaHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    ThrowIfFailed(device->CreateDescriptorHeap(&msaaHeapDesc, IID_PPV_ARGS(&msaaHeap)));
    msaaDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    D3D12_RESOURCE_DESC msaaDesc = {};
    msaaDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    msaaDesc.Width = width;
    msaaDesc.Height = height;
    msaaDesc.DepthOrArraySize = 1;
    msaaDesc.MipLevels = 1;
    msaaDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    msaaDesc.SampleDesc.Count = gSampleDesc;
    msaaDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    msaaDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    msaaDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = msaaDesc.Format;
    clearValue.Color[0] = 0.05f;
    clearValue.Color[1] = 0.05f;
    clearValue.Color[2] = 0.08f;
    clearValue.Color[3] = 1.0f;


    // Back buffers
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvH(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    CD3DX12_CPU_DESCRIPTOR_HANDLE msaaRtvH(msaaHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kBackBufferCount; ++i) {
        // back buffers
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvH);
        backBuffers[i]->SetName(L"RTV");
        rtvH.ptr += rtvDescriptorSize;
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i])));
    }

    // MSAA buffers
    CreateMSAATextures(width, height);

    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[frameIndex].Get(), nullptr, IID_PPV_ARGS(&cmdList)));
    cmdList->Close();

    // Fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ROOT SIGNATURE
    // Root signature
    // 1)
    CD3DX12_ROOT_PARAMETER rp[2];
    rp[0].InitAsConstantBufferView(0);                                                               // b0
    // 2)
    CD3DX12_DESCRIPTOR_RANGE range;
    range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);         // t0
    // Sampler
    CD3DX12_STATIC_SAMPLER_DESC samp(0,
        D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
        D3D12_TEXTURE_ADDRESS_MODE_CLAMP);                                                                       // s0
    CD3DX12_ROOT_SIGNATURE_DESC rsDesc(2, rp, 1, &samp, D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);
    ComPtr<ID3DBlob> sig, err;
    ThrowIfFailed(D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err));
    ThrowIfFailed(device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&rootSig)));

    // Shaders
    ComPtr<ID3DBlob> vsb, psb;
    ThrowIfFailed(D3DCompile(g_vs, strlen(g_vs), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsb, &err));
    ThrowIfFailed(D3DCompile(g_ps, strlen(g_ps), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &psb, &err));

    D3D12_INPUT_ELEMENT_DESC inputLayout[] = {
        {"POSITION",0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},         // SV_POSITION
        {"TEXCOORD",0, DXGI_FORMAT_R32G32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},        // TEXCOORD0
    };

    D3D12_RASTERIZER_DESC rasterDesc = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    rasterDesc.CullMode = D3D12_CULL_MODE_NONE; // <— disables culling

    D3D12_RENDER_TARGET_BLEND_DESC rtBlendDesc = {};
    rtBlendDesc.BlendEnable = TRUE;
    rtBlendDesc.SrcBlend = D3D12_BLEND_SRC_ALPHA;
    rtBlendDesc.DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
    rtBlendDesc.BlendOp = D3D12_BLEND_OP_ADD;
    rtBlendDesc.SrcBlendAlpha = D3D12_BLEND_ONE;
    rtBlendDesc.DestBlendAlpha = D3D12_BLEND_ZERO;
    rtBlendDesc.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    rtBlendDesc.RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;

    D3D12_BLEND_DESC blendDesc = {};
    blendDesc.AlphaToCoverageEnable = FALSE;
    blendDesc.IndependentBlendEnable = FALSE;
    blendDesc.RenderTarget[0] = rtBlendDesc;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.VS = { vsb->GetBufferPointer(), vsb->GetBufferSize() };
    psoDesc.PS = { psb->GetBufferPointer(), psb->GetBufferSize() };
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.RasterizerState = rasterDesc;
    psoDesc.BlendState = blendDesc;// CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_B8G8R8A8_UNORM;
    psoDesc.SampleDesc.Count = gSampleDesc;
    psoDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)));


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// QUAD
    // Quad vertices (cover most of the window)
    Vertex quad[6] = {
        {{-1.0f,-1.0f, 0},{0,1}},
        {{-1.0f, 1.0f, 0},{0,0}},
        {{ 1.0f,-1.0f, 0},{1,1}},
        {{ 1.0f,-1.0f, 0},{1,1}},
        {{-1.0f, 1.0f, 0},{0,0}},
        {{ 1.0f, 1.0f, 0},{1,0}}
    };
    const UINT vbSize = sizeof(quad);

    // Upload VB
    ComPtr<ID3D12Resource> vbUpload;
    CD3DX12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
    ThrowIfFailed(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&vbUpload)));
    ThrowIfFailed(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&vb)));

    // Copy data
    void* p;
    CD3DX12_RANGE r(0, 0);
    vbUpload->Map(0, &r, &p);
    memcpy(p, quad, vbSize);
    vbUpload->Unmap(0, nullptr);

    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));
    cmdList->CopyBufferRegion(vb.Get(), 0, vbUpload.Get(), 0, vbSize);
    CD3DX12_RESOURCE_BARRIER vbBarrier = CD3DX12_RESOURCE_BARRIER::Transition(vb.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
    cmdList->ResourceBarrier(1, &vbBarrier);
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);
    WaitGPU();
    vbView = { vb->GetGPUVirtualAddress(), vbSize, sizeof(Vertex) };

    // Constant buffer
    CD3DX12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(float)*16 + sizeof(float)*2); // one 4x4 matrix + 2 floats (mouse coords)
    ThrowIfFailed(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&cb)));
    cb->Map(0, nullptr, reinterpret_cast<void**>(&cbCpu));


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SRV HEAP
    // SRV heap
    D3D12_DESCRIPTOR_HEAP_DESC sh{};
    sh.NumDescriptors = 1; sh.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    sh.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(device->CreateDescriptorHeap(&sh, IID_PPV_ARGS(&srvHeap)));
    srvCpu = srvHeap->GetCPUDescriptorHandleForHeapStart();
    srvGpu = srvHeap->GetGPUDescriptorHandleForHeapStart();

    // Browser DX12 texture
    D3D12_RESOURCE_DESC td = {};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = browserW;
    td.Height = browserH;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Flags = D3D12_RESOURCE_FLAG_NONE;
    td.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    ThrowIfFailed(device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&browserTex)));

    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};
    sv.Format = td.Format;
    sv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    sv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    sv.Texture2D.MipLevels = 1;
    device->CreateShaderResourceView(browserTex.Get(), &sv, srvCpu);

    // Upload heap for software paints
    rowPitch = ((browserW * 4 + 255) & ~255u);
    uploadSize = rowPitch * browserH;
    CD3DX12_RESOURCE_DESC upDesc = CD3DX12_RESOURCE_DESC::Buffer(uploadSize);
    ThrowIfFailed(device->CreateCommittedResource(&uploadHeap, D3D12_HEAP_FLAG_NONE, &upDesc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadTexResource)));
    uploadTexResource->SetName(L"UploadTexture");

    //// D3D11-on-12
    D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    ID3D12CommandQueue* queues[] = { queue.Get() };
    ThrowIfFailed(D3D11On12CreateDevice(device.Get(), 0, &fl, 1, reinterpret_cast<IUnknown**>(queues), 1, 0, &d3d11Device, &d3d11ctx, nullptr));


    // Create DirectX11 shared texture resource to simulate OnAcceleratedPaint from CEF
    // to be removed later

    D3D11_TEXTURE2D_DESC desc = {};

    // target GPU shareable texture 
    desc.Width = browserW;
    desc.Height = browserH;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;       // GPU shareable texture!
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = 0;                // no access from CPU
    desc.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE; // or _KEYEDMUTEX (but as soon as it is not supported by CEF in default, we don't use it)

    ThrowIfFailed(d3d11Device->CreateTexture2D(&desc, nullptr, &sharedBrowserTex));

    // upload texture
    desc.Width = browserW;
    desc.Height = browserH;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DYNAMIC;       // texture to update from CPU side, but not shareable!
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;   // we'll write from CPU
    desc.MiscFlags = 0; // we can't share this texture

    ThrowIfFailed(d3d11Device->CreateTexture2D(&desc, nullptr, &uploadTex));

    // create handle to share
    ThrowIfFailed(sharedBrowserTex.As(&dxgiRes));

    ThrowIfFailed(dxgiRes->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, L"SharedBrowserTex", &sharedBrowserHandle));

    // create query to wait for the texture to be fully transfered
//    D3D11_QUERY_DESC queryDesc = { D3D11_QUERY_EVENT, 0 };
    ThrowIfFailed(d3d11Device->CreateQuery(&queryDesc, &query));

    // move to render per frame
//    UpdateTexture12(uploadTexResource);
//    UpdateTexture11to12(uploadTexResource, 0); // green-ish texture

        // imgui
    D3D12_DESCRIPTOR_HEAP_DESC imguiHeapDesc = {};
    imguiHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    imguiHeapDesc.NumDescriptors = 1;
    imguiHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    device->CreateDescriptorHeap(&imguiHeapDesc, IID_PPV_ARGS(&g_pd3dSrvDescHeap));

    ImGui_ImplDX12_Init(NULL, kBackBufferCount, device.Get(), g_pd3dSrvDescHeap.Get()->GetCPUDescriptorHandleForHeapStart(), g_pd3dSrvDescHeap.Get()->GetGPUDescriptorHandleForHeapStart());

}


void DX12Context::Finalize()
{
    CloseHandle(sharedBrowserHandle);
}


void DX12Context::UpdateTexture11to12(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds)
{
    // 1) populate DirectX11 upload texture with data
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    ThrowIfFailed(d3d11ctx->Map(uploadTex.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));

    // mapped.pData points to the first row
    // mapped.RowPitch is the number of bytes per row (may be > width * bytesPerPixel)

    // Build a rotation matrix (pitch, yaw, roll in radians)
    DirectX::XMMATRIX rotMat = DirectX::XMMatrixRotationRollPitchYaw(
        DirectX::XMConvertToRadians(0),   // pitch
        DirectX::XMConvertToRadians(0),  // yaw
        DirectX::XMConvertToRadians(float(milliseconds/ 1000.0f) * 45.0f)    // roll
    );

    uint32_t* p = reinterpret_cast<uint32_t*>(mapped.pData);
    if (p)
    {
        // fill texture with something
        for (UINT h = 0; h < browserH; ++h)
        {
            for (UINT v = 0; v < browserW; ++v)
            {
                DirectX::XMFLOAT2 pos = DirectX::XMFLOAT2(
                    float(2 * int(v) - int(browserW)) / float(browserW),
                    float(2 * int(h) - int(browserH)) / float(browserH)
                );
                DirectX::XMVECTOR vec = DirectX::XMLoadFloat2(&pos);

                // Transform the vector
                DirectX::XMVECTOR vRotated = DirectX::XMVector3Transform(vec, rotMat);

                DirectX::XMVECTOR lenV = DirectX::XMVector2Length(vRotated);
                float len = DirectX::XMVectorGetX(lenV);

                uint32_t c = 0xFFFFFFFF;    // white by default

                if ( (len < 0.35 || len > 0.65f) || DirectX::XMVectorGetX(vRotated) < 0)
                {
                     c = 
                          0xFF                                                  << 24
                        | ((0xFF * h) / browserH)                               << 16
                        | (byte(((milliseconds / 8) % 256) + (h * v) / 32))     << 8
                        | ((0xFF * v) / browserW)
                    ;
                }
                p[h * (rowPitch/4) + v] = c;
            }
        }
    }
    d3d11ctx->Unmap(uploadTex.Get(), 0);


    // 2) transfer upload to final texture sharedBrowserTex
    // wait for the operation to be fully finished before "pass" the DirectX11 texture to DirectX12

//    d3d11ctx->Begin(query.Get());
    d3d11ctx->CopyResource(sharedBrowserTex.Get(), uploadTex.Get());

    // flush, or..
//    d3d11ctx->Flush(); // pushes all pending commands to the GPU
    // ..wait until the operation is finished
    d3d11ctx->End(query.Get());
    while (S_OK != d3d11ctx->GetData(query.Get(), nullptr, 0, 0)) {
        Sleep(0); // or yield
    }

    // 3) get the DirectX11 texture in DirectX12
    device->OpenSharedHandle(sharedBrowserHandle, IID_PPV_ARGS(&sharedTex12));

    CD3DX12_RESOURCE_BARRIER texBarrierBefore12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &texBarrierBefore12);
    cmdList->CopyResource(browserTex.Get(), sharedTex12.Get());
    CD3DX12_RESOURCE_BARRIER texBarrierAfter12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &texBarrierAfter12);
}

void DX12Context::UpdateTexture11to12WithMouse(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds, float mouseX, float mouseY, bool mouseValid)
{
    // 1) populate DirectX11 upload texture with data
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    ThrowIfFailed(d3d11ctx->Map(uploadTex.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));

    // mapped.pData points to the first row
    // mapped.RowPitch is the number of bytes per row (may be > width * bytesPerPixel)

    uint32_t* p = reinterpret_cast<uint32_t*>(mapped.pData);
    if (p)
    {
        // fill texture with something
        for (UINT h = 0; h < browserH; ++h)
        {
            for (UINT v = 0; v < browserW; ++v)
            {
                //DirectX::XMFLOAT2 pos = DirectX::XMFLOAT2(
                //    float(2 * int(v) - int(browserW)) / float(browserW) - mouseX,
                //    float(2 * int(h) - int(browserH)) / float(browserH) - mouseY
                //);
                //DirectX::XMVECTOR vec = DirectX::XMLoadFloat2(&pos);

                //DirectX::XMVECTOR lenV = DirectX::XMVector2Length(vec);
                //float len = DirectX::XMVectorGetX(lenV);

                uint32_t c = 0xFFFFFFFF;    // white by default

//                if (len > 0.014 || !mouseValid)
                {
                    c =
                        0xFF << 24
                        | ((0xFF * h) / browserH) << 16
                        | (byte(((milliseconds / 8) % 256) + (h * v) / 32)) << 8
                        | ((0xFF * v) / browserW)
                        ;
                }
                p[h * (rowPitch / 4) + v] = c;
            }
        }
    }
    d3d11ctx->Unmap(uploadTex.Get(), 0);


    // 2) transfer upload to final texture sharedBrowserTex
    // wait for the operation to be fully finished before "pass" the DirectX11 texture to DirectX12

//    d3d11ctx->Begin(query.Get());
    d3d11ctx->CopyResource(sharedBrowserTex.Get(), uploadTex.Get());

    // flush, or..
//    d3d11ctx->Flush(); // pushes all pending commands to the GPU
    // ..wait until the operation is finished
    d3d11ctx->End(query.Get());
    while (S_OK != d3d11ctx->GetData(query.Get(), nullptr, 0, 0)) {
        Sleep(0); // or yield
    }

    // 3) get the DirectX11 texture in DirectX12
    device->OpenSharedHandle(sharedBrowserHandle, IID_PPV_ARGS(&sharedTex12));

    CD3DX12_RESOURCE_BARRIER texBarrierBefore12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &texBarrierBefore12);
    cmdList->CopyResource(browserTex.Get(), sharedTex12.Get());
    CD3DX12_RESOURCE_BARRIER texBarrierAfter12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &texBarrierAfter12);
}



void DX12Context::UpdateTexture12(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds)
{
    {
        byte* p = static_cast<byte*>(malloc(uploadSize));
        if (p)
        {
            // fill texture with something
            for (UINT h = 0; h < browserH; ++h)
            {
                for (UINT v = 0; v < browserW; ++v)
                {
                    p[h * rowPitch + v * 4 + 0] = (0xFF * v) / browserW;
                    p[h * rowPitch + v * 4 + 1] = (0xFF * h) / browserH;
                    p[h * rowPitch + v * 4 + 2] = byte(((milliseconds / 8) % 256) + (h * v) / 32);
                    p[h * rowPitch + v * 4 + 3] = 0xFF;
                }
            }

            //CD3DX12_RANGE r(0, 0);
            //browserTex->Map(0, &r, reinterpret_cast<void**>(&p));
            //browserTex->Unmap(0, nullptr);

            // use the cmdList that is in progress
            //ThrowIfFailed(alloc[frameIndex]->Reset());
            //ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));

            D3D12_SUBRESOURCE_DATA uploadData = {};
            uploadData.pData = reinterpret_cast<BYTE*>(p);
            uploadData.RowPitch = browserW * 4;
            uploadData.SlicePitch = uploadSize;

            CD3DX12_RESOURCE_BARRIER texBarrierBefore = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
            cmdList->ResourceBarrier(1, &texBarrierBefore);
            UpdateSubresources(cmdList.Get(), browserTex.Get(), uploadTexResource.Get(), 0, 0, 1, &uploadData);
            CD3DX12_RESOURCE_BARRIER texBarrierAfter = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
            cmdList->ResourceBarrier(1, &texBarrierAfter);

            // use the cmdList that is in progress
            //ThrowIfFailed(cmdList->Close());

            //ID3D12CommandList* lists[] = { cmdList.Get() };
            //queue->ExecuteCommandLists(1, lists);
            //WaitGPU();
        }

        free(p);
    }
}

void DX12Context::Resize(UINT w, UINT h)
{
    if (!device) return;
    // sync on GPU done
    WaitGPU();

    for (UINT i = 0; i < kBackBufferCount; ++i) backBuffers[i].Reset();
    DXGI_SWAP_CHAIN_DESC scd{};
    swapChain->GetDesc(&scd);
    ThrowIfFailed(swapChain->ResizeBuffers(kBackBufferCount, w, h, scd.BufferDesc.Format, scd.Flags));
    frameIndex = swapChain->GetCurrentBackBufferIndex();
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvH(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kBackBufferCount; ++i) {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvH);
        rtvH.ptr += rtvDescriptorSize;
    }

    // re-create MSAA textures for the changed resolution
    CreateMSAATextures(w, h);
}

void DX12Context::CreateMSAATextures(UINT newWidth, UINT newHeight)
{
    D3D12_RESOURCE_DESC msaaDesc = {};
    msaaDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    msaaDesc.Width = newWidth;
    msaaDesc.Height = newHeight;
    msaaDesc.DepthOrArraySize = 1;
    msaaDesc.MipLevels = 1;
    msaaDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    msaaDesc.SampleDesc.Count = gSampleDesc;
    msaaDesc.SampleDesc.Quality = DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN;
    msaaDesc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    msaaDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    D3D12_CLEAR_VALUE clearValue = {};
    clearValue.Format = msaaDesc.Format;
    clearValue.Color[0] = 0.05f;
    clearValue.Color[1] = 0.05f;
    clearValue.Color[2] = 0.08f;
    clearValue.Color[3] = 1.0f;

    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);

    CD3DX12_CPU_DESCRIPTOR_HANDLE msaaRtvH(msaaHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kBackBufferCount; ++i) 
    {
        // release previous resources
        msaaRenderTargets[i].Reset();

        // MSAA buffers
        ThrowIfFailed(device->CreateCommittedResource(
            &defaultHeap,
            D3D12_HEAP_FLAG_NONE,
            &msaaDesc,
            D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
            &clearValue,
            IID_PPV_ARGS(&msaaRenderTargets[i])
        ));
        msaaRenderTargets[i]->SetName(L"msaaRT");
        device->CreateRenderTargetView(msaaRenderTargets[i].Get(), nullptr, msaaRtvH);
        msaaRtvH.ptr += msaaDescriptorSize;
    }
}



// Builds an MVP matrix in DirectXMath (row-major)
inline DirectX::XMMATRIX BuildMVP(const DirectX::XMMATRIX& model, const DirectX::XMMATRIX& view, const DirectX::XMMATRIX& projection)
{
    // In DirectXMath (row-major), transformations are applied left-to-right
    // So: vertex * model * view * projection
    return XMMatrixMultiply(XMMatrixMultiply(model, view), projection);
}

DirectX::XMMATRIX RotationMatrixFromTwoVectors(DirectX::FXMVECTOR from, DirectX::FXMVECTOR to)
{
    // Normalize input vectors
    DirectX::XMVECTOR v0 = DirectX::XMVector3Normalize(from);
    DirectX::XMVECTOR v1 = DirectX::XMVector3Normalize(to);

    // Dot product
    float dot = DirectX::XMVectorGetX(DirectX::XMVector3Dot(v0, v1));

    // If vectors are almost identical -> identity matrix
    if (dot > 0.999999f)
        return DirectX::XMMatrixIdentity();

    // If vectors are opposite -> 180 rotation around any perpendicular axis
    if (dot < -0.999999f)
    {
        DirectX::XMVECTOR ortho = DirectX::XMVector3Orthogonal(v0);
        DirectX::XMVECTOR q = DirectX::XMQuaternionRotationAxis(DirectX::XMVector3Normalize(ortho), DirectX::XM_PI);
        return DirectX::XMMatrixRotationQuaternion(q);
    }

    // Cross product
    DirectX::XMVECTOR cross = DirectX::XMVector3Cross(v0, v1);

    // Build quaternion directly
    DirectX::XMVECTOR q = DirectX::XMVectorSet(
        DirectX::XMVectorGetX(cross),
        DirectX::XMVectorGetY(cross),
        DirectX::XMVectorGetZ(cross),
        1.0f + dot
    );

    // Normalize quaternion
    q = DirectX::XMQuaternionNormalize(q);

    // Convert quaternion to matrix
    return DirectX::XMMatrixRotationQuaternion(q);
}



 bool IntersectRayAndPlane(DirectX::XMVECTOR rayDir, DirectX::XMVECTOR rayOrigin, DirectX::XMVECTOR planeNormal, DirectX::XMVECTOR planeOrigin, DirectX::XMVECTOR &intersectionPoint)
{
     // Compute dot product between ray direction and plane normal
     float denom = DirectX::XMVectorGetX(DirectX::XMVector3Dot(rayDir, planeNormal));

     // Check if ray is parallel to the plane
     if (fabs(denom) < 1e-6f)
     {
         return false; // No intersection, ray is parallel
     }

     // Compute vector from ray origin to a point on the plane
     DirectX::XMVECTOR diff = DirectX::XMVectorSubtract(planeOrigin, rayOrigin);

     // Compute distance along ray to intersection point
     float t = DirectX::XMVectorGetX(DirectX::XMVector3Dot(diff, planeNormal)) / denom;

     if (t < 0.0f)
     {
         return false; // Intersection is behind the ray origin
     }
     else
     {
         // Compute intersection point
         intersectionPoint = DirectX::XMVectorAdd(rayOrigin, DirectX::XMVectorScale(rayDir, t));
         return true;
     }

 }

 // constant buffer: MVP + mouse pos
 struct CBstruct
 {
     DirectX::XMFLOAT4X4 mvp;
     Vector2D mousePos;
 };


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// BEGIN DRAW
 Vector2D DX12Context::Begin(std::chrono::steady_clock::time_point timeStamp, float mouseX, float mouseY)
{
    auto duration = timeStamp.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), pso.Get()));

    // update texture per frame
//    UpdateTexture12(uploadTexResource, milliseconds);   // blue-ish texture
//    UpdateTexture11to12(uploadTexResource, milliseconds); // green-ish texture

    CD3DX12_RESOURCE_BARRIER toRT1 = CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RESOLVE_DEST);
    cmdList->ResourceBarrier(1, &toRT1);

    CD3DX12_RESOURCE_BARRIER toRT2 = CD3DX12_RESOURCE_BARRIER::Transition(msaaRenderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmdList->ResourceBarrier(1, &toRT2);

    FLOAT clear[4] = { 0.05f,0.05f,0.08f,1.0f };
    // no MSAA
    //    CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
    //cmdList->ClearRenderTargetView(rtv, clear, 0, nullptr);
    //cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

    // with MSAA
    CD3DX12_CPU_DESCRIPTOR_HANDLE msaartv(msaaHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, msaaDescriptorSize);
    cmdList->ClearRenderTargetView(msaartv, clear, 0, nullptr);
    cmdList->OMSetRenderTargets(1, &msaartv, FALSE, nullptr);

    D3D12_VIEWPORT vp{ 0,0,(float)gClientWidth,(float)gClientHeight,-1,1 };
    D3D12_RECT sc{ 0,0,gClientWidth,gClientHeight };
    cmdList->RSSetViewports(1, &vp);
    cmdList->RSSetScissorRects(1, &sc);

    ID3D12DescriptorHeap* heaps[] = { srvHeap.Get() };
    cmdList->SetDescriptorHeaps(1, heaps);

    cmdList->SetGraphicsRootSignature(rootSig.Get());


    // rotation by mouse
    DirectX::XMVECTOR mouseVector = DirectX::XMVectorSet(mouseX, mouseY, RotationStrength, 0.0f);
    DirectX::XMVECTOR normalPlane = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);

    DirectX::XMMATRIX model = RotationMatrixFromTwoVectors(normalPlane, mouseVector);
//    DirectX::XMMATRIX model = DirectX::XMMatrixIdentity();    // Debug: flat plane
//    DirectX::XMMATRIX model = DirectX::XMMatrixRotationNormal(DirectX::XMVectorSet(-1.0f, 0.0f, 0.0f, 0.0f), 45.0*DirectX::XM_PI / 180.0); // Debug: 45 deg

    const float SCREEN_DIST = 1.4f; // distance between plane and camera, so zoom in/out the plane on the screen

    DirectX::XMMATRIX view = DirectX::XMMatrixTranslation(0.0f, 0.0f, SCREEN_DIST); // Move object forward

    DirectX::XMMATRIX projection = DirectX::XMMatrixPerspectiveFovLH(
        DirectX::XMConvertToRadians(90.0f),         // Field of view
        float(gClientWidth) / float(gClientHeight),       // Aspect ratio
        0.1f,                                                 // Near plane
        3.0f);                                                // Far plane

    // Build MVP
    CBstruct cbStruct;
    DirectX::XMMATRIX mvpOut = BuildMVP(model, view, projection);

    // Store into a constant buffer-friendly struct
    DirectX::XMStoreFloat4x4(&cbStruct.mvp, mvpOut);

    // project mouse to 3D plane
    float mouseXproj = mouseX;
    float mouseYproj = mouseY;

    //// debug mouse position
    //CHAR buf[1024];
    //sprintf_s(buf, 1024, "mx=%.3f, my=%.3f\n", mouseX, mouseY);
    //OutputDebugStringA(buf);

    DirectX::XMVECTOR mouseRay = DirectX::XMVector3Normalize(DirectX::XMVectorSet(mouseX, mouseY, 1.0f, 0.0f));
    DirectX::XMVECTOR normalPlaneRotated = DirectX::XMVector3Normalize(DirectX::XMVector3TransformNormal(normalPlane, model));
    DirectX::XMVECTOR planeOrigin = DirectX::XMVectorSet(0.0f, 0.0f, SCREEN_DIST, 0.0f);

    DirectX::XMVECTOR intersectionPoint;
    bool isIntersect = IntersectRayAndPlane(mouseRay, DirectX::XMVectorZero(), normalPlaneRotated, planeOrigin, intersectionPoint);

    // project to plane space
    intersectionPoint = DirectX::XMVectorSet(DirectX::XMVectorGetX(intersectionPoint), DirectX::XMVectorGetY(intersectionPoint), DirectX::XMVectorGetZ(intersectionPoint) - SCREEN_DIST, 0.0f);
    DirectX::XMMATRIX invModel = DirectX::XMMatrixInverse(nullptr, model);

    DirectX::XMVECTOR intersectionPointInPlane = DirectX::XMVector4Transform(intersectionPoint, invModel);


    mouseXproj = DirectX::XMVectorGetX(intersectionPointInPlane);
    mouseYproj = -DirectX::XMVectorGetY(intersectionPointInPlane);

    // for debug purposes - fill the test texture
//    UpdateTexture11to12WithMouse(uploadTexResource, milliseconds, mouseXproj, mouseYproj, isIntersect); // green-ish texture with mouse dot

    cbStruct.mousePos = Vector2D {
        (mouseXproj + 1.0f) / 2.0f,   //-1..+1 => 0..1
        (mouseYproj + 1.0f) / 2.0f   //-1..+1 => 0..1
    };

    memcpy(cbCpu, &cbStruct, sizeof(cbStruct));

    cmdList->SetGraphicsRootConstantBufferView(0, cb->GetGPUVirtualAddress());

    UINT uu = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(srvGpu, 0/*m_deviceResources->GetCurrentFrameIndex()*/, uu);
    cmdList->SetGraphicsRootDescriptorTable(1, gpuHandle);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->IASetVertexBuffers(0, 1, &vbView);

    return cbStruct.mousePos;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// END DRAW
void DX12Context::End()
{
    // render the browser texture
    cmdList->DrawInstanced(6, 1, 0, 0);

    // ImGui
    ImGui_ImplDX12_NewFrame(cmdList.Get(), DX12Context::gClientWidth, DX12Context::gClientHeight);
    // 1. Show a simple window
    // Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
    {
        ImGui::SetNextWindowSize(ImVec2(400, 150), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("Console output:");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::Text("%i x %i", DX12Context::gClientWidth, DX12Context::gClientHeight);
        ImGui::End();

        //ImVec4 clear_col = ImColor(114, 144, 154);
        //ImGui::ColorEdit3("clear color", (float*)&clear_col);
    }
    cmdList.Get()->SetDescriptorHeaps(1, g_pd3dSrvDescHeap.GetAddressOf());
    ImGui::Render();


    //MSAA
    CD3DX12_RESOURCE_BARRIER toResolve = CD3DX12_RESOURCE_BARRIER::Transition(msaaRenderTargets[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
    cmdList->ResourceBarrier(1, &toResolve);

    cmdList->ResolveSubresource( backBuffers[frameIndex].Get(), 0, msaaRenderTargets[frameIndex].Get(), 0, DXGI_FORMAT_B8G8R8A8_UNORM);

    CD3DX12_RESOURCE_BARRIER toPresent = CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_PRESENT);
    cmdList->ResourceBarrier(1, &toPresent);

    ThrowIfFailed(cmdList->Close());

    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);

    ThrowIfFailed(swapChain->Present(1, 0));

    WaitGPU();

    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

// Upload a software bitmap to browserTex (fallback path)
void DX12Context::UploadSoftwareBitmap(const void* srcBGRA, UINT srcStride)
{
    std::lock_guard<std::mutex> lock(mtx);

    // Copy to upload heap
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    ThrowIfFailed(d3d11ctx->Map(uploadTex.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped));
    for (UINT y = 0; y < browserH; ++y) {
        memcpy(reinterpret_cast<BYTE*>(mapped.pData) + y * ((browserW * 4 + 255) & ~255u), reinterpret_cast<const BYTE*>(srcBGRA) + y * srcStride, browserW * 4);
    }
    d3d11ctx->Unmap(uploadTex.Get(), 0);

    d3d11ctx->CopyResource(sharedBrowserTex.Get(), uploadTex.Get());

    // flush, or..
//    d3d11ctx->Flush(); // pushes all pending commands to the GPU
    // ..wait until the operation is finished
    d3d11ctx->End(query.Get());
    while (S_OK != d3d11ctx->GetData(query.Get(), nullptr, 0, 0)) {
        Sleep(0); // or yield
    }

    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));

    CD3DX12_RESOURCE_BARRIER texBarrierBefore12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &texBarrierBefore12);
    cmdList->CopyResource(browserTex.Get(), sharedTex12.Get());
    CD3DX12_RESOURCE_BARRIER texBarrierAfter12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &texBarrierAfter12);

    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);
    WaitGPU();
}

// Copy from D3D11 shared texture into sharedTex12 (GPU path)
void DX12Context::CopyFromD3D11Shared(HANDLE sharedBrowserHandle)
{
    std::lock_guard<std::mutex> lock(mtx);

    // very quickli copy the DirectX11 texture to DirectX12 browser texture
    // there is no sync possible, so just do it.

    ThrowIfFailed(device->OpenSharedHandle(sharedBrowserHandle, IID_PPV_ARGS(&sharedTex12)));

    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));

    CD3DX12_RESOURCE_BARRIER texBarrierBefore12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_DEST);
    cmdList->ResourceBarrier(1, &texBarrierBefore12);
    cmdList->CopyResource(browserTex.Get(), sharedTex12.Get());
    CD3DX12_RESOURCE_BARRIER texBarrierAfter12 = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &texBarrierAfter12);

    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);
    WaitGPU();
}
