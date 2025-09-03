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

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using Microsoft::WRL::ComPtr;

// ------------------------- DX12 helpers ------------------------------

static const float RotationStrength = 8.0f;   // how strong the window reacts on mouse. Less the value - stronger the feedback.

/*static*/ int DX12Context::gClientWidth = 1280;
/*static*/ int DX12Context::gClientHeight = 720;


static const char* g_vs = R"(
cbuffer CB : register(b0)
{
  float4x4 mvp;
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
Texture2D tex0 : register(t0);
SamplerState samp0 : register(s0);

float4 main(float4 pos:SV_POSITION, float2 uv:TEXCOORD0) : SV_Target
{
  return tex0.Sample(samp0, uv);
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


void DX12Context::Init(HWND hwnd, UINT width, UINT height) {
    browserW = 1024; browserH = 1024;

    UINT flags = 0;
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
    sd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
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

    // Back buffers
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvH(rtvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < kBackBufferCount; ++i) {
        ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffers[i])));
        device->CreateRenderTargetView(backBuffers[i].Get(), nullptr, rtvH);
        rtvH.ptr += rtvDescriptorSize;
        ThrowIfFailed(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc[i])));
    }
    ThrowIfFailed(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc[frameIndex].Get(), nullptr, IID_PPV_ARGS(&cmdList)));
    cmdList->Close();

    // Fence
    ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// ROOT SIGNATURE
    // Root signature
    // 1)
    CD3DX12_ROOT_PARAMETER rp[2];
    rp[0].InitAsConstantBufferView(0);                                                                         // b0
    // 2)
    CD3DX12_DESCRIPTOR_RANGE range;
    range.Init(D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0);
    rp[1].InitAsDescriptorTable(1, &range, D3D12_SHADER_VISIBILITY_PIXEL);                   // t0
    // Sampler
    CD3DX12_STATIC_SAMPLER_DESC samp(0, D3D12_FILTER_MIN_MAG_MIP_LINEAR);                                                  // s0
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

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig.Get();
    psoDesc.VS = { vsb->GetBufferPointer(), vsb->GetBufferSize() };
    psoDesc.PS = { psb->GetBufferPointer(), psb->GetBufferSize() };
    psoDesc.InputLayout = { inputLayout, _countof(inputLayout) };
    psoDesc.RasterizerState = rasterDesc;
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState.DepthEnable = FALSE;
    psoDesc.DepthStencilState.StencilEnable = FALSE;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = 1;
    ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)));


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// QUAD
    // Quad vertices (cover most of the window)
    Vertex quad[6] = {
        {{-0.98f,-0.76f, 0},{0,1}},
        {{-0.98f, 0.76f, 0},{0,0}},
        {{ 0.98f,-0.76f, 0},{1,1}},
        {{ 0.98f,-0.76f, 0},{1,1}},
        {{-0.98f, 0.76f, 0},{0,0}},
        {{ 0.98f, 0.76f, 0},{1,0}}
    };
    const UINT vbSize = sizeof(quad);

    // Upload VB
    ComPtr<ID3D12Resource> vbUpload;
    CD3DX12_HEAP_PROPERTIES uploadHeap(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_HEAP_PROPERTIES defaultHeap(D3D12_HEAP_TYPE_DEFAULT);
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
    CD3DX12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(256); // one 4x4 matrix
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
    td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
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

    // move to render per frame
//    UpdateTexture(uploadTexResource);


    // D3D11-on-12
    //D3D_FEATURE_LEVEL fl = D3D_FEATURE_LEVEL_11_0;
    //ID3D12CommandQueue* queues[] = { queue.Get() };
    //ThrowIfFailed(D3D11On12CreateDevice(device.Get(), 0, &fl, 1, reinterpret_cast<IUnknown**>(queues), 1, 0, &d3d11, &d3d11ctx, nullptr));
    //ThrowIfFailed(d3d11.As(&d3d11on12));
    //ThrowIfFailed(d3d11on12->CreateWrappedResource(browserTex.Get(), &d3d11Flags, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, IID_PPV_ARGS(&wrappedBrowserTex)));
}

void DX12Context::UpdateTexture(Microsoft::WRL::ComPtr<ID3D12Resource>& uploadTexResource, long long milliseconds)
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
                    p[h * rowPitch + v * 4 + 1] = ((milliseconds/8) % 256) + (h * v) /32;
                    p[h * rowPitch + v * 4 + 2] = (0xFF * h) / browserH;
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

        //// --- Create the SRV ---
        //D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        //srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        //srvDesc.Format = texDesc.Format;
        //srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        //srvDesc.Texture2D.MostDetailedMip = 0;
        //srvDesc.Texture2D.MipLevels = texDesc.MipLevels;
        //srvDesc.Texture2D.ResourceMinLODClamp = 0.0f;

        //CD3DX12_CPU_DESCRIPTOR_HANDLE handle(srvHandle, i, srvDescriptorSize);
        //device->CreateShaderResourceView(texture.Get(), &srvDesc, srvGpu);

        //ThrowIfFailed(alloc[frameIndex]->Reset());
        //ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));
        //cmdList->CopyBufferRegion(vb.Get(), 0, vbUpload.Get(), 0, vbSize);
        //CD3DX12_RESOURCE_BARRIER vbBarrier = CD3DX12_RESOURCE_BARRIER::Transition(vb.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        //cmdList->ResourceBarrier(1, &vbBarrier);
        //ThrowIfFailed(cmdList->Close());
        //ID3D12CommandList* lists[] = { cmdList.Get() };
        //queue->ExecuteCommandLists(1, lists);
        //WaitGPU();
    }
}

void DX12Context::Resize(UINT w, UINT h)
{
    if (!device) return;
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




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// BEGIN DRAW
void DX12Context::Begin(std::chrono::steady_clock::time_point timeStamp, float mouseX, float mouseY)
{
    auto duration = timeStamp.time_since_epoch();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), pso.Get()));

    // update texture per frame
    UpdateTexture(uploadTexResource, milliseconds);

    CD3DX12_RESOURCE_BARRIER toRT = CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);
    cmdList->ResourceBarrier(1, &toRT);

    CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(rtvHeap->GetCPUDescriptorHandleForHeapStart(), frameIndex, rtvDescriptorSize);
    FLOAT clear[4] = { 0.05f,0.05f,0.08f,1.0f };
    cmdList->ClearRenderTargetView(rtv, clear, 0, nullptr);
    cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);

    D3D12_VIEWPORT vp{ 0,0,(float)gClientWidth,(float)gClientHeight,-1,1 };
    D3D12_RECT sc{ 0,0,gClientWidth,gClientHeight };
    cmdList->RSSetViewports(1, &vp);
    cmdList->RSSetScissorRects(1, &sc);

    ID3D12DescriptorHeap* heaps[] = { srvHeap.Get() };
    cmdList->SetDescriptorHeaps(1, heaps);

    cmdList->SetGraphicsRootSignature(rootSig.Get());

    //// Simple ortho MVP
    //float mvp[16] = {
    //    1,0,0,0,
    //    0,1,0,0,
    //    0,0,1,0,
    //    0,0,0,1
    //};

    // rotation by mouse
    DirectX::XMVECTOR mouseVector = DirectX::XMVectorSet(mouseX, -mouseY, RotationStrength, 0.0f);
    DirectX::XMVECTOR normalPlane = DirectX::XMVectorSet(0.0f, 0.0f, 1.0f, 0.0f);

    DirectX::XMMATRIX model = RotationMatrixFromTwoVectors(normalPlane, mouseVector);

// rotate by timer    DirectX::XMMATRIX model = DirectX::XMMatrixRotationY(DirectX::XMConvertToRadians(milliseconds * 0.06f)); // Move object forward



    DirectX::XMMATRIX view = DirectX::XMMatrixTranslation(0.0f, 0.0f, 1.0f); // Move object forward
//    model = DirectX::XMMatrixIdentity();

    //DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(
    //    DirectX::XMVectorSet(0.0f, 0.0f, -1.0f, 1.0f), // Camera position
    //    DirectX::XMVectorSet(0.0f, 0.0f, 0.0f, 1.0f),   // Look at origin
    //    DirectX::XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f));  // Up direction
    //view = DirectX::XMMatrixIdentity();


    DirectX::XMMATRIX projection = DirectX::XMMatrixPerspectiveFovLH(
        DirectX::XMConvertToRadians(90.0f),         // Field of view
        float(gClientWidth) / float(gClientHeight),       // Aspect ratio
        0.1f,                                                 // Near plane
        3.0f);                                                // Far plane

    // Build MVP
    DirectX::XMMATRIX mvpOut = BuildMVP(model, view, projection);

//    mvpOut = DirectX::XMMatrixIdentity();

    // Store into a constant buffer-friendly struct
    DirectX::XMFLOAT4X4 mvp;
    DirectX::XMStoreFloat4x4(&mvp, mvpOut);


    //float angle = std::abs((milliseconds * 0.01f) * float(M_PI) / 180.0f); // radians
    //float c = std::cos(angle);
    //float s = std::sin(angle);
    //float mvpOrig[16] = {
    // c,        0.0f,       s,      0.0f,
    // 0.0,      1.0f,       0.0f,   0.0f,
    // -s,       0.0f,       c,      0.0f,
    // 0.0f,     0.0f,       0.0f,   1.0f,
    //};

    memcpy(cbCpu, &mvp, sizeof(mvp));
//    memcpy(cbCpu, &mvpOrig, sizeof(mvpOrig));

    cmdList->SetGraphicsRootConstantBufferView(0, cb->GetGPUVirtualAddress());

    UINT uu = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    CD3DX12_GPU_DESCRIPTOR_HANDLE gpuHandle(srvGpu, 0/*m_deviceResources->GetCurrentFrameIndex()*/, uu);
    cmdList->SetGraphicsRootDescriptorTable(1, gpuHandle);

    cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmdList->IASetVertexBuffers(0, 1, &vbView);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// END DRAW
void DX12Context::End()
{
    cmdList->DrawInstanced(6, 1, 0, 0);

    CD3DX12_RESOURCE_BARRIER toPresent = CD3DX12_RESOURCE_BARRIER::Transition(backBuffers[frameIndex].Get(), D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
    cmdList->ResourceBarrier(1, &toPresent);
    ThrowIfFailed(cmdList->Close());

    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);

    ThrowIfFailed(swapChain->Present(1, 0));

    WaitGPU();

    frameIndex = swapChain->GetCurrentBackBufferIndex();
}

// Upload a software bitmap to browserTex (fallback path)
void DX12Context::UploadSoftwareBitmap(const void* srcBGRA, UINT srcStride) {
    std::lock_guard<std::mutex> lock(mtx);
    // Copy to upload heap
    BYTE* mapped = nullptr;
    CD3DX12_RANGE r(0, 0);
    uploadHeap->Map(0, &r, reinterpret_cast<void**>(&mapped));
    for (UINT y = 0; y < browserH; ++y) {
        memcpy(mapped + y * ((browserW * 4 + 255) & ~255u),
            reinterpret_cast<const BYTE*>(srcBGRA) + y * srcStride,
            browserW * 4);
    }
    uploadHeap->Unmap(0, nullptr);

    // Copy into browserTex
    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));
    D3D12_SUBRESOURCE_DATA sub{};
    sub.pData = mapped; // mapped is nullptr now; but UpdateSubresources copies from CPU pointer argument; instead use CopyTextureRegion with placed footprint
    // Better: use CopyTextureRegion from upload heap footprint
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
    UINT rows; UINT64 rowSize, total;
    D3D12_RESOURCE_DESC td = browserTex->GetDesc();
    device->GetCopyableFootprints(&td, 0, 1, 0, &fp, &rows, &rowSize, &total);

    D3D12_TEXTURE_COPY_LOCATION dst{ browserTex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX };
    dst.SubresourceIndex = 0;
    D3D12_TEXTURE_COPY_LOCATION src{ uploadHeap.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT };
    src.PlacedFootprint = fp;

    cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);

    CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &toSRV);
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);
    WaitGPU();
}

// Copy from D3D11 shared texture into wrappedBrowserTex (GPU path)
void DX12Context::CopyFromD3D11Shared(ID3D11Texture2D* srcTex) {
    std::lock_guard<std::mutex> lock(mtx);
    // Acquire wrapped resource, copy, release, flush
    ID3D11Resource* wrapped = wrappedBrowserTex.Get();
    d3d11on12->AcquireWrappedResources(&wrapped, 1);
    d3d11ctx->CopyResource(wrapped, srcTex);
    d3d11on12->ReleaseWrappedResources(&wrapped, 1);
    d3d11ctx->Flush();

    // Ensure DX12 resource is ready as SRV
    ThrowIfFailed(alloc[frameIndex]->Reset());
    ThrowIfFailed(cmdList->Reset(alloc[frameIndex].Get(), nullptr));
    CD3DX12_RESOURCE_BARRIER toSRV = CD3DX12_RESOURCE_BARRIER::Transition(browserTex.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    cmdList->ResourceBarrier(1, &toSRV);
    ThrowIfFailed(cmdList->Close());
    ID3D12CommandList* lists[] = { cmdList.Get() };
    queue->ExecuteCommandLists(1, lists);
    WaitGPU();
}
