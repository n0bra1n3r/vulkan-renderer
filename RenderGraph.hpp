// RenderGraph.hpp
//
// - Encapsulates acquire -> record -> submit -> present flow
// - Manages per-frame semaphores and fences
// - Demonstrates image layout transitions using synchronization2 (pipelineBarrier2 / ImageMemoryBarrier2)
// - Provides a minimal "pass" API: each pass supplies a record callback that is called with the per-frame command buffer

#pragma once

#include <functional>

#include <vulkan/vulkan_raii.hpp>

struct RenderPassNode
{
    // Human-readable name (for students / debugging)
    std::string name;

    // Record callback: receives RAII command buffer and acquired image index
    // The callback must record commands into the provided command buffer.
    using RecordFunc = std::function<void(const vk::raii::CommandBuffer&, uint32_t)>;
    RecordFunc recordFunc;

    // Simple image-layout transition requirements for the primary color attachment used by the pass.
    // If no transition is needed, set oldLayout == newLayout.
    vk::ImageLayout oldLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout newLayout = vk::ImageLayout::eUndefined;

    // Access & stage masks for the barrier that moves image from oldLayout->newLayout
    vk::AccessFlags2 srcAccessMask = {};
    vk::AccessFlags2 dstAccessMask = {};
    vk::PipelineStageFlags2 srcStageMask = {};
    vk::PipelineStageFlags2 dstStageMask = {};
};

class RenderGraph
{
public:
    RenderGraph(vk::raii::Device& device,
                vk::raii::SwapchainKHR& swapchain,
                vk::raii::Queue& graphicsQueue,
                vk::raii::Queue& presentQueue,
                vk::raii::CommandPool& commandPool);

    // Add a render pass node. Nodes are executed in the order they are added.
    void addPass(RenderPassNode const& node);

    // Initialize per-frame resources (command buffers, semaphores, fences).
    // Must be called after creating swapchain and image views.
    void init();

    // Execute full frame: acquire, record each pass, submit, present.
    // This implementation uses a single submit of the full set of recorded command buffers
    // and the classic SubmitInfo with semaphores and a fence. Image transitions inside passes
    // use pipelineBarrier2 (ImageMemoryBarrier2 + DependencyInfo).
    void executeFrame();

private:
    vk::raii::Device& m_device;
    vk::raii::SwapchainKHR& m_swapchain;
    vk::raii::Queue& m_graphicsQueue;
    vk::raii::Queue& m_presentQueue;
    vk::raii::CommandPool& m_commandPool;

    std::vector<vk::Image> m_swapchainImages;
    std::vector<RenderPassNode> m_passes;
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;
    std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
    std::vector<vk::raii::Fence> m_inFlightFences;

    uint32_t m_imageCount = 0;
    uint64_t m_currentFrame = 0;
};