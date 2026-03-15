#include "RenderGraph.hpp"

Gfx::RenderGraph::RenderGraph(vk::raii::Device& device, vk::raii::SwapchainKHR& swapchain, vk::raii::Queue& graphicsQueue, vk::raii::Queue& presentQueue, vk::raii::CommandPool& commandPool)
    : m_device(device),
    m_swapchain(swapchain),
    m_graphicsQueue(graphicsQueue),
    m_presentQueue(presentQueue),
    m_commandPool(commandPool)
{
    m_swapchainImages = m_swapchain.getImages();
}

void Gfx::RenderGraph::addPass(Gfx::RenderPassNode const& node)
{
    m_passes.push_back(node);
}

void Gfx::RenderGraph::init()
{
    m_imageCount = static_cast<uint32_t>(m_swapchainImages.size());
    if (m_imageCount == 0) throw std::runtime_error("Swapchain has zero images");

    // allocate one command buffer per swapchain image (common simple approach)
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = m_imageCount;

    // vk::raii::CommandBuffers returns a container of RAII CommandBuffer objects;
    // move them into our vector so we can index per image.
    vk::raii::CommandBuffers tempCmds{ m_device, allocInfo };
    m_commandBuffers.reserve(m_imageCount);
    for (uint32_t i = 0; i < m_imageCount; ++i) {
        m_commandBuffers.emplace_back(std::move(tempCmds[i]));
    }

    // create per-frame semaphores and fences
    m_presentCompleteSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    for (uint32_t i = 0; i < m_imageCount; ++i) {
        m_presentCompleteSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo{});
        m_renderFinishedSemaphores.emplace_back(m_device, vk::SemaphoreCreateInfo{});
        // start signaled so the first wait doesn't block forever if user forgets
        m_inFlightFences.emplace_back(m_device, vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
}

void Gfx::RenderGraph::executeFrame()
{
    // Round-robin frame index for per-frame sync objects
    const uint32_t frameIndex = m_currentFrame % m_imageCount;
    auto& inFlightFence = m_inFlightFences[frameIndex];
    auto& presentComplete = m_presentCompleteSemaphores[frameIndex];
    auto& renderFinished = m_renderFinishedSemaphores[frameIndex];
    auto& cmd = m_commandBuffers[frameIndex];

    // Wait for fence for this frame to be signaled (previous GPU work finished)
    m_device.waitForFences(*inFlightFence, VK_TRUE, UINT64_MAX);

    // Acquire next image
    auto acquireResult = m_swapchain.acquireNextImage(UINT64_MAX, *presentComplete, nullptr);
    uint32_t imageIndex = acquireResult.second;

    // reset command buffer for this image
    cmd.begin(vk::CommandBufferBeginInfo{ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // For each pass, optionally insert an image layout transition, then call the user record callback.
    // We assume all passes render to the swapchain color image directly.
    for (auto const& pass : m_passes)
    {
        // If requested, issue an ImageMemoryBarrier2 via pipelineBarrier2 (synchronization2)
        if (pass.oldLayout != pass.newLayout)
        {
            vk::ImageMemoryBarrier2 barrier{};
            barrier.srcStageMask = pass.srcStageMask;
            barrier.srcAccessMask = pass.srcAccessMask;
            barrier.dstStageMask = pass.dstStageMask;
            barrier.dstAccessMask = pass.dstAccessMask;
            barrier.oldLayout = pass.oldLayout;
            barrier.newLayout = pass.newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_swapchainImages[imageIndex];
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;

            vk::DependencyInfo dependencyInfo{};
            dependencyInfo.imageMemoryBarrierCount = 1;
            dependencyInfo.pImageMemoryBarriers = &barrier;

            cmd.pipelineBarrier2(dependencyInfo);
        }

        // Call the pass record function to record draw/compute commands.
        if (pass.recordFunc) {
            pass.recordFunc(cmd, imageIndex);
        }
    }

    cmd.end();

    m_device.resetFences(*inFlightFence);

    // Submit: wait on presentComplete, signal renderFinished
    vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &*presentComplete;
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*renderFinished;

    m_graphicsQueue.submit(submitInfo, *inFlightFence);

    // Present: wait on renderFinished
    vk::PresentInfoKHR presentInfo{};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &*renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &*m_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    m_presentQueue.presentKHR(presentInfo);

    // Advance frame index
    ++m_currentFrame;
}
