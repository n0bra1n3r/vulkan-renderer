#include "RenderGraph.hpp"

Gfx::RenderGraph::RenderGraph(
    vk::raii::Device& device, 
    vk::raii::SwapchainKHR& swapchain,
    std::vector<vk::raii::Image>& swapchainDepthImages,
    vk::raii::Queue& graphicsQueue, 
    vk::raii::Queue& presentQueue, 
    vk::raii::CommandPool& commandPool)
    : m_device(device),
      m_swapchain(swapchain),
      m_graphicsQueue(graphicsQueue),
      m_presentQueue(presentQueue),
      m_commandPool(commandPool)
{
	m_swapchainColorImages = m_swapchain.getImages();

    for (auto& depthImage : swapchainDepthImages) {
        m_swapchainDepthImages.push_back(*depthImage);
	}
}

void Gfx::RenderGraph::addPass(Gfx::RenderPassNode const& node)
{
    m_passes.push_back(node);
}

void Gfx::RenderGraph::init()
{
    m_imageCount = static_cast<uint32_t>(m_swapchainColorImages.size());
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
    m_device.waitForFences(*inFlightFence, true, UINT64_MAX);

    // Acquire next image
    auto acquireResult = m_swapchain.acquireNextImage(UINT64_MAX, *presentComplete, nullptr);
    auto imageIndex = acquireResult.second;

    // reset command buffer for this image
    cmd.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

	std::vector<vk::ImageMemoryBarrier2> barriers{};
	barriers.reserve(2);

    // For each pass, optionally insert an image layout transition, then call the user record callback.
    // We assume all passes render to the swapchain color image directly.
    for (auto const& pass : m_passes)
    {
        // If requested, issue an ImageMemoryBarrier2 via pipelineBarrier2 (synchronization2)
        if (pass.colorTransitionInfo.oldLayout != pass.colorTransitionInfo.newLayout) {
			vk::ImageMemoryBarrier2 barrier{};
            barrier.srcStageMask = pass.colorTransitionInfo.srcStageMask;
            barrier.srcAccessMask = pass.colorTransitionInfo.srcAccessMask;
            barrier.dstStageMask = pass.colorTransitionInfo.dstStageMask;
            barrier.dstAccessMask = pass.colorTransitionInfo.dstAccessMask;
            barrier.oldLayout = pass.colorTransitionInfo.oldLayout;
            barrier.newLayout = pass.colorTransitionInfo.newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_swapchainColorImages[imageIndex];
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
			barriers.emplace_back(std::move(barrier));
        }

        if (pass.depthTransitionInfo.oldLayout != pass.depthTransitionInfo.newLayout) {
            vk::ImageMemoryBarrier2 barrier{};
            barrier.srcStageMask = pass.depthTransitionInfo.srcStageMask;
            barrier.srcAccessMask = pass.depthTransitionInfo.srcAccessMask;
            barrier.dstStageMask = pass.depthTransitionInfo.dstStageMask;
            barrier.dstAccessMask = pass.depthTransitionInfo.dstAccessMask;
            barrier.oldLayout = pass.depthTransitionInfo.oldLayout;
            barrier.newLayout = pass.depthTransitionInfo.newLayout;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = m_swapchainDepthImages[imageIndex];
            barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eDepth;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;
            barriers.emplace_back(std::move(barrier));
        }

        if (barriers.size()) {
            vk::DependencyInfo dependencyInfo{};
            dependencyInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            dependencyInfo.pImageMemoryBarriers = barriers.data();
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
