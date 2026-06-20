#include "RenderGraph.hpp"

#include <unordered_map>
#include <unordered_set>

using Gfx::RenderGraph;
using Gfx::GraphicsPipelineBuilder;

void RenderGraph::init()
{
    for (const auto& builder : m_pipelineBuilders)
    {
        auto pipeline = m_rhi.createPipeline(builder.m_pipelineCreateInfo);
        auto descriptorSetsVec = m_rhi.createDescriptorSets({ { pipeline.getDescriptorSetLayout(), builder.m_descriptorBindings } });

        m_renderPasses.emplace_back(GraphicsPass{
            std::move(builder.m_name),
            std::move(pipeline),
            std::move(descriptorSetsVec[0]),
            builder.m_vertexBuffer,
            builder.m_indexBuffer,
            builder.m_drawCommandBuffer,
            std::move(builder.m_colorTargetImages),
            builder.m_depthTargetImage,
            builder.m_usesSwapChainColor,
            builder.m_usesSwapChainDepth,
            std::move(builder.m_shaderReadImages),
        });
    }

    // allocate one command buffer per swapchain image (common simple approach)
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = *m_rhi.getCommandPool();
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = m_rhi.getMaxFramesInFlight();

    // vk::raii::CommandBuffers returns a container of RAII CommandBuffer objects;
    // move them into our vector so we can index per image.
    vk::raii::CommandBuffers tempCmds{ m_rhi.getDevice(), allocInfo };
    m_commandBuffers.reserve(allocInfo.commandBufferCount);
    for (uint32_t i = 0; i < allocInfo.commandBufferCount; ++i)
    {
        m_commandBuffers.emplace_back(std::move(tempCmds[i]));
    }

    // create per-frame semaphores and fences
    m_presentCompleteSemaphores.clear();
    m_renderFinishedSemaphores.clear();
    m_inFlightFences.clear();

    for (uint32_t i = 0; i < allocInfo.commandBufferCount; ++i)
    {
        m_presentCompleteSemaphores.emplace_back(m_rhi.getDevice(), vk::SemaphoreCreateInfo{});
        m_renderFinishedSemaphores.emplace_back(m_rhi.getDevice(), vk::SemaphoreCreateInfo{});
        // start signaled so the first wait doesn't block forever if user forgets
        m_inFlightFences.emplace_back(m_rhi.getDevice(), vk::FenceCreateInfo(vk::FenceCreateFlagBits::eSignaled));
    }
}

void RenderGraph::executeFrame()
{
    // Auto-initialize if not done yet
    if (m_commandBuffers.empty())
    {
        init();
    }

    auto& inFlightFence = m_inFlightFences[m_frameIndex];
    auto& presentComplete = m_presentCompleteSemaphores[m_frameIndex];
    auto& renderFinished = m_renderFinishedSemaphores[m_frameIndex];
    auto& commandBuffer = m_commandBuffers[m_frameIndex];

    // Wait for fence for this frame to be signaled (previous GPU work finished)
    m_rhi.getDevice().waitForFences(*inFlightFence, true, UINT64_MAX);

    m_frameIndex = m_rhi.acquireNextSwapChainImage(*presentComplete).second;

    executeRenderPasses();

    // reset the fence to unsignaled before submit
    m_rhi.getDevice().resetFences(*inFlightFence);

    // Submit: wait on presentComplete, signal renderFinished
    vk::PipelineStageFlags waitStages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &*presentComplete;
    submitInfo.pWaitDstStageMask = &waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*m_commandBuffers[m_frameIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*renderFinished;

    m_rhi.presentSwapChainImage(m_frameIndex, submitInfo, *inFlightFence);
}

// ---------------------------------------------------------------------------
// Reender pass execution with automatic image layout transitions
// ---------------------------------------------------------------------------

void RenderGraph::executeRenderPasses()
{
    // ---- Per-frame image layout state tracking ----
    // Only images that appear as render targets in ANY pass are tracked.
    // Static textures (shader-read only, never a render target) are assumed to already
    // be in eShaderReadOnlyOptimal and are not transitioned.
    std::unordered_map<const Image*, vk::ImageLayout> imageLayouts;
    vk::ImageLayout swapChainColorLayout = vk::ImageLayout::eUndefined;
    vk::ImageLayout swapChainDepthLayout = vk::ImageLayout::eUndefined;

    // Which render targets have been cleared this frame (determines loadOp)
    std::unordered_set<const Image*> clearedImages;
    bool swapChainColorCleared = false;
    bool swapChainDepthCleared = false;

    // Seed the tracking map with all render-target images across every pass
    for (const auto& renderPass : m_renderPasses)
    {
        for (const auto* image : renderPass.colorTargetImages)
        {
            imageLayouts.try_emplace(image, vk::ImageLayout::eUndefined);
        }

        if (renderPass.depthTargetImage)
        {
            imageLayouts.try_emplace(renderPass.depthTargetImage, vk::ImageLayout::eUndefined);
        }
    }

    auto swapChainExtent = m_rhi.getSwapChainExtent();
    bool anyPassUsedSwapChainColor = false;

    // ---- Helper: populate a barrier based on old/new layout ----
    auto addImageBarrier = [](
        std::vector<vk::ImageMemoryBarrier2>& barriers,
        vk::Image image,
        vk::ImageLayout oldLayout,
        vk::ImageLayout newLayout,
        vk::ImageAspectFlags aspect)
        {
            vk::ImageMemoryBarrier2 barrier{};
            barrier.oldLayout = oldLayout;
            barrier.newLayout = newLayout;
            barrier.image = image;
            barrier.subresourceRange.aspectMask = aspect;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.layerCount = 1;

            // Source access / stage (what produced the previous content)
            switch (oldLayout)
            {
            case vk::ImageLayout::eUndefined:
                barrier.srcAccessMask = {};
                barrier.srcStageMask = (aspect & vk::ImageAspectFlagBits::eDepth)
                    ? (vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests)
                    : vk::PipelineStageFlagBits2::eTopOfPipe;
                break;
            case vk::ImageLayout::eColorAttachmentOptimal:
                barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
                barrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
                break;
            case vk::ImageLayout::eDepthAttachmentOptimal:
                barrier.srcAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
                barrier.srcStageMask = vk::PipelineStageFlagBits2::eLateFragmentTests;
                break;
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                barrier.srcAccessMask = vk::AccessFlagBits2::eShaderRead;
                barrier.srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
                break;
            default:
                break;
            }

            // Destination access / stage (what will consume the image next)
            switch (newLayout)
            {
            case vk::ImageLayout::eColorAttachmentOptimal:
                barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
                if (oldLayout == newLayout) // WAW hazard - also need read for blending
                    barrier.dstAccessMask |= vk::AccessFlagBits2::eColorAttachmentRead;
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
                break;
            case vk::ImageLayout::eDepthAttachmentOptimal:
                barrier.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
                if (oldLayout == newLayout)
                    barrier.dstAccessMask |= vk::AccessFlagBits2::eDepthStencilAttachmentRead;
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests
                    | vk::PipelineStageFlagBits2::eLateFragmentTests;
                break;
            case vk::ImageLayout::eShaderReadOnlyOptimal:
                barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
                break;
            case vk::ImageLayout::ePresentSrcKHR:
                barrier.dstAccessMask = {};
                barrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
                break;
            default:
                break;
            }

            barriers.push_back(barrier);
        };

    const auto& commandBuffer = m_commandBuffers[m_frameIndex];

    commandBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });

    // ================================================================
    // Per-pass loop
    // ================================================================
    for (const auto& renderPass : m_renderPasses)
    {
        std::vector<vk::ImageMemoryBarrier2> barriers;

        // ---- 1. Shader-read transitions (render targets from earlier passes) ----
        for (const auto* img : renderPass.shaderReadImages)
        {
            auto it = imageLayouts.find(img);
            if (it != imageLayouts.end() && it->second != vk::ImageLayout::eShaderReadOnlyOptimal)
            {
                bool isDepth = !!(img->getCreateInfo().usage & vk::ImageUsageFlagBits::eDepthStencilAttachment);
                vk::ImageAspectFlags aspect = isDepth
                    ? vk::ImageAspectFlags(vk::ImageAspectFlagBits::eDepth)
                    : vk::ImageAspectFlags(vk::ImageAspectFlagBits::eColor);
                addImageBarrier(barriers, img->getImages()[m_frameIndex],
                    it->second, vk::ImageLayout::eShaderReadOnlyOptimal, aspect);
                it->second = vk::ImageLayout::eShaderReadOnlyOptimal;
            }
            // Images NOT in the tracking map are static textures - no barrier needed.
        }

        // ---- 2. Color render-target transitions ----
        for (const auto* img : renderPass.colorTargetImages)
        {
            auto& layout = imageLayouts[img];
            addImageBarrier(barriers, img->getImages()[m_frameIndex],
                layout, vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageAspectFlagBits::eColor);
            layout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // ---- 3. Depth render-target transition (user image) ----
        if (renderPass.depthTargetImage)
        {
            auto& layout = imageLayouts[renderPass.depthTargetImage];
            addImageBarrier(barriers, renderPass.depthTargetImage->getImages()[m_frameIndex],
                layout, vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth);
            layout = vk::ImageLayout::eDepthAttachmentOptimal;
        }

        // ---- 4. Swap-chain color transition ----
        if (renderPass.usesSwapChainColor)
        {
            anyPassUsedSwapChainColor = true;
            addImageBarrier(barriers, m_rhi.getSwapChainImages()[m_frameIndex],
                swapChainColorLayout, vk::ImageLayout::eColorAttachmentOptimal,
                vk::ImageAspectFlagBits::eColor);
            swapChainColorLayout = vk::ImageLayout::eColorAttachmentOptimal;
        }

        // ---- 5. Swap-chain depth transition ----
        if (renderPass.usesSwapChainDepth)
        {
            addImageBarrier(barriers, m_rhi.getDepthImages()[m_frameIndex],
                swapChainDepthLayout, vk::ImageLayout::eDepthAttachmentOptimal,
                vk::ImageAspectFlagBits::eDepth);
            swapChainDepthLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        }

        // ---- Issue barriers ----
        if (!barriers.empty())
        {
            vk::DependencyInfo depInfo{};
            depInfo.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
            depInfo.pImageMemoryBarriers = barriers.data();
            commandBuffer.pipelineBarrier2(depInfo);
        }

        // ---- Viewport & scissor ----
        commandBuffer.setViewport(
            0,
            vk::Viewport{
                0.0f,
                0.0f,
                static_cast<float>(swapChainExtent.width),
                static_cast<float>(swapChainExtent.height),
                0.0f,
                1.0f,
            });
        commandBuffer.setScissor(0, vk::Rect2D{ { 0, 0 }, swapChainExtent });

        // ---- Build rendering attachment infos ----
        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        std::vector<vk::RenderingAttachmentInfo> colorAttachmentInfos;

        for (const auto* img : renderPass.colorTargetImages)
        {
            bool firstUse = clearedImages.insert(img).second;
            vk::RenderingAttachmentInfo info{};
            info.imageView = img->getImageView(m_frameIndex);
            info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            info.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
            info.storeOp = vk::AttachmentStoreOp::eStore;
            info.clearValue = clearColor;
            colorAttachmentInfos.push_back(info);
        }

        if (renderPass.usesSwapChainColor)
        {
            bool firstUse = !swapChainColorCleared;
            swapChainColorCleared = true;
            vk::RenderingAttachmentInfo info{};
            info.imageView = m_rhi.getSwapChainImageView(m_frameIndex);
            info.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            info.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
            info.storeOp = vk::AttachmentStoreOp::eStore;
            info.clearValue = clearColor;
            colorAttachmentInfos.push_back(info);
        }

        // Depth attachment (user image OR swap-chain depth, at most one)
        vk::RenderingAttachmentInfo depthAttachmentInfo{};
        bool hasDepth = (renderPass.depthTargetImage != nullptr) || renderPass.usesSwapChainDepth;

        if (hasDepth)
        {
            vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

            if (renderPass.depthTargetImage)
            {
                bool firstUse = clearedImages.insert(renderPass.depthTargetImage).second;
                depthAttachmentInfo.imageView = renderPass.depthTargetImage->getImageView(m_frameIndex);
                depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depthAttachmentInfo.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                depthAttachmentInfo.clearValue = clearDepth;
            }
            else // swap-chain depth
            {
                bool firstUse = !swapChainDepthCleared;
                swapChainDepthCleared = true;
                depthAttachmentInfo.imageView = m_rhi.getDepthImageView(m_frameIndex);
                depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
                depthAttachmentInfo.loadOp = firstUse ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
                depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
                depthAttachmentInfo.clearValue = clearDepth;
            }
        }

        // ---- Begin dynamic rendering ----
        vk::RenderingInfo renderingInfo{};
        renderingInfo.renderArea.extent = swapChainExtent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(colorAttachmentInfos.size());
        renderingInfo.pColorAttachments = colorAttachmentInfos.data();
        if (hasDepth)
            renderingInfo.pDepthAttachment = &depthAttachmentInfo;

        commandBuffer.beginRendering(renderingInfo);

        // ---- Bind pipeline ----
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, renderPass.pipeline);

        // ---- Bind descriptor sets ----
        if (!renderPass.descriptorSets.empty())
        {
            commandBuffer.bindDescriptorSets(
                vk::PipelineBindPoint::eGraphics,
                renderPass.pipeline.getPipelineLayout(),
                0,
                *renderPass.descriptorSets[m_frameIndex],
                nullptr);
        }

        // ---- Draw ----
        if (renderPass.drawCommandBuffer)
        {
            // Indexed indirect draw (mesh geometry)
            if (renderPass.vertexBuffer)
            {
                commandBuffer.bindVertexBuffers(0, renderPass.vertexBuffer->getBuffer(0), { vk::DeviceSize(0) });
            }

            if (renderPass.indexBuffer)
            {
                commandBuffer.bindIndexBuffer(renderPass.indexBuffer->getBuffer(0), 0, vk::IndexType::eUint32);
            }

            auto drawCount = static_cast<uint32_t>(renderPass.drawCommandBuffer->getCreateInfo().size / sizeof(vk::DrawIndexedIndirectCommand));
            commandBuffer.drawIndexedIndirect(
                renderPass.drawCommandBuffer->getBuffer(0),
                0,
                drawCount,
                static_cast<uint32_t>(sizeof(vk::DrawIndexedIndirectCommand)));
        }
        else
        {
            // Full-screen triangle (no vertex data needed)
            commandBuffer.draw(3, 1, 0, 0);
        }

        // ---- End rendering ----
        commandBuffer.endRendering();
    }

    // ---- Final transition: swap-chain color -> present ----
    if (anyPassUsedSwapChainColor)
    {
        vk::ImageMemoryBarrier2 presentBarrier{};
        presentBarrier.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        presentBarrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        presentBarrier.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        presentBarrier.dstAccessMask = {};
        presentBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        presentBarrier.newLayout = vk::ImageLayout::ePresentSrcKHR;
        presentBarrier.image = m_rhi.getSwapChainImages()[m_frameIndex];
        presentBarrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
        presentBarrier.subresourceRange.levelCount = 1;
        presentBarrier.subresourceRange.layerCount = 1;

        vk::DependencyInfo depInfo{};
        depInfo.imageMemoryBarrierCount = 1;
        depInfo.pImageMemoryBarriers = &presentBarrier;
        commandBuffer.pipelineBarrier2(depInfo);
    }

    commandBuffer.end();
}
