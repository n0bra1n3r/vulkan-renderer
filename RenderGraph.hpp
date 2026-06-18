// RenderGraph.hpp
//
// - Encapsulates acquire -> record -> submit -> present flow
// - Manages per-frame semaphores and fences
// - Demonstrates image layout transitions using synchronization2 (pipelineBarrier2 / ImageMemoryBarrier2)
// - Provides a minimal "pass" API: each pass supplies a record callback that is called with the per-frame command buffer
//
// Usage sketch:
//   RenderGraph rg(device, swapChain, graphicsQueue, presentQueue, commandPool, swapChainImageViews, swapChainExtent);
//   rg.addPass("GeometryPass", [](vk::raii::CommandBuffer &cmd, uint32_t imageIndex){ /* record drawing */ }, 
//              /*transition*/ vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
//              /*src/dst access/stage*/ {}, vk::AccessFlagBits2::eColorAttachmentWrite,
//              vk::PipelineStageFlagBits2::eTopOfPipe, vk::PipelineStageFlagBits2::eColorAttachmentOutput);
//   rg.init(); // allocates command buffers and sync objects
//   // each frame:
//     rg.executeFrame();
//
//
// Note: this code focuses on synchronization and orchestration. It intentionally avoids higher-level resource/lifetime
// management (frame graphs with automatic aliasing, barriers across many resources, queue ownership transfers, etc.).

#pragma once

#include <functional>

#include "RHI.hpp"
#include "Pipeline.hpp"
#include "Image.hpp"
#include "Buffer.hpp"

namespace Gfx
{
    struct GraphicsPipelineBuilder
    {
        GraphicsPipelineBuilder(RHI& rhi) : m_rhi(rhi) {}

        GraphicsPipelineBuilder& vertexShader(std::string name)
        {
            pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eVertex);
            return *this;
        }

        template<typename T>
        GraphicsPipelineBuilder& vertexType()
        {
            pipelineCreateInfo.vertexInputBinding = T::getBindingDescription();
            pipelineCreateInfo.vertexInputAttributes = T::getAttributeDescriptions();
            return *this;
        }

        GraphicsPipelineBuilder& fragmentShader(std::string name)
        {
            pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eFragment);
            return *this;
        }

        GraphicsPipelineBuilder shaderBinding(const Image& image, vk::ShaderStageFlagBits stage, bool hasSampler = true)
        {
            auto index = static_cast<uint32_t>(pipelineCreateInfo.descriptorSetLayoutBindings.size());
            auto descriptorType =
                hasSampler ?
                vk::DescriptorType::eCombinedImageSampler :
                vk::DescriptorType::eSampledImage;
            pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                1,
                stage,
                nullptr);
            return *this;
        }

        GraphicsPipelineBuilder shaderBinding(const std::vector<Image>& images, vk::ShaderStageFlagBits stage, bool hasSampler = true)
        {
            auto index = static_cast<uint32_t>(pipelineCreateInfo.descriptorSetLayoutBindings.size());
            auto descriptorType =
                hasSampler ? 
                vk::DescriptorType::eCombinedImageSampler : 
                vk::DescriptorType::eSampledImage;
            pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                static_cast<uint32_t>(images.size()),
                stage, 
                nullptr);
            return *this;
        }

        GraphicsPipelineBuilder shaderBinding(const Buffer& buffer, vk::ShaderStageFlagBits stage)
        {
            auto index = static_cast<uint32_t>(pipelineCreateInfo.descriptorSetLayoutBindings.size());
            const auto& createInfo = buffer.getCreateInfo();
            auto descriptorType =
                (createInfo.usage & vk::BufferUsageFlagBits::eStorageBuffer) ?
                vk::DescriptorType::eStorageBuffer :
                vk::DescriptorType::eUniformBuffer;
            pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                1,
                stage,
                nullptr);
            return *this;
        }

        GraphicsPipelineBuilder vertexShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(buffer, vk::ShaderStageFlagBits::eVertex);
        }

        GraphicsPipelineBuilder fragmentShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(buffer, vk::ShaderStageFlagBits::eFragment);
        }

        GraphicsPipelineBuilder fragmentShaderBinding(const Image& image, bool hasSampler = true)
        {
            return shaderBinding(image, vk::ShaderStageFlagBits::eFragment, hasSampler);
        }

        GraphicsPipelineBuilder fragmentShaderBinding(const std::vector<Image>& images, bool hasSampler = true)
        {
            return shaderBinding(images, vk::ShaderStageFlagBits::eFragment, hasSampler);
        }

        GraphicsPipelineBuilder allShadersBinding(const Buffer& buffer)
        {
            return shaderBinding(buffer, vk::ShaderStageFlagBits::eAllGraphics);
        }

        GraphicsPipelineBuilder renderTarget(const Image& image)
        {
            const auto& createInfo = image.getCreateInfo();
 
            if (createInfo.usage & vk::ImageUsageFlagBits::eColorAttachment)
            {
                pipelineCreateInfo.colorAttachments.emplace_back(createInfo.format);
            }
            else
            {
                pipelineCreateInfo.depthAttachment = createInfo.format;
            }
            return *this;
        }

        GraphicsPipelineBuilder renderTargetSwapChainColor()
        {
            pipelineCreateInfo.colorAttachments.emplace_back(m_rhi.getSurfaceFormat());
            return *this;
        }

        GraphicsPipelineBuilder renderTargetSwapChainDepth()
        {
            pipelineCreateInfo.depthAttachment = m_rhi.getDepthFormat();
            return *this;
        }

        Pipeline build()
        {
            return m_rhi.createGraphicsPipeline(pipelineCreateInfo);
        }

    private:
        RHI& m_rhi;
        GraphicsPipelineCreateInfo pipelineCreateInfo;
    };

    struct RenderPassNode
    {
        // Human-readable name (for students / debugging)
        std::string name;

        // Record callback: receives RAII command buffer and acquired image index
        // The callback must record commands into the provided command buffer.
        using RecordFunc = std::function<void(vk::raii::CommandBuffer&, uint32_t)>;
        RecordFunc recordFunc;

        struct AttachmentTransitionInfo
        {
            std::vector<vk::Image> images; // images to transition (e.g. swapchain image for color, depth image for depth)
            vk::ImageAspectFlagBits aspectMask; // aspect of the image to transition (e.g. color, depth, stencil)

            // Simple image-layout transition requirements for the attachments used by the pass.
            // If no transition is needed, set oldLayout == newLayout.
            vk::ImageLayout oldLayout = vk::ImageLayout::eUndefined;
            vk::ImageLayout newLayout = vk::ImageLayout::eUndefined;

            // Access & stage masks for the barrier that moves image from oldLayout->newLayout
            vk::AccessFlags2 srcAccessMask = {};
            vk::AccessFlags2 dstAccessMask = {};
            vk::PipelineStageFlags2 srcStageMask = {};
            vk::PipelineStageFlags2 dstStageMask = {};
        };

        struct BufferTransitionInfo
        {
            std::vector<vk::Buffer> buffers; // buffers to transition (e.g. uniform buffer, storage buffer)

            vk::AccessFlags2 srcAccessMask = {};
            vk::AccessFlags2 dstAccessMask = {};
            vk::PipelineStageFlags2 srcStageMask = {};
            vk::PipelineStageFlags2 dstStageMask = {};
		};

        std::vector<AttachmentTransitionInfo> attachmentInfos;
        std::vector<BufferTransitionInfo> bufferInfos;
    };

    class RenderGraph
    {
    public:
        // Construct with references to objects managed elsewhere (HelloTriangleApplication keeps lifetime)
        RenderGraph(const RHI& rhi);
        RenderGraph(const RenderGraph&) = delete;

        // Add a render pass node. Nodes are executed in the order they are added.
        void addPass(const RenderPassNode& node) { m_passes.push_back(node); }

        // Initialize per-frame resources (command buffers, semaphores, fences).
        // Must be called after creating swapchain and image views.
        void init();

        // Execute full frame: acquire, record each pass, submit, present.
        // This implementation uses a single submit of the full set of recorded command buffers
        // and the classic SubmitInfo with semaphores and a fence. Image transitions inside passes
        // use pipelineBarrier2 (ImageMemoryBarrier2 + DependencyInfo).
        void executeFrame();

        GraphicsPipelineBuilder buildGraphicsPipeline(RHI& rhi)
        {
            return GraphicsPipelineBuilder(rhi);
        }

    private:
		const RHI& m_rhi;

        // recorded passes
        std::vector<RenderPassNode> m_passes;

        // per-swapchain-image command buffers (RAII)
        std::vector<vk::raii::CommandBuffer> m_commandBuffers;

        // per-frame synchronization objects
        std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::raii::Fence> m_inFlightFences;

        uint64_t m_currentFrame = 0;
    };
}