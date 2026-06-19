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
    template<typename PipelineCreateInfo>
    class PipelineBuilder
    {
    public:
        PipelineBuilder(RHI& rhi) : m_rhi(rhi) {}

        Pipeline build()
        {
            auto pipeline = m_rhi.createPipeline(m_pipelineCreateInfo);
            m_descriptorSetConfig.layout = pipeline.getDescriptorSetLayout();
            return std::move(pipeline);
        }

        const DescriptorSetConfig& getDescriptorSetConfig() const
        {
            return m_descriptorSetConfig;
        }

    protected:
        RHI& m_rhi;
        PipelineCreateInfo m_pipelineCreateInfo;
        DescriptorSetConfig m_descriptorSetConfig;
    };

    class ComputePipelineBuilder : public PipelineBuilder<ComputePipelineCreateInfo>
    {
    public:
        using PipelineBuilder::PipelineBuilder;

        ComputePipelineBuilder& shader(std::string name)
        {
            m_pipelineCreateInfo.shader = name;
            return *this;
        }

        ComputePipelineBuilder& shaderBinding(const Buffer& buffer)
        {
            auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());
            const auto& createInfo = buffer.getCreateInfo();

            auto descriptorType =
                (createInfo.usage & vk::BufferUsageFlagBits::eStorageBuffer) ?
                vk::DescriptorType::eStorageBuffer :
                vk::DescriptorType::eUniformBuffer;

            m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                1,
                vk::ShaderStageFlagBits::eCompute,
                nullptr);

            DescriptorBinding descriptorBinding{};

            std::vector<vk::DescriptorBufferInfo> resourceInfos{};

            if (descriptorType == vk::DescriptorType::eUniformBuffer)
            {
                for (size_t i = 0; i < buffer.getBufferCount(); i++)
                {
                    vk::DescriptorBufferInfo resourceInfo = {
                        buffer.getBuffer(i),
                        0,
                        createInfo.size,
                    };
                    resourceInfos.emplace_back(std::move(resourceInfo));
                }
            }
            else
            {
                resourceInfos = {{
                    buffer.getBuffer(0),
                    0,
                    createInfo.size,
                }};
            }

            descriptorBinding.type = descriptorType;
            descriptorBinding.data = resourceInfos;

            m_descriptorSetConfig.bindings.emplace_back(std::move(descriptorBinding));

            return *this;
        }
    };

    class GraphicsPipelineBuilder : public PipelineBuilder<GraphicsPipelineCreateInfo>
    {
    public:
        using PipelineBuilder::PipelineBuilder;

        GraphicsPipelineBuilder& vertexShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eVertex);
            return *this;
        }

        template<typename T>
        GraphicsPipelineBuilder& vertexType()
        {
            m_pipelineCreateInfo.vertexInputBinding = T::getBindingDescription();
            m_pipelineCreateInfo.vertexInputAttributes = T::getAttributeDescriptions();
            return *this;
        }

        GraphicsPipelineBuilder& fragmentShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eFragment);
            return *this;
        }

        GraphicsPipelineBuilder& shaderBinding(std::variant<const std::vector<Image>*, const Image*> images, vk::ShaderStageFlagBits stage, const Sampler& sampler = nullptr)
        {
            auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());

            auto descriptorType =
                sampler.getSampler() != nullptr ? 
                vk::DescriptorType::eCombinedImageSampler : 
                vk::DescriptorType::eSampledImage;
            auto descriptorCount = 
                std::holds_alternative<const std::vector<Image>*>(images) ? 
                std::get<const std::vector<Image>*>(images)->size() : 
                1;

            m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                static_cast<uint32_t>(descriptorCount),
                stage, 
                nullptr);

            DescriptorBinding descriptorBinding{};

            std::vector<std::vector<vk::DescriptorImageInfo>> resourceInfos{};

            if (descriptorCount == 1)
            {
                const auto& image = *std::get<const Image*>(images);

                for (size_t i = 0; i < image.getImages().size(); i++)
                {
                    vk::DescriptorImageInfo resourceInfo = {
                        sampler.getSampler(),
                        image.getImageView(i),
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                    };
                    resourceInfos.emplace_back(std::vector<vk::DescriptorImageInfo>{ std::move(resourceInfo) });
                }
            }
            else
            {
                resourceInfos.resize(1);

                const auto& imageArray = *std::get<const std::vector<Image>*>(images);

                for (const auto& image : imageArray)
                {
                    vk::DescriptorImageInfo resourceInfo = {
                        sampler.getSampler(),
                        image.getImageView(0),
                        vk::ImageLayout::eShaderReadOnlyOptimal,
                    };

                    resourceInfos[0].emplace_back(std::move(resourceInfo));
                }
            }

            descriptorBinding.type = descriptorType;
            descriptorBinding.data = resourceInfos;

            m_descriptorSetConfig.bindings.emplace_back(std::move(descriptorBinding));

            return *this;
        }

        GraphicsPipelineBuilder& shaderBinding(const Buffer* buffer, vk::ShaderStageFlagBits stage)
        {
            auto index = static_cast<uint32_t>(m_pipelineCreateInfo.descriptorSetLayoutBindings.size());
            const auto& createInfo = buffer->getCreateInfo();

            auto descriptorType =
                (createInfo.usage & vk::BufferUsageFlagBits::eStorageBuffer) ?
                vk::DescriptorType::eStorageBuffer :
                vk::DescriptorType::eUniformBuffer;

            m_pipelineCreateInfo.descriptorSetLayoutBindings.emplace_back(
                index,
                descriptorType,
                1,
                stage,
                nullptr);

            DescriptorBinding descriptorBinding{};

            std::vector<vk::DescriptorBufferInfo> resourceInfos{};

            if (descriptorType == vk::DescriptorType::eUniformBuffer)
            {
                for (size_t i = 0; i < buffer->getBufferCount(); i++)
                {
                    vk::DescriptorBufferInfo resourceInfo = { 
                        buffer->getBuffer(i),
                        0,
                        createInfo.size,
                    };
                    resourceInfos.emplace_back(std::move(resourceInfo));
                }
            }
            else
            {
                resourceInfos = {{
                    buffer->getBuffer(0),
                    0,
                    createInfo.size,
                }};
            }

            descriptorBinding.type = descriptorType;
            descriptorBinding.data = resourceInfos;

            m_descriptorSetConfig.bindings.emplace_back(std::move(descriptorBinding));

            return *this;
        }

        GraphicsPipelineBuilder& vertexShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eVertex);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eFragment);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const Image& image, const Sampler& sampler = nullptr)
        {
            return shaderBinding(&image, vk::ShaderStageFlagBits::eFragment, sampler);
        }

        GraphicsPipelineBuilder& fragmentShaderBinding(const std::vector<Image>& images, const Sampler& sampler = nullptr)
        {
            return shaderBinding(&images, vk::ShaderStageFlagBits::eFragment, sampler);
        }

        GraphicsPipelineBuilder& allShadersBinding(const Buffer& buffer)
        {
            return shaderBinding(&buffer, vk::ShaderStageFlagBits::eAllGraphics);
        }

        GraphicsPipelineBuilder& renderTarget(const Image& image)
        {
            const auto& createInfo = image.getCreateInfo();
 
            if (createInfo.usage & vk::ImageUsageFlagBits::eColorAttachment)
            {
                m_pipelineCreateInfo.colorAttachments.emplace_back(createInfo.format);
            }
            else
            {
                m_pipelineCreateInfo.depthAttachment = createInfo.format;
            }
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainColor()
        {
            m_pipelineCreateInfo.colorAttachments.emplace_back(m_rhi.getSurfaceFormat());
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainDepth()
        {
            m_pipelineCreateInfo.depthAttachment = m_rhi.getDepthFormat();
            return *this;
        }
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
        RenderGraph(RHI& rhi);
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

        ComputePipelineBuilder& computePipeline()
        {
            m_builders.emplace_back(ComputePipelineBuilder(m_rhi));
            return std::get<ComputePipelineBuilder>(m_builders.back());
        }

        GraphicsPipelineBuilder& graphicsPipeline()
        {
            m_builders.emplace_back(GraphicsPipelineBuilder(m_rhi));
            return std::get<GraphicsPipelineBuilder>(m_builders.back());
        }

        template<typename T>
        T buildDescriptorSets()
        {
            std::vector<DescriptorSetConfig> configs;

            for (const auto& builder : m_builders)
            {
                auto config = std::holds_alternative<ComputePipelineBuilder>(builder) ?
                    std::get<ComputePipelineBuilder>(builder).getDescriptorSetConfig() :
                    std::get<GraphicsPipelineBuilder>(builder).getDescriptorSetConfig();

                configs.emplace_back(std::move(config));
            }

            auto descriptorSets = m_rhi.createDescriptorSets(configs);

            T container;
            auto* containerPtr = reinterpret_cast<std::vector<DescriptorSet>*>(&container);
            for (auto& descriptorSetArray : descriptorSets)
            {
                ::new (static_cast<void*>(containerPtr)) std::vector<DescriptorSet>(std::move(descriptorSetArray));
                containerPtr += 1;
            }
            return std::move(container);
        }

    private:
		RHI& m_rhi;

        // recorded passes
        std::vector<RenderPassNode> m_passes;

        // per-swapchain-image command buffers (RAII)
        std::vector<vk::raii::CommandBuffer> m_commandBuffers;

        // per-frame synchronization objects
        std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::raii::Fence> m_inFlightFences;

        uint64_t m_currentFrame = 0;

        std::vector<std::variant<ComputePipelineBuilder, GraphicsPipelineBuilder>> m_builders;
    };
}