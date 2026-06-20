// RenderGraph.hpp
//
// - Encapsulates acquire -> record -> submit -> present flow
// - Manages per-frame semaphores and fences
// - Demonstrates image layout transitions using synchronization2 (pipelineBarrier2 / ImageMemoryBarrier2)
// - Provides a minimal "pass" API: each pass supplies a record callback that is called with the per-frame command buffer
// - Provides a higher-level GraphicsPassBuilder API that automates pipeline creation, descriptor set management,
//   rendering setup, and image layout transitions
//
// Usage (new pass-based API):
//   RenderGraph graph(rhi);
//   graph.graphicsPass("GBuffer")
//       .vertexShader("Shaders/gbuffer.vert.spv")
//       .fragmentShader("Shaders/gbuffer.frag.spv")
//       .vertexBuffer<Vertex>(vertexBuffer)
//       .indexBuffer(indexBuffer)
//       .drawCommandBuffer(drawCommandBuffer)
//       .allShadersBinding(uniformBuffer)
//       .renderTarget(gbufferAlbedoImage)
//       .renderTargetSwapChainDepth()
//       .build();
//   // each frame:
//     graph.executeFrame();

#pragma once

#include <functional>

#include "RHI.hpp"
#include "Pipeline.hpp"
#include "DescriptorSet.hpp"
#include "Image.hpp"
#include "Buffer.hpp"

namespace Gfx
{
    class RenderGraph;

    template<typename PipelineCreateInfo>
    class PipelineBuilder
    {
    protected:
        PipelineBuilder(const std::string& name) : m_name(name) {}

        PipelineCreateInfo m_pipelineCreateInfo;
        std::vector<DescriptorBinding> m_descriptorBindings;

        std::string m_name;
    };

    class ComputePipelineBuilder : public PipelineBuilder<ComputePipelineCreateInfo>
    {
    private:
        friend class RenderGraph;

        ComputePipelineBuilder(const std::string& name, uint32_t minDispatchThreadCount) :
            PipelineBuilder(name),
            m_minDispatchThreadCount(minDispatchThreadCount)
        {}

    public:
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
                for (int i = 0; i < buffer.getBufferCount(); i++)
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
                resourceInfos = { {
                    buffer.getBuffer(0),
                    0,
                    createInfo.size,
                } };
            }

            descriptorBinding.type = descriptorType;
            descriptorBinding.data = resourceInfos;

            m_descriptorBindings.emplace_back(std::move(descriptorBinding));

            return *this;
        }

    private:
        uint32_t m_minDispatchThreadCount;
    };

    class GraphicsPipelineBuilder : public PipelineBuilder<GraphicsPipelineCreateInfo>
    {
    private:
        friend class RenderGraph;

        GraphicsPipelineBuilder(const std::string& name, vk::Format swapChainColorFormat, vk::Format swapChainDepthFormat) :
            PipelineBuilder(name),
            m_swapChainColorFormat(swapChainColorFormat),
            m_swapChainDepthFormat(swapChainDepthFormat)
        {}

    public:
        GraphicsPipelineBuilder& vertexShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eVertex);
            return *this;
        }

        template<typename T>
        GraphicsPipelineBuilder& vertexBuffer(const Buffer& buffer)
        {
            m_pipelineCreateInfo.vertexInputBinding = T::getBindingDescription();
            m_pipelineCreateInfo.vertexInputAttributes = T::getAttributeDescriptions();
            m_vertexBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& indexBuffer(const Buffer& buffer)
        {
            m_indexBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& drawCommandBuffer(const Buffer& buffer)
        {
            m_drawCommandBuffer = std::move(buffer.getInfo());
            return *this;
        }

        GraphicsPipelineBuilder& fragmentShader(std::string name)
        {
            m_pipelineCreateInfo.shaders.emplace_back(name, vk::ShaderStageFlagBits::eFragment);
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
                m_colorTargetImages.emplace_back(std::move(image.getInfo()));
            }
            else
            {
                m_pipelineCreateInfo.depthAttachment = createInfo.format;
                m_depthTargetImage = std::move(image.getInfo());
            }
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainColor()
        {
            m_pipelineCreateInfo.colorAttachments.emplace_back(m_swapChainColorFormat);
            m_usesSwapChainColor = true;
            return *this;
        }

        GraphicsPipelineBuilder& renderTargetSwapChainDepth()
        {
            m_pipelineCreateInfo.depthAttachment = m_swapChainDepthFormat;
            m_usesSwapChainDepth = true;
            return *this;
        }

    private:
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

                m_shaderReadImages.emplace_back(std::move(image.getInfo()));

                for (int i = 0; i < image.getImageCount(); i++)
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
                    m_shaderReadImages.emplace_back(std::move(image.getInfo()));

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

            m_descriptorBindings.emplace_back(std::move(descriptorBinding));

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
                for (int i = 0; i < buffer->getBufferCount(); i++)
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
                resourceInfos = { {
                    buffer->getBuffer(0),
                    0,
                    createInfo.size,
                } };
            }

            descriptorBinding.type = descriptorType;
            descriptorBinding.data = resourceInfos;

            m_descriptorBindings.emplace_back(std::move(descriptorBinding));

            return *this;
        }

    private:
        vk::Format m_swapChainColorFormat;
        vk::Format m_swapChainDepthFormat;

        BufferInfo m_vertexBuffer{};
        BufferInfo m_indexBuffer{};
        BufferInfo m_drawCommandBuffer{};

        std::vector<ImageInfo> m_colorTargetImages{};
        ImageInfo m_depthTargetImage{};
        bool m_usesSwapChainColor = false;
        bool m_usesSwapChainDepth = false;

        std::vector<ImageInfo> m_shaderReadImages{};
    };

    // ---- Render pass data stored by the render graph ----

    struct ComputePass
    {
        std::string name;
        uint32_t minDispatchThreadCount;
        Pipeline pipeline;
        std::vector<DescriptorSet> descriptorSets;
    };

    struct GraphicsPass
    {
        std::string name;
        Pipeline pipeline;
        std::vector<DescriptorSet> descriptorSets;
        BufferInfo vertexBuffer;
        BufferInfo indexBuffer;
        BufferInfo drawCommandBuffer;
        std::vector<ImageInfo> colorTargetImages;
        ImageInfo depthTargetImage;
        bool usesSwapChainColor;
        bool usesSwapChainDepth;
        std::vector<ImageInfo> shaderReadImages;
    };

    class RenderGraph
    {
    public:
        RenderGraph(RHI& rhi) : m_rhi(rhi) {}
        RenderGraph(const RenderGraph&) = delete;

        ComputePipelineBuilder& computePass(const std::string& name, uint32_t minDispatchThreadCount)
        {
            m_pipelineBuilders.emplace_back(ComputePipelineBuilder(name, minDispatchThreadCount));
            return std::get<ComputePipelineBuilder>(m_pipelineBuilders.back());
        }

        GraphicsPipelineBuilder& graphicsPass(const std::string& name)
        {
            m_pipelineBuilders.emplace_back(GraphicsPipelineBuilder(name, m_rhi.getSurfaceFormat(), m_rhi.getDepthFormat()));
            return std::get<GraphicsPipelineBuilder>(m_pipelineBuilders.back());
        }

        // Initialize per-frame resources (command buffers, semaphores, fences).
        // Must be called after creating swapchain and image views.
        void init();

        // Execute full frame: acquire, record each pass, submit, present.
        // When compiled passes are registered (via graphicsPass().build()), they are executed
        // with automatic image layout transitions. Otherwise, legacy RenderPassNode passes are used.
        void executeFrame();

        uint64_t getFrameIndex() const { return m_imageIndex; }

    private:
        void executeRenderPasses();

    private:
        RHI& m_rhi;

        std::vector<std::variant<ComputePass, GraphicsPass>> m_renderPasses;

        std::vector<std::variant<ComputePipelineBuilder, GraphicsPipelineBuilder>> m_pipelineBuilders;

        std::vector<vk::raii::CommandBuffer> m_commandBuffers;

        // per-frame synchronization objects
        std::vector<vk::raii::Semaphore> m_presentCompleteSemaphores;
        std::vector<vk::raii::Semaphore> m_renderFinishedSemaphores;
        std::vector<vk::raii::Fence> m_inFlightFences;

        int m_imageIndex = 0;
    };
}
