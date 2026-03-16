#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

#include "Api.hpp"
#include "Buffer.hpp"
#include "Image.hpp"
#include "RenderGraph.hpp"

#include <vulkan/vulkan_raii.hpp>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct Vertex
{
    glm::vec3 position;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }
};

struct Shape
{
    vk::DrawIndexedIndirectCommand drawCmd;
};

struct Cube : public Shape
{
    static std::vector<Vertex> getVertices() {
        return {
            // Front face (z = +0.5)
            {{-0.5f, -0.5f,  0.5f}, {1.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 1.0f}},
            {{-0.5f,  0.5f,  0.5f}, {1.0f, 1.0f}},
            // Back face (z = -0.5)
            {{ 0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}},
            {{-0.5f,  0.5f, -0.5f}, {0.0f, 1.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {1.0f, 1.0f}},
            // Left face (x = -0.5)
            {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
            {{-0.5f, -0.5f,  0.5f}, {0.0f, 0.0f}},
            {{-0.5f,  0.5f,  0.5f}, {0.0f, 1.0f}},
            {{-0.5f,  0.5f, -0.5f}, {1.0f, 1.0f}},
            // Right face (x = +0.5)
            {{ 0.5f, -0.5f,  0.5f}, {1.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {1.0f, 1.0f}},
            // Top face (y = +0.5)
            {{-0.5f,  0.5f,  0.5f}, {1.0f, 0.0f}},
            {{ 0.5f,  0.5f,  0.5f}, {0.0f, 0.0f}},
            {{ 0.5f,  0.5f, -0.5f}, {0.0f, 1.0f}},
            {{-0.5f,  0.5f, -0.5f}, {1.0f, 1.0f}},
            // Bottom face (y = -0.5)
            {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f}},
            {{ 0.5f, -0.5f, -0.5f}, {0.0f, 0.0f}},
            {{ 0.5f, -0.5f,  0.5f}, {0.0f, 1.0f}},
            {{-0.5f, -0.5f,  0.5f}, {1.0f, 1.0f}},
        };
    }

    static std::vector<uint32_t> getIndices() {
        return {
            // Front
            0, 1, 2, 2, 3, 0,
            // Back
            4, 5, 6, 6, 7, 4,
            // Left
            8, 9, 10, 10, 11, 8,
            // Right
            12, 13, 14, 14, 15, 12,
            // Top
            16, 17, 18, 18, 19, 16,
            // Bottom
            20, 21, 22, 22, 23, 20
        };
    }
};

struct UniformBufferObject
{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

struct StorageBufferObject
{
    glm::vec3 colour;
};

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* m_window = nullptr;
	Gfx::Api m_gfx;

    vk::raii::DescriptorSetLayout m_descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout m_pipelineLayout = nullptr;
    vk::raii::Pipeline m_graphicsPipeline = nullptr;
    Gfx::Image m_textureImage = nullptr;
    vk::raii::ImageView m_textureImageView = nullptr;
    vk::raii::Sampler m_textureSampler = nullptr;
    Gfx::Buffer m_vertexBuffer = nullptr;
    Gfx::Buffer m_indexBuffer = nullptr;
    Gfx::Buffer m_indirectBuffer = nullptr;
    Gfx::Buffer m_storageBuffer = nullptr;
    std::vector<Gfx::Buffer> m_uniformBuffers;
    std::vector<void*> m_uniformBuffersMapped;
    vk::raii::DescriptorPool m_descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> m_descriptorSets;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        m_window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
		m_gfx.init("Vulkan App", getRequiredExtensions(), glfwGetWin32Window(m_window));

        createDescriptorSetLayout();
        createGraphicsPipeline();
		createTextureImage();
        createTextureImageView();
        createVertexBuffer();
        createIndexBuffer();
        createIndirectBuffer();
        createUniformBuffers();
        createStorageBuffer();
		createDescriptorPool();
        createDescriptorSets();

        initRenderPasses();
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
		return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding ssboLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding samplerLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, ssboLayoutBinding, samplerLayoutBinding };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        m_descriptorSetLayout = vk::raii::DescriptorSetLayout(m_gfx.getDevice(), layoutInfo);
    }

    static std::vector<char> readFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");
        }

        std::vector<char> buffer(file.tellg());

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }

    vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const {
        vk::ShaderModuleCreateInfo createInfo{};
        createInfo.codeSize = code.size() * sizeof(char);
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        return vk::raii::ShaderModule{ m_gfx.getDevice(), createInfo};
    }

    void createGraphicsPipeline() {
		auto surfaceFormat = m_gfx.getSwapChainSurfaceFormat();

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &surfaceFormat;

        auto fragCode = readFile("Shaders/main.frag.spv");
        auto vertCode = readFile("Shaders/main.vert.spv");

        auto fragModule = createShaderModule(fragCode);
        auto vertModule = createShaderModule(vertCode);

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fragShaderStageInfo.module = fragModule;
        fragShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vertShaderStageInfo.module = vertModule;
        vertShaderStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        std::vector<vk::DynamicState> dynamicStates = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        vk::PipelineDynamicStateCreateInfo dynamicState({}, dynamicStates.size(), dynamicStates.data());

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.cullMode = vk::CullModeFlagBits::eBack;
        rasterizer.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisampling{};
        multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask =
            vk::ColorComponentFlagBits::eR |
            vk::ColorComponentFlagBits::eG |
            vk::ColorComponentFlagBits::eB |
            vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &*m_descriptorSetLayout;

        m_pipelineLayout = vk::raii::PipelineLayout(m_gfx.getDevice(), pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.pNext = &pipelineRenderingCreateInfo;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = m_pipelineLayout;

        m_graphicsPipeline = vk::raii::Pipeline(m_gfx.getDevice(), nullptr, pipelineInfo);
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        auto pixels = stbi_load("Textures/statue.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        std::vector<uint8_t> imageBytes(pixels, pixels + (texWidth * texHeight * 4));

        stbi_image_free(pixels);

        vk::ImageCreateInfo imageInfo{};
        imageInfo.imageType = vk::ImageType::e2D;
        imageInfo.format = vk::Format::eR8G8B8A8Srgb;
        imageInfo.extent.width = texWidth;
        imageInfo.extent.height = texHeight;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

		m_textureImage = m_gfx.makeImage(imageInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
		m_gfx.updateImage(m_textureImage, imageBytes);
    }

    void createTextureImageView() {
        m_textureImageView = m_gfx.makeImageView(m_textureImage);

        vk::PhysicalDeviceProperties properties = m_gfx.getPhysicalDevice().getProperties();
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.anisotropyEnable = true;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        m_textureSampler = vk::raii::Sampler(m_gfx.getDevice(), samplerInfo);
    }

    void createVertexBuffer() {
		auto vertices = Cube::getVertices();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;

		m_vertexBuffer = m_gfx.makeBuffer(bufferInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
		m_gfx.updateBuffer(m_vertexBuffer, vertices);
	}

    void createIndexBuffer() {
		auto indices = Cube::getIndices();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
		m_indexBuffer = m_gfx.makeBuffer(bufferInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
		m_gfx.updateBuffer(m_indexBuffer, indices);
    }

    void createIndirectBuffer() {
        auto indices = Cube::getIndices();

        const vk::DrawIndexedIndirectCommand drawCmd = {
            static_cast<uint32_t>(indices.size()), // indexCount
            1, // instanceCount
            0, // firstIndex
            0, // vertexOffset
            0  // firstInstance
        };

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmd);
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
		m_indirectBuffer = m_gfx.makeBuffer(bufferInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);
		m_gfx.updateBuffer(m_indirectBuffer, drawCmd);
	}

    void createUniformBuffers() {
        m_uniformBuffers.clear();
        m_uniformBuffersMapped.clear();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        for (uint8_t i = 0; i < m_gfx.getMaxFramesInFlight(); i++) {
            auto uniformBuffer = m_gfx.makeBuffer(bufferInfo,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            vk::raii::DeviceMemory uniformBufferMemory = nullptr;

            m_uniformBuffersMapped.emplace_back(uniformBuffer.map());
            m_uniformBuffers.emplace_back(std::move(uniformBuffer));
        }
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(StorageBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

		m_storageBuffer = m_gfx.makeBuffer(bufferInfo, vk::MemoryPropertyFlagBits::eDeviceLocal);

		StorageBufferObject ssboData{};
		ssboData.colour = glm::vec3(1.0f, 1.0f, 0.0f);

		m_gfx.updateBuffer(m_storageBuffer, ssboData);
    }

    void createDescriptorPool() {
		auto maxFramesInFlight = m_gfx.getMaxFramesInFlight();

        std::array<vk::DescriptorPoolSize, 3> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, maxFramesInFlight },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = maxFramesInFlight;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        m_descriptorPool = vk::raii::DescriptorPool(m_gfx.getDevice(), poolInfo);
    }

    void createDescriptorSets() {
        auto maxFramesInFlight = m_gfx.getMaxFramesInFlight();

        std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, *m_descriptorSetLayout);

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        m_descriptorSets = m_gfx.getDevice().allocateDescriptorSets(allocInfo);

        // storage buffer descriptor info (same buffer for all sets)
        vk::DescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = m_storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = sizeof(StorageBufferObject);

        for (uint8_t i = 0; i < maxFramesInFlight; i++) {
            vk::DescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = m_uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(UniformBufferObject);

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.sampler = m_textureSampler;
            imageInfo.imageView = m_textureImageView;
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet uboWrite{};
            uboWrite.dstSet = m_descriptorSets[i];
            uboWrite.dstBinding = 0;
            uboWrite.dstArrayElement = 0;
            uboWrite.descriptorCount = 1;
            uboWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
            uboWrite.pBufferInfo = &uboBufferInfo;

            m_gfx.getDevice().updateDescriptorSets(uboWrite, {});

            vk::WriteDescriptorSet ssboWrite{};
            ssboWrite.dstSet = m_descriptorSets[i];
            ssboWrite.dstBinding = 1;
            ssboWrite.dstArrayElement = 0;
            ssboWrite.descriptorCount = 1;
            ssboWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
            ssboWrite.pBufferInfo = &storageBufferInfo;

            m_gfx.getDevice().updateDescriptorSets(ssboWrite, {});

            vk::WriteDescriptorSet imageWrite{};
            imageWrite.dstSet = m_descriptorSets[i];
            imageWrite.dstBinding = 2;
            imageWrite.dstArrayElement = 0;
            imageWrite.descriptorCount = 1;
            imageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            imageWrite.pImageInfo = &imageInfo;

            m_gfx.getDevice().updateDescriptorSets(imageWrite, {});
        }
    }

    void initRenderPasses()
    {
        auto& renderGraph = m_gfx.getRenderGraph();
        auto swapChainExtent = m_gfx.getSwapChainExtent();

        // Main rendering pass: transition Undefined -> ColorAttachmentOptimal and record in the pass
        Gfx::RenderPassNode mainPass{};
        mainPass.name = "MainPass";
        mainPass.oldLayout = vk::ImageLayout::eUndefined;
        mainPass.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainPass.srcAccessMask = {}; // from undefined
        mainPass.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainPass.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        mainPass.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;

        // record the same rendering commands previously inside recordCommandBuffer (beginRendering, bind pipeline, draw, endRendering)
        mainPass.recordFunc = [this, swapChainExtent](const vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
        {
			updateUniformBuffer(imageIndex, swapChainExtent);

            vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
            vk::RenderingAttachmentInfo attachmentInfo{};
            attachmentInfo.imageView = m_gfx.getSwapChainImageView(imageIndex);
            attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            attachmentInfo.clearValue = clearColor;

            vk::RenderingInfo renderingInfo{};
            renderingInfo.renderArea.offset.x = 0;
            renderingInfo.renderArea.offset.y = 0;
            renderingInfo.renderArea.extent = m_gfx.getSwapChainExtent();
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &attachmentInfo;

            cmd.beginRendering(renderingInfo);
            cmd.bindVertexBuffers(0, { m_vertexBuffer }, { 0 });
            cmd.bindIndexBuffer(m_indexBuffer, 0, vk::IndexType::eUint32);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_pipelineLayout, 0, *m_descriptorSets[imageIndex], nullptr);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, m_graphicsPipeline);

            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

            cmd.drawIndexedIndirect(m_indirectBuffer, 0, 1, static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

            cmd.endRendering();
        };

        renderGraph->addPass(mainPass);

        // Final transition pass: move from color attachment -> present.
        Gfx::RenderPassNode presentTransition{};
        presentTransition.name = "PresentTransition";
        presentTransition.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        presentTransition.newLayout = vk::ImageLayout::ePresentSrcKHR;
        presentTransition.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        presentTransition.dstAccessMask = {};
        presentTransition.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        presentTransition.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
        presentTransition.recordFunc = nullptr; // no recording, just a layout transition

        renderGraph->addPass(presentTransition);

        // finally initialize (allocates per-image command buffers and per-frame sync objects)
        renderGraph->init();
    }

    void updateUniformBuffer(uint32_t currentImage, vk::Extent2D swapChainExtent) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;

        memcpy(m_uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        m_gfx.getRenderGraph()->executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(m_window)) {
            glfwPollEvents();
            drawFrame();
        }

        m_gfx.getDevice().waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(m_window);

        glfwTerminate();
    }
};

int main() {
    try {
        HelloTriangleApplication app;
        app.run();
    }
    catch (const std::exception& e) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}