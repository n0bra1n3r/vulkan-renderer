#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

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

#include "Gfx.h"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

struct Vertex
{
    glm::vec2 position;
    glm::vec2 texCoord;

    static vk::VertexInputBindingDescription getBindingDescription() {
        vk::VertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        return bindingDescription;
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
    }
};

const std::vector<Vertex> vertices =
{
    {{-0.5, -0.5}, {1.0, 0.0}},
    {{0.5, -0.5}, {0.0, 0.0}},
    {{0.5, 0.5}, {0.0, 1.0}},
    {{-0.5, 0.5}, {1.0, 1.0}}
};

const std::vector<uint32_t> indices =
{
    0, 1, 2, 2, 3, 0
};

const vk::DrawIndexedIndirectCommand drawCmd = {
    static_cast<uint32_t>(indices.size()), // indexCount
    1, // instanceCount
    0, // firstIndex
    0, // vertexOffset
    0  // firstInstance
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
    GLFWwindow* window = nullptr;
	Gfx gfx;

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    vk::raii::Image textureImage = nullptr;
    vk::raii::DeviceMemory textureImageMemory = nullptr;
    vk::raii::ImageView textureImageView = nullptr;
    vk::raii::Sampler textureSampler = nullptr;
    vk::raii::Buffer vertexBuffer = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer indexBuffer = nullptr;
    vk::raii::DeviceMemory indexBufferMemory = nullptr;
    vk::raii::Buffer indirectBuffer = nullptr;
    vk::raii::DeviceMemory indirectBufferMemory = nullptr;
    vk::raii::Buffer storageBuffer = nullptr;
    vk::raii::DeviceMemory storageBufferMemory = nullptr;
    std::vector<vk::raii::Buffer> uniformBuffers;
    std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
		gfx.init("Vulkan App", getRequiredExtensions(), glfwGetWin32Window(window));

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

        descriptorSetLayout = vk::raii::DescriptorSetLayout(gfx.getDevice(), layoutInfo);
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

        return vk::raii::ShaderModule{ gfx.getDevice(), createInfo};
    }

    void createGraphicsPipeline() {
		auto surfaceFormat = gfx.getSwapChainSurfaceFormat();

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
        pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;

        pipelineLayout = vk::raii::PipelineLayout(gfx.getDevice(), pipelineLayoutInfo);

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
        pipelineInfo.layout = pipelineLayout;

        graphicsPipeline = vk::raii::Pipeline(gfx.getDevice(), nullptr, pipelineInfo);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
        auto memProperties = gfx.getPhysicalDevice().getMemoryProperties();

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");
    }

    void createBuffer(const vk::BufferCreateInfo& bufferInfo, const vk::MemoryPropertyFlags& properties, vk::raii::Buffer& buffer, vk::raii::DeviceMemory& bufferMemory) {
        buffer = vk::raii::Buffer(gfx.getDevice(), bufferInfo);
        vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        bufferMemory = vk::raii::DeviceMemory(gfx.getDevice(), allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }

    void createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties, vk::raii::Image& image, vk::raii::DeviceMemory& imageMemory) {
        image = vk::raii::Image(gfx.getDevice(), imageInfo);
        vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
        vk::MemoryAllocateInfo allocInfo{};
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);
        imageMemory = vk::raii::DeviceMemory(gfx.getDevice(), allocInfo);
        image.bindMemory(imageMemory, 0);
    }

    vk::raii::ImageView createImageView(vk::raii::Image& image, vk::Format format) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = image;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
        return vk::raii::ImageView(gfx.getDevice(), viewInfo);
    }

    template<class T>
    void uploadBuffer(const std::vector<T>& contents, const vk::raii::Buffer& buffer) {
        vk::BufferCreateInfo stagingInfo{};
        stagingInfo.size = sizeof(contents[0]) * contents.size();
        stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(
            stagingInfo,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, stagingInfo.size);
        memcpy(data, contents.data(), stagingInfo.size);
        stagingBufferMemory.unmapMemory();

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = gfx.getCommandPool();
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;

        auto commandCopyBuffer = std::move(gfx.getDevice().allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(stagingBuffer, buffer, vk::BufferCopy(0, 0, stagingInfo.size));
        commandCopyBuffer.end();

        vk::SubmitInfo submitInfo{};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &*commandCopyBuffer;

		auto graphicsQueue = gfx.getGraphicsQueue();

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    template<class T>
    void uploadImage(const std::vector<T>& contents, uint32_t width, uint32_t height, const vk::raii::Image& image) {
        vk::BufferCreateInfo stagingInfo{};
        stagingInfo.size = sizeof(contents[0]) * contents.size();
        stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

        vk::raii::Buffer stagingBuffer = nullptr;
        vk::raii::DeviceMemory stagingBufferMemory = nullptr;

        createBuffer(
            stagingInfo,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingBuffer,
            stagingBufferMemory);

        void* data = stagingBufferMemory.mapMemory(0, stagingInfo.size);
        memcpy(data, contents.data(), stagingInfo.size);
        stagingBufferMemory.unmapMemory();

        vk::CommandBufferAllocateInfo allocInfo{};
        allocInfo.commandPool = gfx.getCommandPool();
        allocInfo.level = vk::CommandBufferLevel::ePrimary;
        allocInfo.commandBufferCount = 1;

        auto commandCopyBuffer = std::move(gfx.getDevice().allocateCommandBuffers(allocInfo).front());
        commandCopyBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
		transitionImageLayout(commandCopyBuffer, image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        vk::BufferImageCopy region{};
        region.imageSubresource = { vk::ImageAspectFlagBits::eColor, 0, 0, 1 };
        region.imageExtent.width = width;
        region.imageExtent.height = height;
        region.imageExtent.depth = 1;
        commandCopyBuffer.copyBufferToImage(stagingBuffer, image, vk::ImageLayout::eTransferDstOptimal, { region });
        transitionImageLayout(commandCopyBuffer, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
        commandCopyBuffer.end();

        vk::SubmitInfo submitInfo{};
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &*commandCopyBuffer;

        auto graphicsQueue = gfx.getGraphicsQueue();

        graphicsQueue.submit(submitInfo, nullptr);
        graphicsQueue.waitIdle();
    }

    void transitionImageLayout(const vk::raii::CommandBuffer& commandBuffer, const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::ImageMemoryBarrier barrier{};
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.image = image;
        barrier.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };

        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal)
        {
            barrier.srcAccessMask = {};
            barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        }
        else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal)
        {
            barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        }
        else
        {
            throw std::invalid_argument("unsupported layout transition!");
        }

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
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

        createImage(
            imageInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            textureImage,
            textureImageMemory);

		uploadImage(imageBytes, texWidth, texHeight, textureImage);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);

        vk::PhysicalDeviceProperties properties = gfx.getPhysicalDevice().getProperties();
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.anisotropyEnable = true;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        textureSampler = vk::raii::Sampler(gfx.getDevice(), samplerInfo);
    }

    void createVertexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo, 
            vk::MemoryPropertyFlagBits::eDeviceLocal, 
            vertexBuffer, 
            vertexBufferMemory);

		uploadBuffer(vertices, vertexBuffer);
	}

    void createIndexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indexBuffer,
            indexBufferMemory);

		uploadBuffer(indices, indexBuffer);
    }

    void createIndirectBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmd);
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            indirectBuffer,
            indirectBufferMemory);

        uploadBuffer(std::vector<vk::DrawIndexedIndirectCommand>{ drawCmd }, indirectBuffer);
	}

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMemory.clear();
        uniformBuffersMapped.clear();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        for (uint8_t i = 0; i < gfx.getMaxFramesInFlight(); i++) {
            vk::raii::Buffer uniformBuffer = nullptr;
            vk::raii::DeviceMemory uniformBufferMemory = nullptr;
            createBuffer(
                bufferInfo,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
                uniformBuffer,
                uniformBufferMemory);
            uniformBuffers.emplace_back(std::move(uniformBuffer));
            uniformBuffersMemory.emplace_back(std::move(uniformBufferMemory));
            uniformBuffersMapped.emplace_back(uniformBuffersMemory[i].mapMemory(0, bufferInfo.size));
        }
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(StorageBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

        createBuffer(
            bufferInfo,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            storageBuffer,
            storageBufferMemory);

		StorageBufferObject ssboData{};
		ssboData.colour = glm::vec3(1.0f, 1.0f, 0.0f);

        uploadBuffer(std::vector<StorageBufferObject>{ssboData}, storageBuffer);
    }

    void createDescriptorPool() {
		auto maxFramesInFlight = gfx.getMaxFramesInFlight();

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

        descriptorPool = vk::raii::DescriptorPool(gfx.getDevice(), poolInfo);
    }

    void createDescriptorSets() {
        auto maxFramesInFlight = gfx.getMaxFramesInFlight();

        std::vector<vk::DescriptorSetLayout> layouts(maxFramesInFlight, *descriptorSetLayout);

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets = gfx.getDevice().allocateDescriptorSets(allocInfo);

        // storage buffer descriptor info (same buffer for all sets)
        vk::DescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = sizeof(StorageBufferObject);

        for (uint8_t i = 0; i < maxFramesInFlight; i++) {
            vk::DescriptorBufferInfo uboBufferInfo{};
            uboBufferInfo.buffer = uniformBuffers[i];
            uboBufferInfo.offset = 0;
            uboBufferInfo.range = sizeof(UniformBufferObject);

            vk::DescriptorImageInfo imageInfo{};
            imageInfo.sampler = textureSampler;
            imageInfo.imageView = textureImageView;
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::WriteDescriptorSet uboWrite{};
            uboWrite.dstSet = descriptorSets[i];
            uboWrite.dstBinding = 0;
            uboWrite.dstArrayElement = 0;
            uboWrite.descriptorCount = 1;
            uboWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
            uboWrite.pBufferInfo = &uboBufferInfo;

            gfx.getDevice().updateDescriptorSets(uboWrite, {});

            vk::WriteDescriptorSet ssboWrite{};
            ssboWrite.dstSet = descriptorSets[i];
            ssboWrite.dstBinding = 1;
            ssboWrite.dstArrayElement = 0;
            ssboWrite.descriptorCount = 1;
            ssboWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
            ssboWrite.pBufferInfo = &storageBufferInfo;

            gfx.getDevice().updateDescriptorSets(ssboWrite, {});

            vk::WriteDescriptorSet imageWrite{};
            imageWrite.dstSet = descriptorSets[i];
            imageWrite.dstBinding = 2;
            imageWrite.dstArrayElement = 0;
            imageWrite.descriptorCount = 1;
            imageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            imageWrite.pImageInfo = &imageInfo;

            gfx.getDevice().updateDescriptorSets(imageWrite, {});
        }
    }

    void initRenderPasses()
    {
        auto& renderGraph = gfx.getRenderGraph();
        auto swapChainExtent = gfx.getSwapChainExtent();

        // Main rendering pass: transition Undefined -> ColorAttachmentOptimal and record in the pass
        RenderPassNode mainPass{};
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
            attachmentInfo.imageView = gfx.getSwapChainImageView(imageIndex);
            attachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            attachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            attachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            attachmentInfo.clearValue = clearColor;

            vk::RenderingInfo renderingInfo{};
            renderingInfo.renderArea.offset.x = 0;
            renderingInfo.renderArea.offset.y = 0;
            renderingInfo.renderArea.extent = gfx.getSwapChainExtent();
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &attachmentInfo;

            cmd.beginRendering(renderingInfo);
            cmd.bindVertexBuffers(0, *vertexBuffer, { 0 });
            cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[imageIndex], nullptr);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

            cmd.drawIndexedIndirect(*indirectBuffer, 0, 1, static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

            cmd.endRendering();
        };

        renderGraph->addPass(mainPass);

        // Final transition pass: move from color attachment -> present.
        RenderPassNode presentTransition{};
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

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    void drawFrame() {
        gfx.getRenderGraph()->executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        gfx.getDevice().waitIdle();
    }

    void cleanup() {
        glfwDestroyWindow(window);

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