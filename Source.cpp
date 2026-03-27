#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <chrono>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>

#include "Buffer.hpp"
#include "Image.hpp"
#include "RenderGraph.hpp"
#include "RHI.hpp"

#undef max

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

struct Texture
{
    std::vector<uint8_t> imageData;
    int width;
	int height;
};

struct Instance
{
    glm::mat4 model;
    glm::vec3 colour;
};

struct UniformBufferObject
{
    glm::mat4 view;
    glm::mat4 proj;
	glm::quat rotation;
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

    Gfx::RHI rhi{};
    Gfx::RenderGraph graph{ rhi };

    std::vector<Vertex> vertices{};
    std::vector<uint32_t> indices{};
	std::vector<Texture> textures{};
    std::vector<vk::DrawIndexedIndirectCommand> drawCmds{};
	std::vector<Instance> instances{};

    vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
    vk::raii::PipelineLayout pipelineLayout = nullptr;
    vk::raii::Pipeline graphicsPipeline = nullptr;
    std::vector<Gfx::Image> textureImages{};
    std::vector<vk::raii::ImageView> textureImageViews{};
    vk::raii::Sampler textureSampler = nullptr;
    Gfx::Buffer vertexBuffer = nullptr;
    Gfx::Buffer indexBuffer = nullptr;
    Gfx::Buffer indirectBuffer = nullptr;
    Gfx::Buffer storageBuffer = nullptr;
    std::vector<Gfx::Buffer> uniformBuffers;
    std::vector<void*> uniformBuffersMapped;
    vk::raii::DescriptorPool descriptorPool = nullptr;
    std::vector<vk::raii::DescriptorSet> descriptorSets;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Renderer", nullptr, nullptr);
    }

    void initVulkan() {
		rhi.init("Vulkan Renderer", getRequiredExtensions(), glfwGetWin32Window(window));

        loadFloor();
        loadModel();

        createDescriptorSetLayout();
		createGraphicsPipeline();
        // create and initialize the render graph (allocates per-image command-buffers and sync)
        initRenderGraph();
		createTextureImages();
        createTextureImageViews();
        createVertexBuffer();
        createIndexBuffer();
        createIndirectBuffer();
        createUniformBuffers();
        createStorageBuffer();
		createDescriptorPool();
        createDescriptorSets();
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        return std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
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

		return vk::raii::ShaderModule{ rhi.getDevice(), createInfo};
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding ssboLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding samplerLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, textures.size(), vk::ShaderStageFlagBits::eFragment, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = { uboLayoutBinding, ssboLayoutBinding, samplerLayoutBinding };

        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        descriptorSetLayout = vk::raii::DescriptorSetLayout(rhi.getDevice(), layoutInfo);
    }

    void createGraphicsPipeline() {
		auto surfaceFormat = rhi.getSurfaceFormat();    

        vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
        pipelineRenderingCreateInfo.colorAttachmentCount = 1;
        pipelineRenderingCreateInfo.pColorAttachmentFormats = &surfaceFormat;
        pipelineRenderingCreateInfo.depthAttachmentFormat = rhi.getDepthFormat();

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

        vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.depthTestEnable = true;
        depthStencil.depthWriteEnable = true;
        depthStencil.depthCompareOp = vk::CompareOp::eLess;

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;

        pipelineLayout = vk::raii::PipelineLayout(rhi.getDevice(), pipelineLayoutInfo);

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
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.layout = pipelineLayout;

        graphicsPipeline = vk::raii::Pipeline(rhi.getDevice(), nullptr, pipelineInfo);
	}

    void loadFloor()
    {
        std::vector<Vertex> quad{
            {{-0.5, -0.5, 0.0}, {1.0, 0.0}},
            {{0.5, -0.5, 0.0}, {0.0, 0.0}},
            {{0.5, 0.5, 0.0}, {0.0, 1.0}},
            {{-0.5, 0.5, 0.0}, {1.0, 1.0}}
        };

        std::vector<uint32_t> quadIndices{
            0, 1, 2, 2, 3, 0
        };

        vk::DrawIndexedIndirectCommand drawCmd{
            static_cast<uint32_t>(quadIndices.size()), // index count
            1, // instance count
            static_cast<uint32_t>(indices.size()), // first index
            static_cast<int32_t>(vertices.size()), // vertex offset
            static_cast<uint32_t>(drawCmds.size()) // first instance
		};

        drawCmds.emplace_back(std::move(drawCmd));

        vertices.insert(vertices.end(), quad.begin(), quad.end());
        indices.insert(indices.end(), quadIndices.begin(), quadIndices.end());

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Textures/statue.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        texture.imageData.resize(texture.width * texture.height * 4);
        memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

        textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5)) * glm::scale(glm::mat4(1.0f), glm::vec3(4.0));
        instance.colour = glm::vec3(1.0f, 1.0f, 0.0f);

        instances.emplace_back(std::move(instance));
    }

    template<typename T>
    std::vector<T> ReadAccessor(const tinygltf::Model& model, const tinygltf::Accessor& accessor) {
        auto& view = model.bufferViews[accessor.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + accessor.byteOffset;
        auto stride = accessor.ByteStride(view);
        auto count = accessor.count;

        std::vector<T> out(count);

        if (stride == sizeof(T)) {
            // tightly packed
            memcpy(out.data(), dataPtr, count * sizeof(T));
        }
        else {
            // interleaved
            for (size_t i = 0; i < count; i++) {
                memcpy(&out[i], dataPtr + i * stride, sizeof(T));
            }
        }

        return out;
    }

    uint32_t LoadPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive)
    {
        const tinygltf::Accessor& posAcc =
            model.accessors[primitive.attributes.at("POSITION")];
        auto positions = ReadAccessor<glm::vec3>(model, posAcc);

        std::vector<glm::vec2> texCoords;
        if (primitive.attributes.count("TEXCOORD_0"))
        {
            auto& texCoordAcc = model.accessors[primitive.attributes.at("TEXCOORD_0")];
            texCoords = ReadAccessor<glm::vec2>(model, texCoordAcc);
        }
        else
        {
            texCoords.resize(positions.size(), glm::vec2(0));
        }

        vertices.reserve(vertices.size() + positions.size());
        for (size_t i = 0; i < positions.size(); i++)
        {
            vertices.emplace_back(Vertex{ positions[i], texCoords[i] });
        }

        auto& idxAcc = model.accessors[primitive.indices];
        auto& view = model.bufferViews[idxAcc.bufferView];
        auto& buffer = model.buffers[view.buffer];

        auto dataPtr = buffer.data.data() + view.byteOffset + idxAcc.byteOffset;

        auto count = idxAcc.count;

		indices.reserve(indices.size() + count);

        switch (idxAcc.componentType) {
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: 
        {
            auto src = reinterpret_cast<const uint16_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT: 
        {
            auto src = reinterpret_cast<const uint32_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE: 
        {
            auto src = reinterpret_cast<const uint8_t*>(dataPtr);
            for (size_t i = 0; i < count; i++)
            {
                indices.emplace_back(src[i]);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported index type");
        }

        return count;
    }

    void loadModel() {
        tinygltf::TinyGLTF loader{};

		tinygltf::Model model;
		std::string err;
        if (!loader.LoadASCIIFromFile(&model, &err, nullptr, "Models/CesiumMan.gltf")) {
			throw std::runtime_error("Failed to load model: " + err);
        }

        for (auto& mesh : model.meshes) {
            for (auto& primitive : mesh.primitives) {
                vk::DrawIndexedIndirectCommand drawCmd{
                    0, // index count
                    1, // instance count
                    static_cast<uint32_t>(indices.size()), // first index
                    static_cast<int32_t>(vertices.size()), // vertex offset
                    static_cast<uint32_t>(drawCmds.size()) // first instance
                };
                
                drawCmd.indexCount = LoadPrimitive(model, primitive);

                drawCmds.emplace_back(std::move(drawCmd));
            }
        }

        Texture texture{};

        int texChannels;
        auto pixels = stbi_load("Models/CesiumMan_img0.jpg", &texture.width, &texture.height, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

		texture.imageData.resize(texture.width * texture.height * 4);
		memcpy(texture.imageData.data(), pixels, texture.imageData.size());

        stbi_image_free(pixels);

		textures.emplace_back(std::move(texture));

        Instance instance{};
        instance.model = glm::translate(glm::mat4(1.0f), glm::vec3(0.0, 0.0, -0.5));
        instance.colour = glm::vec3(1.0f, 1.0f, 1.0f);

        instances.emplace_back(std::move(instance));
    }

    void createTextureImages() {
		textureImages.reserve(textures.size());

        for (auto& texture : textures) {
            vk::ImageCreateInfo imageInfo{};
            imageInfo.imageType = vk::ImageType::e2D;
            imageInfo.format = vk::Format::eR8G8B8A8Srgb;
            imageInfo.extent.width = texture.width;
            imageInfo.extent.height = texture.height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled;

            textureImages.emplace_back(std::move(rhi.createImage(imageInfo)));
            rhi.updateImage(textureImages.back(), texture.imageData);
        }
    }

    void createTextureImageViews() {
        for (size_t i = 0; i < textures.size(); i++)
        {
			textureImageViews.emplace_back(std::move(rhi.createImageView(textureImages[i])));
        }

        vk::PhysicalDeviceProperties properties = rhi.getPhysicalDevice().getProperties();
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = vk::Filter::eLinear;
        samplerInfo.minFilter = vk::Filter::eLinear;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.anisotropyEnable = true;
        samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;

        textureSampler = vk::raii::Sampler(rhi.getDevice(), samplerInfo);
    }

    void createVertexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst;

        vertexBuffer = rhi.createBuffer(bufferInfo);
		rhi.updateBuffer(vertexBuffer, vertices);
	}

    void createIndexBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(indices[0]) * indices.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
        indexBuffer = rhi.createBuffer(bufferInfo);
		rhi.updateBuffer(indexBuffer, indices);
    }

    void createIndirectBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(drawCmds[0]) * drawCmds.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eTransferDst;
        
        indirectBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(indirectBuffer, drawCmds);
	}

    void createUniformBuffers() {
        uniformBuffers.clear();
        uniformBuffersMapped.clear();

        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(UniformBufferObject);
        bufferInfo.usage = vk::BufferUsageFlagBits::eUniformBuffer;

        for (size_t i = 0; i < rhi.getMaxFramesInFlight(); i++) {
            Gfx::Buffer uniformBuffer = nullptr;
            vk::raii::DeviceMemory uniformBufferMemory = nullptr;
            uniformBuffer = rhi.createBuffer(bufferInfo,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent);
            uniformBuffersMapped.emplace_back(uniformBuffer.map());
            uniformBuffers.emplace_back(std::move(uniformBuffer));
        }
    }

    void createStorageBuffer() {
        vk::BufferCreateInfo bufferInfo{};
        bufferInfo.size = sizeof(instances[0]) * instances.size();
        bufferInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst;

        storageBuffer = rhi.createBuffer(bufferInfo);
        rhi.updateBuffer(storageBuffer, instances);
    }

    void createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 3> poolSizes = {
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, rhi.getMaxFramesInFlight() },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1 },
        };

        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
        poolInfo.maxSets = static_cast<uint32_t>(rhi.getMaxFramesInFlight());
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();

        descriptorPool = vk::raii::DescriptorPool(rhi.getDevice(), poolInfo);
    }

    void createDescriptorSets() {
        // storage buffer descriptor info (same buffer for all sets)
        vk::DescriptorBufferInfo storageBufferInfo{};
        storageBufferInfo.buffer = storageBuffer;
        storageBufferInfo.offset = 0;
        storageBufferInfo.range = sizeof(Instance);

        vk::DescriptorBufferInfo uboBufferInfo{};
        uboBufferInfo.offset = 0;
        uboBufferInfo.range = sizeof(UniformBufferObject);

        std::vector<vk::DescriptorImageInfo> imageInfos{};
        for (size_t i = 0; i < textures.size(); i++) 
        {
            vk::DescriptorImageInfo imageInfo{};
            imageInfo.sampler = textureSampler;
            imageInfo.imageView = textureImageViews[i];
            imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
			imageInfos.emplace_back(std::move(imageInfo));
        }

        vk::WriteDescriptorSet uboWrite{};
        uboWrite.dstBinding = 0;
        uboWrite.dstArrayElement = 0;
        uboWrite.descriptorCount = 1;
        uboWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        uboWrite.pBufferInfo = &uboBufferInfo;

        vk::WriteDescriptorSet ssboWrite{};
        ssboWrite.dstBinding = 1;
        ssboWrite.dstArrayElement = 0;
        ssboWrite.descriptorCount = 1;
        ssboWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
        ssboWrite.pBufferInfo = &storageBufferInfo;
        
        vk::WriteDescriptorSet imageWrite{};
        imageWrite.dstBinding = 2;
        imageWrite.dstArrayElement = 0;
        imageWrite.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        imageWrite.descriptorCount = imageInfos.size();
        imageWrite.pImageInfo = imageInfos.data();

        std::vector<vk::DescriptorSetLayout> layouts(rhi.getMaxFramesInFlight(), *descriptorSetLayout);

        vk::DescriptorSetAllocateInfo allocInfo{};
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets = rhi.getDevice().allocateDescriptorSets(allocInfo);

        for (size_t i = 0; i < rhi.getMaxFramesInFlight(); i++) {
            uboBufferInfo.buffer = uniformBuffers[i];
            uboWrite.dstSet = descriptorSets[i];
            ssboWrite.dstSet = descriptorSets[i];
            imageWrite.dstSet = descriptorSets[i];

            rhi.getDevice().updateDescriptorSets(uboWrite, {});
            rhi.getDevice().updateDescriptorSets(ssboWrite, {});
            rhi.getDevice().updateDescriptorSets(imageWrite, {});
        }
    }

    // New: create and initialize the RenderGraph and add the passes used by the app
    void initRenderGraph()
    {
        // Main rendering pass: transition Undefined -> ColorAttachmentOptimal and record in the pass
        Gfx::RenderPassNode mainPass{};
        mainPass.name = "MainPass";
        Gfx::RenderPassNode::AttachmentTransitionInfo mainColorTransition{ rhi.getSwapChain().getImages(), vk::ImageAspectFlagBits::eColor};
        mainColorTransition.oldLayout = vk::ImageLayout::eUndefined;
        mainColorTransition.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainColorTransition.srcAccessMask = {}; // from undefined
        mainColorTransition.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainColorTransition.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
        mainColorTransition.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        mainPass.transitionInfos.emplace_back(mainColorTransition);
        Gfx::RenderPassNode::AttachmentTransitionInfo mainDepthTransition{ rhi.getDepthImages(), vk::ImageAspectFlagBits::eDepth};
        mainDepthTransition.oldLayout = vk::ImageLayout::eUndefined;
        mainDepthTransition.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
        mainDepthTransition.srcAccessMask = {}; // from undefined
        mainDepthTransition.dstAccessMask = vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
        mainDepthTransition.srcStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
        mainDepthTransition.dstStageMask = vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests;
		mainPass.transitionInfos.emplace_back(std::move(mainDepthTransition));

        // record the same rendering commands previously inside recordCommandBuffer (beginRendering, bind pipeline, draw, endRendering)
        mainPass.recordFunc = [this](vk::raii::CommandBuffer& cmd, uint32_t imageIndex)
        {
			updateUniformBuffer(imageIndex);

            vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
            vk::RenderingAttachmentInfo colorAttachmentInfo{};
            colorAttachmentInfo.imageView = rhi.getSwapChainImageView(imageIndex);
            colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
            colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
            colorAttachmentInfo.clearValue = clearColor;

            vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1, 0);
            vk::RenderingAttachmentInfo depthAttachmentInfo{};
            depthAttachmentInfo.imageView = rhi.getDepthImageView(imageIndex);
            depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
            depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
            depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
            depthAttachmentInfo.clearValue = clearDepth;

            auto swapChainExtent = rhi.getSwapChainExtent();

            vk::RenderingInfo renderingInfo{};
            renderingInfo.renderArea.offset.x = 0;
            renderingInfo.renderArea.offset.y = 0;
            renderingInfo.renderArea.extent = swapChainExtent;
            renderingInfo.layerCount = 1;
            renderingInfo.colorAttachmentCount = 1;
            renderingInfo.pColorAttachments = &colorAttachmentInfo;
            renderingInfo.pDepthAttachment = &depthAttachmentInfo;

            cmd.beginRendering(renderingInfo);
            cmd.bindVertexBuffers(0, *vertexBuffer, { 0 });
            cmd.bindIndexBuffer(*indexBuffer, 0, vk::IndexType::eUint32);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, *descriptorSets[imageIndex], nullptr);

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

            cmd.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
            cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));

            cmd.drawIndexedIndirect(*indirectBuffer, 0, drawCmds.size(), static_cast<uint32_t>(sizeof(VkDrawIndexedIndirectCommand)));

            cmd.endRendering();
        };

        graph.addPass(mainPass);

        // Final transition pass: move from color attachment -> present.
        Gfx::RenderPassNode presentTransition{};
        presentTransition.name = "PresentTransition";
        mainColorTransition.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
        mainColorTransition.newLayout = vk::ImageLayout::ePresentSrcKHR;
        mainColorTransition.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
        mainColorTransition.dstAccessMask = {};
        mainColorTransition.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
        mainColorTransition.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
		presentTransition.transitionInfos.emplace_back(std::move(mainColorTransition));
        presentTransition.recordFunc = nullptr; // no recording, just a layout transition

        graph.addPass(presentTransition);

        // finally initialize (allocates per-image command buffers and per-frame sync objects)
        graph.init();
    }

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

		auto swapChainExtent = rhi.getSwapChainExtent();

        UniformBufferObject ubo{};
        ubo.view = lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), static_cast<float>(swapChainExtent.width) / static_cast<float>(swapChainExtent.height), 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        ubo.rotation = glm::angleAxis(time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

        memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
    }

    // removed transitionImageLayout and recordCommandBuffer methods - RenderGraph now handles transitions + per-pass recording

    // removed createSyncObjects - RenderGraph manages per-frame sync

    void drawFrame() {
        // delegate frame orchestration to the render graph
        graph.executeFrame();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
            drawFrame();
        }

        rhi.getDevice().waitIdle();
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
