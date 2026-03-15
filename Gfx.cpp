#include "Gfx.h"

#include <fstream>
#include <windows.h>

#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES

#include <glm/glm.hpp>
#include <vulkan/vulkan_raii.hpp>

#include "RenderGraph.hpp"

#undef max

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
};

static std::tuple<uint32_t, uint32_t> findQueueFamilies(const vk::raii::PhysicalDevice& physicalDevice, const vk::raii::SurfaceKHR& surface) {
    // find the index of the first queue family that supports graphics
    auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports graphics
    auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
        [](vk::QueueFamilyProperties const& qfp)
        {
            return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
        });

    auto graphicsIndex = static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));

    // determine a queueFamilyIndex that supports present
    // first check if the graphicsIndex is good enough
    auto presentIndex = physicalDevice.getSurfaceSupportKHR(graphicsIndex, *surface)
        ? graphicsIndex
        : static_cast<uint32_t>(queueFamilyProperties.size());

    if (presentIndex == queueFamilyProperties.size())
    {
        // the graphicsIndex doesn't support present -> look for another family index that supports both
        // graphics and present
        for (size_t i = 0; i < queueFamilyProperties.size(); i++)
        {
            if ((queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics) &&
                physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
            {
                graphicsIndex = static_cast<uint32_t>(i);
                presentIndex = graphicsIndex;
                break;
            }
        }
        if (presentIndex == queueFamilyProperties.size())
        {
            // there's nothing like a single family index that supports both graphics and present -> look for another
            // family index that supports present
            for (size_t i = 0; i < queueFamilyProperties.size(); i++)
            {
                if (physicalDevice.getSurfaceSupportKHR(static_cast<uint32_t>(i), *surface))
                {
                    presentIndex = static_cast<uint32_t>(i);
                    break;
                }
            }
        }
    }

    if ((graphicsIndex == queueFamilyProperties.size()) || (presentIndex == queueFamilyProperties.size()))
    {
        throw std::runtime_error("Could not find a queue for graphics or present -> terminating");
    }

    return { graphicsIndex, presentIndex };
}

static uint32_t chooseSwapMinImageCount(vk::SurfaceCapabilitiesKHR const& surfaceCapabilities) {
    auto minImageCount = surfaceCapabilities.minImageCount + 1;
    if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount)) {
        minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
}

static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.format == vk::Format::eB8G8R8A8Srgb &&
            availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }

    return vk::PresentModeKHR::eFifo;
}

static vk::Extent2D chooseSwapExtent(void* window, const vk::SurfaceCapabilitiesKHR& capabilities) {
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    RECT rect;
    GetClientRect(static_cast<HWND>(window), &rect);
    uint32_t width = rect.right - rect.left;
    uint32_t height = rect.bottom - rect.top;

    return {
        glm::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
        glm::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
    };
}

static uint32_t findMemoryType(const vk::raii::PhysicalDevice& physicalDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void Gfx::init(const std::string& appName, const std::vector<const char*>& extensions, void* window) {
    createInstance(appName, extensions);
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain(window);
    createImageViews();
    createCommandPool();
    initRenderGraph();
}

void Gfx::createInstance(const std::string& appName, const std::vector<const char*>& extensions) {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "Vulkan Renderer";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion14;

    // Get the required layers
    std::vector<char const*> requiredLayers{};
    if (enableValidationLayers) {
        requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = m_context.enumerateInstanceLayerProperties();
    if (std::any_of(requiredLayers.begin(), requiredLayers.end(),
        [&layerProperties](auto const& requiredLayer)
        {
            return std::none_of(layerProperties.begin(), layerProperties.end(),
                [requiredLayer](auto const& layerProperty)
                {
                    return strcmp(layerProperty.layerName, requiredLayer) == 0;
                });
        }))
    {
        throw std::runtime_error("One or more required layers are not supported!");
    }

    // Check if the required GLFW extensions are supported by the Vulkan implementation.
    auto extensionProperties = m_context.enumerateInstanceExtensionProperties();
    for (size_t i = 0; i < extensions.size(); ++i)
    {
        auto extension = extensions[i];
        if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
            [extension](auto const& extensionProperty)
            {
                return std::strcmp(extensionProperty.extensionName, extension) == 0;
            }))
        {
            throw std::runtime_error(std::string("Required GLFW extension not supported: ") + extension);
        }
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
    createInfo.ppEnabledLayerNames = requiredLayers.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    m_instance = vk::raii::Instance(m_context, createInfo);
}

void Gfx::createSurface(void* window) {
    vk::Win32SurfaceCreateInfoKHR createInfo{};
    createInfo.sType = vk::StructureType::eWin32SurfaceCreateInfoKHR;
    createInfo.hwnd = static_cast<HWND>(window);
    createInfo.hinstance = GetModuleHandle(nullptr);

    m_surface = vk::raii::SurfaceKHR(m_instance, createInfo);
}

void Gfx::pickPhysicalDevice() {
    auto devices = m_instance.enumeratePhysicalDevices();
    const auto devIter = std::find_if(devices.begin(), devices.end(),
        [&](auto const& device)
        {
            auto queueFamilies = device.getQueueFamilyProperties();
            auto isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;
            const auto qfpIter = std::find_if(queueFamilies.begin(), queueFamilies.end(),
                [](vk::QueueFamilyProperties const& qfp)
                {
                    return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
                });
            isSuitable = isSuitable && (qfpIter != queueFamilies.end());
            auto extensions = device.enumerateDeviceExtensionProperties();
            auto found = true;
            for (auto const& extension : deviceExtensions) {
                auto extensionIter = std::find_if(extensions.begin(), extensions.end(),
                    [extension](auto const& ext)
                    {
                        return strcmp(ext.extensionName, extension) == 0;
                    });
                found = found && extensionIter != extensions.end();
            }
            isSuitable = isSuitable && found;
            if (isSuitable) {
                m_physicalDevice = device;
            }
            return isSuitable;
        });
    if (devIter == devices.end()) {
        throw std::runtime_error("failed to find a suitable GPU!");
    }
}

void Gfx::createLogicalDevice() {
    std::tie(m_graphicsFamily, m_presentFamily) = findQueueFamilies(m_physicalDevice, m_surface);
    auto queuePriority = 0.0f;

    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.queueFamilyIndex = m_graphicsFamily;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    vk::PhysicalDeviceFeatures2 features2{};
    features2.features.samplerAnisotropy = true;
    vk::PhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.dynamicRendering = true; // Enable dynamic rendering from Vulkan 1.3
    vulkan13Features.synchronization2 = true; // enable synchronization2 from the extension
    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynamicStateFeatures{};
    extDynamicStateFeatures.extendedDynamicState = true; // Enable extended dynamic state from the extension

    // Create a chain of feature structures
    auto featureChain = vk::StructureChain<
        vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
    { features2, vulkan13Features, extDynamicStateFeatures };

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

    m_device = vk::raii::Device(m_physicalDevice, deviceCreateInfo);
    m_graphicsQueue = vk::raii::Queue(m_device, m_graphicsFamily, 0);
    m_presentQueue = vk::raii::Queue(m_device, m_presentFamily, 0);
}

void Gfx::createSwapChain(void* window) {
    auto surfaceCapabilities = m_physicalDevice.getSurfaceCapabilitiesKHR(m_surface);
    auto availableFormats = m_physicalDevice.getSurfaceFormatsKHR(m_surface);
    auto availablePresentModes = m_physicalDevice.getSurfacePresentModesKHR(m_surface);

    m_swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);
    m_swapChainExtent = chooseSwapExtent(window, surfaceCapabilities);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.flags = vk::SwapchainCreateFlagsKHR();
    swapChainCreateInfo.surface = m_surface;
    swapChainCreateInfo.minImageCount = chooseSwapMinImageCount(surfaceCapabilities);
    swapChainCreateInfo.imageFormat = m_swapChainSurfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = m_swapChainSurfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = m_swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1; // keep 1 unless rendering for VR
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment; // we are rendering to image directly
    swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;  // don't apply further transformation
    swapChainCreateInfo.presentMode = chooseSwapPresentMode(availablePresentModes);
    swapChainCreateInfo.clipped = true;  // don't update the pixels that are obscured

    uint32_t queueFamilyIndices[] = { m_graphicsFamily, m_presentFamily };

    if (m_graphicsFamily != m_presentFamily) {
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        swapChainCreateInfo.queueFamilyIndexCount = 2;
        swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
    }

    m_swapChain = vk::raii::SwapchainKHR(m_device, swapChainCreateInfo);
    m_swapChainImages = m_swapChain.getImages();
	m_maxFramesInFlight = static_cast<uint8_t>(m_swapChainImages.size());
}

void Gfx::createImageViews() {
    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = m_swapChainSurfaceFormat.format;
    imageViewCreateInfo.subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 };
    imageViewCreateInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    imageViewCreateInfo.subresourceRange.levelCount = 1;
    imageViewCreateInfo.subresourceRange.layerCount = 1;

    for (auto image : m_swapChainImages) {
        imageViewCreateInfo.image = image;
        m_swapChainImageViews.emplace_back(m_device, imageViewCreateInfo);
    }
}

void Gfx::createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = m_graphicsFamily;

    m_commandPool = vk::raii::CommandPool(m_device, poolInfo);
}

void Gfx::initRenderGraph()
{
    // construct the render graph (holds references, does NOT copy objects)
    m_renderGraph.reset(new RenderGraph(m_device, m_swapChain, m_graphicsQueue, m_presentQueue, m_commandPool));
}

Gfx::Buffer Gfx::makeBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties)
{
    vk::raii::Buffer buffer(m_device, bufferInfo);

    auto memRequirements = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, memProperties);

    vk::raii::DeviceMemory bufferMemory(m_device, allocInfo);

    buffer.bindMemory(bufferMemory, 0);

    return Gfx::Buffer(std::move(buffer), std::move(bufferMemory), bufferInfo.size);
}

void Gfx::updateBuffer(const Buffer& buffer, void* contentData, size_t contentSize)
{
    vk::BufferCreateInfo stagingInfo{};
    stagingInfo.size = contentSize;
    stagingInfo.usage = vk::BufferUsageFlagBits::eTransferSrc;

    auto stagingBuffer = makeBuffer(stagingInfo,
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent);

    void* data = stagingBuffer.map();
    memcpy(data, contentData, stagingInfo.size);
    stagingBuffer.unmap();

    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;

    auto commandCopyBuffer = std::move(m_device.allocateCommandBuffers(allocInfo).front());
    commandCopyBuffer.begin({ vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
	vk::BufferCopy region{};
	region.size = stagingInfo.size;
    commandCopyBuffer.copyBuffer(stagingBuffer, buffer, region);
    commandCopyBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandCopyBuffer;

    // TODO: use a fence instead
    m_graphicsQueue.submit(submitInfo, nullptr);
    m_graphicsQueue.waitIdle();
}

Gfx::Image Gfx::makeImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties)
{
    vk::raii::Image image(m_device, imageInfo);

    auto memRequirements = image.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(m_physicalDevice, memRequirements.memoryTypeBits, properties);

    vk::raii::DeviceMemory imageMemory(m_device, allocInfo);

    image.bindMemory(imageMemory, 0);

    return Gfx::Image(std::move(image), std::move(imageMemory), imageInfo.extent);
}
