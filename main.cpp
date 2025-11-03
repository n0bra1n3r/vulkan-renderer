#include <iostream>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<char const*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

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

	vk::raii::Context context{};
	vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    vk::raii::Queue graphicsQueue = nullptr;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() {
        createInstance();
        pickPhysicalDevice();
        createLogicalDevice();
    }

    void createInstance() {
        vk::ApplicationInfo appInfo;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "Vulkan Renderer";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = vk::ApiVersion14;

        // Get the required layers
        std::vector<char const*> requiredLayers;
        if (enableValidationLayers) {
            requiredLayers.assign(validationLayers.begin(), validationLayers.end());
        }

        // Check if the required layers are supported by the Vulkan implementation.
        auto layerProperties = context.enumerateInstanceLayerProperties();
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

        // Get the required instance extensions from GLFW.
        uint32_t glfwExtensionCount = 0;
        auto glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Check if the required GLFW extensions are supported by the Vulkan implementation.
        auto extensionProperties = context.enumerateInstanceExtensionProperties();
        for (uint32_t i = 0; i < glfwExtensionCount; ++i)
        {
            auto glfwExtension = glfwExtensions[i];
            if (std::none_of(extensionProperties.begin(), extensionProperties.end(),
                [glfwExtension](auto const& extensionProperty)
                {
                    return std::strcmp(extensionProperty.extensionName, glfwExtension) == 0;
                }))
            {
                throw std::runtime_error(std::string("Required GLFW extension not supported: ") + glfwExtension);
            }
        }

		vk::InstanceCreateInfo createInfo;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
		createInfo.ppEnabledLayerNames = requiredLayers.data();
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        instance = vk::raii::Instance(context, createInfo);
    }

    void pickPhysicalDevice() {
        auto devices = instance.enumeratePhysicalDevices();
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
                    physicalDevice = device;
                }
                return isSuitable;
            });
        if (devIter == devices.end()) {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
	}

    void createLogicalDevice() {
		auto graphicsIndex = findQueueFamilies(*physicalDevice);
        auto queuePriority = 0.0f;

        vk::DeviceQueueCreateInfo deviceQueueCreateInfo;
        deviceQueueCreateInfo.queueFamilyIndex = graphicsIndex;
		deviceQueueCreateInfo.queueCount = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

        vk::PhysicalDeviceFeatures2 features2{}; // vk::PhysicalDeviceFeatures2 (empty for now)
        vk::PhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.dynamicRendering = true; // Enable dynamic rendering from Vulkan 1.3
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT extDynamicStateFeatures{};
        extDynamicStateFeatures.extendedDynamicState = true; // Enable extended dynamic state from the extension

        // Create a chain of feature structures
        auto featureChain = vk::StructureChain<
            vk::PhysicalDeviceFeatures2,
            vk::PhysicalDeviceVulkan13Features,
            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
        { features2, vulkan13Features, extDynamicStateFeatures };

        vk::DeviceCreateInfo deviceCreateInfo;
        deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
        deviceCreateInfo.queueCreateInfoCount = 1;
        deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
        deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        device = vk::raii::Device(physicalDevice, deviceCreateInfo);
        graphicsQueue = vk::raii::Queue(device, graphicsIndex, 0);
    }

    uint32_t findQueueFamilies(VkPhysicalDevice device) {
        // find the index of the first queue family that supports graphics
        auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

        // get the first index into queueFamilyProperties which supports graphics
        auto graphicsQueueFamilyProperty = std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
            [](vk::QueueFamilyProperties const& qfp) 
            { 
                return qfp.queueFlags & vk::QueueFlagBits::eGraphics;
            });

        return static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
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
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}