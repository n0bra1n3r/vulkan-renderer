#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
    class Buffer;
	class Image;
	class RenderGraph;

    class Api
    {
    public:
        Api() = default;
        Api(const Api&) = delete;

        void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

        const vk::raii::Device& getDevice() const { return m_device; }
        const vk::raii::PhysicalDevice& getPhysicalDevice() const { return m_physicalDevice; }
        const vk::raii::Queue& getGraphicsQueue() const { return m_graphicsQueue; }
        vk::Format getSwapChainSurfaceFormat() const { return m_swapChainSurfaceFormat.format; }
        vk::Format getSwapChainDepthFormat() const { return m_swapChainDepthFormat; }
        vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }
        uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
        const vk::raii::ImageView& getSwapChainColorImageView(size_t index) const { return m_swapChainColorImageViews[index]; }
        const vk::raii::ImageView& getSwapChainDepthImageView(size_t index) const { return m_swapChainDepthImageViews[index]; }
        const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
        const std::unique_ptr<Gfx::RenderGraph>& getRenderGraph() const { return m_renderGraph; }

        Buffer makeBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties);
        void updateBuffer(const Buffer& buffer, void* contentData, size_t contentSize);

        Image makeImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties);
        void updateImage(const Gfx::Image& image, void* contentData, size_t contentSize);
        vk::raii::ImageView makeImageView(const Image& image);

        template<typename T>
        void updateBuffer(const Buffer& buffer, T data) {
            updateBuffer(buffer, &data, sizeof(T));
        }

        template<typename T>
        void updateBuffer(const Buffer& buffer, std::vector<T> data) {
            updateBuffer(buffer, data.data(), data.size() * sizeof(T));
        }

        template<typename T>
        void updateImage(const Image& image, std::vector<T> data) {
            updateImage(image, data.data(), data.size() * sizeof(T));
        }

    private:
        void createInstance(const std::string& appName, const std::vector<const char*>& extensions);
        void createSurface(void* window);
        void pickPhysicalDevice();
        void createLogicalDevice();
        void createSwapChain(void* window);
        void createImageViews();
        void createCommandPool();

        void initRenderGraph();

    private:
        vk::raii::Context m_context{};
        vk::raii::Instance m_instance = nullptr;
        vk::raii::PhysicalDevice m_physicalDevice = nullptr;
        vk::raii::Device m_device = nullptr;
        uint32_t m_graphicsFamily = 0, m_presentFamily = 0;
        vk::raii::Queue m_graphicsQueue = nullptr;
        vk::raii::Queue m_presentQueue = nullptr;
        vk::raii::SurfaceKHR m_surface = nullptr;
        vk::SurfaceFormatKHR m_swapChainSurfaceFormat{};
		vk::Format m_swapChainDepthFormat{};
        vk::Extent2D m_swapChainExtent{};
        vk::raii::SwapchainKHR m_swapChain = nullptr;
        std::vector<vk::Image> m_swapChainColorImages{};
        std::vector<vk::raii::ImageView> m_swapChainColorImageViews{};
        std::vector<vk::raii::Image> m_swapChainDepthImages{};
        std::vector<vk::raii::DeviceMemory> m_swapChainDepthImageMemories{};
        std::vector<vk::raii::ImageView> m_swapChainDepthImageViews{};
        uint8_t m_maxFramesInFlight = 0;
        vk::raii::CommandPool m_commandPool = nullptr;

        std::unique_ptr<RenderGraph> m_renderGraph = nullptr;
    };
}