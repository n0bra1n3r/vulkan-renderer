#pragma once

#include "RenderGraph.hpp"

class Gfx
{
public:
    class Buffer;
    class Image;

public:
	Gfx() = default;
    Gfx(const Gfx&) = delete;

	void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

	const vk::raii::Device& getDevice() const { return m_device; }
	const vk::raii::PhysicalDevice& getPhysicalDevice() const { return m_physicalDevice; }
	const vk::raii::Queue& getGraphicsQueue() const { return m_graphicsQueue; }
    vk::Format getSwapChainSurfaceFormat() const { return m_swapChainSurfaceFormat.format; }
    vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }
	uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
	const vk::raii::ImageView& getSwapChainImageView(size_t index) const { return m_swapChainImageViews[index]; }
	const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
	const std::unique_ptr<RenderGraph>& getRenderGraph() const { return m_renderGraph; }

    Buffer makeBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties);
    void updateBuffer(const Buffer& buffer, void* contentData, size_t contentSize);

    Image makeImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties);
    void updateImage(const Gfx::Image& image, void* contentData, size_t contentSize);

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
    vk::Extent2D m_swapChainExtent{};
    vk::raii::SwapchainKHR m_swapChain = nullptr;
    std::vector<vk::Image> m_swapChainImages{};
    std::vector<vk::raii::ImageView> m_swapChainImageViews{};
	uint8_t m_maxFramesInFlight = 0;
    vk::raii::CommandPool m_commandPool = nullptr;

    std::unique_ptr<RenderGraph> m_renderGraph = nullptr;

public:
    class Buffer
    {
    private:
        friend class Gfx;

        Buffer(vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory, vk::DeviceSize size)
            : m_buffer(std::move(buffer)), m_bufferMemory(std::move(bufferMemory)), m_size(size)
        {
        }

    public:
        Buffer(nullptr_t) {}

        Buffer() = delete;

        operator vk::Buffer() const { return *m_buffer; }
        vk::Buffer operator*() const { return *m_buffer; }

        void* map() const { return m_bufferMemory.mapMemory(0, m_size); }
        void unmap() const { m_bufferMemory.unmapMemory(); }

    private:
        vk::DeviceSize m_size = 0;
        vk::raii::Buffer m_buffer = nullptr;
        // TODO: replace with arena allocator
        vk::raii::DeviceMemory m_bufferMemory = nullptr;
    };

    class Image
    {
    private:
        friend class Gfx;

        Image(vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::Extent3D extent)
			: m_image(std::move(image)), m_bufferMemory(std::move(bufferMemory)), m_extent(extent)
        {
        }

    public:
        Image(nullptr_t) {}

        Image() = delete;

        operator vk::Image() const { return *m_image; }
        vk::Image operator*() const { return *m_image; }

    private:
        vk::Extent3D m_extent{};
        vk::raii::Image m_image = nullptr;
        // TODO: replace with arena allocator
        vk::raii::DeviceMemory m_bufferMemory = nullptr;
    };
};
