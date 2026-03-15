#pragma once

#include "RenderGraph.hpp"

class Gfx
{
public:
    class Buffer
    {
    private:
		friend class Gfx;

        Buffer(vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory, vk::DeviceSize size)
			: m_buffer(std::move(buffer)), m_bufferMemory(std::move(bufferMemory)), m_size(size)
        {}

    public:
		Buffer(nullptr_t) {}

		Buffer() = delete;

		operator vk::Buffer () const { return *m_buffer; }
		vk::Buffer operator*() const { return *m_buffer; }

		void* map() const { return m_bufferMemory.mapMemory(0, m_size); }
		void unmap() const { m_bufferMemory.unmapMemory(); }

    private:
		vk::DeviceSize m_size = 0;
        vk::raii::Buffer m_buffer = nullptr;
		// TODO: replace with arena allocator
        vk::raii::DeviceMemory m_bufferMemory = nullptr;
    };

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
};

