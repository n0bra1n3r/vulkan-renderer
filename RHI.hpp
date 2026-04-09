#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
	class Buffer;
	class Image;
	class Pipeline;

	struct PipelineCreateInfo;

	class RHI
	{
	public:
		RHI() = default;
		RHI(const RHI&) = delete;

		void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

		const vk::raii::PhysicalDevice& getPhysicalDevice() const { return m_physicalDevice; }
		const vk::raii::Device& getDevice() const { return m_device; }
		const vk::raii::SwapchainKHR& getSwapChain() const { return m_swapChain; }
		const vk::raii::Queue& getGraphicsQueue() const { return m_graphicsQueue; }
		const vk::raii::Queue& getPresentQueue() const { return m_presentQueue; }
		uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
		const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
		vk::Format getSurfaceFormat() const { return m_surfaceFormat.format; }
		vk::Format getDepthFormat() const { return m_depthFormat; }
		const std::vector<vk::Image>& getDepthImages() const { return m_depthImages; }
		const vk::raii::ImageView& getSwapChainImageView(int index) const { return m_swapChainImageViews[index]; }
		const vk::raii::ImageView& getDepthImageView(int index) const { return m_depthImageViews[index]; }
		vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }

		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateBuffer(const Buffer& buffer, void* contentData, size_t contentSize);

		Image createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateImage(const Gfx::Image& image, void* contentData, size_t contentSize);

		Pipeline createGraphicsPipeline(const PipelineCreateInfo& createInfo);

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
		void initInstance(const std::string& appName, const std::vector<const char*>& extensions);
		void initSurface(void* window);
		void pickPhysicalDevice();
		void initLogicalDevice();
		void initSwapChain(void* window);
		void initDepthResources();
		void initCommandPool();

	private:
		vk::raii::Context m_context{};
		vk::raii::Instance m_instance = nullptr;
		vk::raii::SurfaceKHR m_surface = nullptr;
		vk::raii::PhysicalDevice m_physicalDevice = nullptr;
		vk::raii::Device m_device = nullptr;
		uint32_t m_graphicsFamily = 0;
		uint32_t m_presentFamily = 0;
		vk::raii::Queue m_graphicsQueue = nullptr;
		vk::raii::Queue m_presentQueue = nullptr;
		vk::SurfaceFormatKHR m_surfaceFormat{};
		vk::Extent2D m_swapChainExtent{};
		vk::raii::SwapchainKHR m_swapChain = nullptr;
		uint8_t m_maxFramesInFlight = 0;
		std::vector<vk::raii::ImageView> m_swapChainImageViews{};
		vk::Format m_depthFormat{};
		std::vector<vk::Image> m_depthImages{};
		std::vector<vk::raii::Image> m_depthImageRefs{};
		std::vector<vk::raii::DeviceMemory> m_depthImageMemories{};
		std::vector<vk::raii::ImageView> m_depthImageViews{};
		vk::raii::CommandPool m_commandPool = nullptr;
	};
}

