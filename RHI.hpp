#pragma once

#include <optional>
#include <variant>
#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
	class Buffer;
	class DescriptorSet;
	class Image;
	class Pipeline;

	struct GraphicsPipelineCreateInfo
	{
		std::vector<std::pair<std::string, vk::ShaderStageFlagBits>> shaders;
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings;
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
		std::vector<vk::Format> colorAttachments;
		std::optional<vk::Format> depthAttachment;
	};

	struct ComputePipelineCreateInfo
	{
		std::string shader;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
	};

	struct DescriptorBinding
	{
		vk::DescriptorType type;
		std::variant<
			std::vector<vk::DescriptorBufferInfo>,
			std::vector<std::vector<vk::DescriptorImageInfo>>
		> data;
	};

	struct DescriptorSetConfig
	{
		vk::DescriptorSetLayout layout;
		std::vector<DescriptorBinding> bindings;
	};

	class RHI
	{
	public:
		RHI() = default;
		RHI(const RHI&) = delete;

		void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

		const vk::raii::PhysicalDevice& getPhysicalDevice() const { return m_physicalDevice; }
		const vk::raii::Device& getDevice() const { return m_device; }
		uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
		const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
		vk::Format getSurfaceFormat() const { return m_surfaceFormat.format; }
		vk::Format getDepthFormat() const { return m_depthFormat; }
		std::vector<vk::Image> getSwapChainImages() const { return m_swapChain.getImages(); }
		std::vector<vk::Image> getDepthImages() const { return m_depthImageObjs; }
		const vk::raii::ImageView& getSwapChainImageView(int index) const { return m_swapChainImageViews[index]; }
		const vk::raii::ImageView& getDepthImageView(int index) const;
		vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }

		std::pair<vk::Result, uint32_t> acquireNextSwapChainImage(const vk::Semaphore& signal) const { return m_swapChain.acquireNextImage(UINT64_MAX, signal, nullptr); }

		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const void* contentData, size_t contentSize, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateBuffer(const Buffer& buffer, const void* contentData, size_t contentSize);

		Image createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateImage(const Gfx::Image& image, const void* contentData, size_t contentSize);

		Pipeline createGraphicsPipeline(const GraphicsPipelineCreateInfo& createInfo);
		Pipeline createComputePipeline(const ComputePipelineCreateInfo& createInfo);

		std::vector<std::vector<DescriptorSet>> createDescriptorSets(const std::vector<DescriptorSetConfig>& configs);

		template<int S>
		std::array<std::vector<DescriptorSet>, S> createDescriptorSets(const std::array<DescriptorSetConfig, S>& configs)
		{
			auto sets = createDescriptorSets(std::vector<DescriptorSetConfig>(configs.begin(), configs.end()));
			std::array<std::vector<DescriptorSet>, S> result{};
			std::move(sets.begin(), sets.end(), result.begin());
			return result;
		}

		template<typename T>
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const T& data, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
		    return createBuffer(bufferInfo, &data, sizeof(T), memProperties);
		}

		template<typename T>
		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, const std::vector<T>& data, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal) {
		    return createBuffer(bufferInfo, data.data(), data.size() * sizeof(T), memProperties);
		}

		template<typename T>
		void updateBuffer(const Buffer& buffer, const T& data) {
			updateBuffer(buffer, &data, sizeof(T));
		}

		template<typename T>
		void updateBuffer(const Buffer& buffer, const std::vector<T>& data) {
			updateBuffer(buffer, data.data(), data.size() * sizeof(T));
		}

		void updateImage(const Image& image, const std::vector<uint8_t>& data) {
			updateImage(image, data.data(), data.size());
		}

		void presentSwapChainImage(uint32_t imageIndex, const vk::SubmitInfo& submitInfo, const vk::Fence& inFlightFence) const;

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
		std::vector<Gfx::Image> m_depthImages{};
		std::vector<vk::Image> m_depthImageObjs{};
		vk::raii::CommandPool m_commandPool = nullptr;
	};
}
