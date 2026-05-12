#pragma once

#include <variant>
#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
	class Buffer;
	class DescriptorSet;
	class Image;
	class Pipeline;

	struct ShaderDesc
	{
		std::string path;
		vk::ShaderStageFlagBits stage;
	};

	struct ColorAttachmentDesc
	{
		vk::Format format;
		vk::ColorComponentFlags writeMask =
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA;
	};

	struct DepthAttachmentDesc
	{
		vk::Format format;
	};

	struct GraphicsPipelineCreateInfo
	{
		std::vector<ShaderDesc> shaders;
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings;
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
		std::vector<ColorAttachmentDesc> colorAttachments;
		DepthAttachmentDesc depthAttachment;
	};

	struct ComputePipelineCreateInfo
	{
		ShaderDesc shader;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
	};

	// Describes a single descriptor binding within a DescriptorSetConfig.
	// data holds either:
	//   - vector<DescriptorBufferInfo>: one entry per frame, or a single entry reused for all frames
	//   - vector<vector<DescriptorImageInfo>>: one inner vector per frame, or a single entry reused for all frames
	struct DescriptorBinding
	{
		uint32_t binding;
		vk::DescriptorType type;
		std::variant<
			std::vector<vk::DescriptorBufferInfo>,             // buffer data, indexed by frame
			std::vector<std::vector<vk::DescriptorImageInfo>>  // image data, indexed by frame
		> data;
	};

	// Describes all descriptor sets to allocate for one pipeline stage.
	struct DescriptorSetConfig
	{
		vk::DescriptorSetLayout layout; // the layout to allocate against
		uint32_t setCount;              // number of sets to allocate (typically maxFramesInFlight)
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
		const vk::raii::SwapchainKHR& getSwapChain() const { return m_swapChain; }
		const vk::raii::Queue& getGraphicsQueue() const { return m_graphicsQueue; }
		const vk::raii::Queue& getPresentQueue() const { return m_presentQueue; }
		uint8_t getMaxFramesInFlight() const { return m_maxFramesInFlight; }
		const vk::raii::CommandPool& getCommandPool() const { return m_commandPool; }
		vk::Format getSurfaceFormat() const { return m_surfaceFormat.format; }
		vk::Format getDepthFormat() const { return m_depthFormat; }
		const std::vector<vk::Image>& getDepthImages() const { return m_depthImageObjs; }
		const vk::raii::ImageView& getSwapChainImageView(int index) const { return m_swapChainImageViews[index]; }
		const vk::raii::ImageView& getDepthImageView(int index) const;
		vk::Extent2D getSwapChainExtent() const { return m_swapChainExtent; }

		Buffer createBuffer(const vk::BufferCreateInfo& bufferInfo, vk::MemoryPropertyFlags memProperties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateBuffer(const Buffer& buffer, const void* contentData, size_t contentSize);

		Image createImage(const vk::ImageCreateInfo& imageInfo, vk::MemoryPropertyFlags properties = vk::MemoryPropertyFlagBits::eDeviceLocal);
		void updateImage(const Gfx::Image& image, const void* contentData, size_t contentSize);

		Pipeline createGraphicsPipeline(const GraphicsPipelineCreateInfo& createInfo);
		Pipeline createComputePipeline(const ComputePipelineCreateInfo& createInfo);

		std::vector<std::vector<DescriptorSet>> createDescriptorSets(const std::vector<DescriptorSetConfig>& configs);

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
