#pragma once

#include "RHI.hpp"

namespace Gfx
{
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

	struct PipelineCreateInfo
	{
		std::vector<ShaderDesc> shaders;
		std::vector<vk::VertexInputBindingDescription> vertexInputBindings;
		std::vector<vk::VertexInputAttributeDescription> vertexInputAttributes;
		std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
		std::vector<ColorAttachmentDesc> colorAttachments;
		DepthAttachmentDesc depthAttachment;
	};

	class Pipeline
	{
	private:
		friend class RHI;

		Pipeline(vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& pipelineLayout, vk::raii::DescriptorSetLayout&& descriptorSetLayout);

	public:
		Pipeline(nullptr_t) :
			m_pipeline(nullptr),
			m_pipelineLayout(nullptr),
			m_descriptorSetLayout(nullptr)
		{
		}

		Pipeline() = delete;

		operator vk::Pipeline() const { return *m_pipeline; }
		vk::Pipeline operator*() const { return *m_pipeline; }

		const vk::raii::PipelineLayout& getPipelineLayout() const { return m_pipelineLayout; }
		const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return m_descriptorSetLayout; }

	private:
		vk::raii::Pipeline m_pipeline;
		vk::raii::PipelineLayout m_pipelineLayout;
		vk::raii::DescriptorSetLayout m_descriptorSetLayout;
	};
}
