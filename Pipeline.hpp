#pragma once

#include "RHI.hpp"

namespace Gfx
{
	class Pipeline
	{
	public:
		enum class Type
		{
			Graphics,
			Compute,
		};

	private:
		friend class RHI;

		Pipeline(Type type, vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& pipelineLayout, vk::raii::DescriptorSetLayout&& descriptorSetLayout);

	public:
		Pipeline(nullptr_t) :
			m_type(Type::Graphics),
			m_pipeline(nullptr),
			m_pipelineLayout(nullptr),
			m_descriptorSetLayout(nullptr)
		{
		}

		Pipeline() = delete;

		operator vk::Pipeline() const { return *m_pipeline; }
		vk::Pipeline operator*() const { return *m_pipeline; }

		const Type& getType() const { return m_type; }
		const vk::raii::PipelineLayout& getPipelineLayout() const { return m_pipelineLayout; }
		const vk::raii::DescriptorSetLayout& getDescriptorSetLayout() const { return m_descriptorSetLayout; }

	private:
		Type m_type;
		vk::raii::Pipeline m_pipeline;
		vk::raii::PipelineLayout m_pipelineLayout;
		vk::raii::DescriptorSetLayout m_descriptorSetLayout;
	};
}
