#include "Pipeline.hpp"

using Gfx::Pipeline;

Pipeline::Pipeline(Pipeline::Type type, vk::raii::Pipeline&& pipeline, vk::raii::PipelineLayout&& pipelineLayout, vk::raii::DescriptorSetLayout&& descriptorSetLayout):
	m_type(type),
	m_pipeline(std::move(pipeline)),
	m_pipelineLayout(std::move(pipelineLayout)),
	m_descriptorSetLayout(std::move(descriptorSetLayout))
{
}
