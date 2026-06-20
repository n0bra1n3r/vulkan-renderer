#include "Image.hpp"

using Gfx::Image;
using Gfx::Sampler;

Image::Image(
    const vk::ImageCreateInfo& createInfo, 
    std::vector<vk::raii::Image>&& images, 
    std::vector<vk::raii::DeviceMemory>&& imageMemories, 
    std::vector<vk::raii::ImageView>&& imageViews):
    m_createInfo(createInfo),
    m_images(std::move(images)),
    m_imageMemories(std::move(imageMemories)),
	m_imageViews(std::move(imageViews))
{}

Sampler::Sampler(const vk::SamplerCreateInfo& createInfo, vk::raii::Sampler&& sampler):
    m_sampler(std::move(sampler))
{}