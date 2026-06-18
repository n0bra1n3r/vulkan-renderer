#include "Image.hpp"

using Gfx::Image;

Image::Image(const vk::ImageCreateInfo& createInfo, std::vector<vk::raii::Image>&& images, std::vector<vk::raii::DeviceMemory>&& imageMemories, std::vector<vk::raii::ImageView>&& imageViews):
    m_createInfo(createInfo),
    m_images(std::move(images)),
    m_imageMemories(std::move(imageMemories)),
	m_imageViews(std::move(imageViews))
{
    m_rawImages.reserve(m_images.size());

    for (const auto& image : m_images)
    {
        m_rawImages.emplace_back(*image);
    }
}