#include "Image.hpp"

using Gfx::Image;

Image::Image(const vk::ImageCreateInfo& createInfo, vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::raii::ImageView&& imageView):
    m_createInfo(createInfo),
    m_image(std::move(image)),
    m_bufferMemory(std::move(bufferMemory)),
	m_imageView(std::move(imageView))
{
}