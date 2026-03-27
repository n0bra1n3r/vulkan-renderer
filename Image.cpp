#include "Image.hpp"

using Gfx::Image;

Image::Image(vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::Extent3D extent, vk::Format format):
    m_image(std::move(image)),
    m_bufferMemory(std::move(bufferMemory)),
    m_extent(extent),
    m_format(format)
{
}