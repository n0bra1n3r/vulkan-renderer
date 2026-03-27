#pragma once

#include "RHI.hpp"

namespace Gfx
{
    class Image
    {
    private:
        friend class RHI;

        Image(vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::Extent3D extent, vk::Format format);

    public:
        Image(nullptr_t):
            m_image(nullptr), 
            m_bufferMemory(nullptr), 
            m_extent({0, 0, 0}), 
			m_format(vk::Format::eUndefined)
        {}

        Image() = delete;

        operator vk::Image() const { return *m_image; }
        vk::Image operator*() const { return *m_image; }

    private:
        vk::raii::Image m_image;
        vk::raii::DeviceMemory m_bufferMemory;
        vk::Extent3D m_extent;
        vk::Format m_format;
    };
}