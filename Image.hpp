#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "Api.hpp"

namespace Gfx
{
    class Image
    {
    private:
        friend class Api;

        Image(vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::Extent3D extent, vk::Format format)
            : m_image(std::move(image)), m_bufferMemory(std::move(bufferMemory)), m_extent(extent), m_format(format)
        {
        }

    public:
        Image(nullptr_t) {}

        Image() = delete;

        operator vk::Image() const { return *m_image; }
        vk::Image operator*() const { return *m_image; }

    private:
        vk::Extent3D m_extent{};
        vk::Format m_format{};
        vk::raii::Image m_image = nullptr;
        // TODO: replace with arena allocator
        vk::raii::DeviceMemory m_bufferMemory = nullptr;
    };
}