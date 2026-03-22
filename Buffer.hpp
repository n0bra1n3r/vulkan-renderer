#pragma once

#include <vulkan/vulkan_raii.hpp>

#include "Api.hpp"

namespace Gfx
{
    class Buffer
    {
    private:
        friend class Api;

        Buffer(vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory, vk::DeviceSize size)
            : m_buffer(std::move(buffer)), m_bufferMemory(std::move(bufferMemory)), m_size(size)
        {
        }

    public:
        Buffer(nullptr_t) {}

        Buffer() = delete;

        operator vk::Buffer() const { return *m_buffer; }
        vk::Buffer operator*() const { return *m_buffer; }

        void* map() const;
        void unmap() const;

    private:
        vk::DeviceSize m_size = 0;
        vk::raii::Buffer m_buffer = nullptr;
        // TODO: replace with arena allocator
        vk::raii::DeviceMemory m_bufferMemory = nullptr;
    };
}