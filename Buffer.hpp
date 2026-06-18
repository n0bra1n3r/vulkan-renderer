#pragma once

#include "RHI.hpp"

namespace Gfx
{
	class Buffer
	{
    private:
        friend class RHI;

        Buffer(const vk::BufferCreateInfo& createInfo, vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory);

    public:
        Buffer(nullptr_t):
            m_createInfo({}),
			m_buffer(nullptr), 
            m_bufferMemory(nullptr)
        {}

        Buffer() = delete;

        operator vk::Buffer() const { return *m_buffer; }
        vk::Buffer operator*() const { return *m_buffer; }

        const vk::BufferCreateInfo& getCreateInfo() const { return m_createInfo; }

        void map();
        void unmap();
		void* getMappedData() const { return m_mappedData; }

    private:
        vk::BufferCreateInfo m_createInfo;
        vk::raii::Buffer m_buffer;
        vk::raii::DeviceMemory m_bufferMemory;
		void* m_mappedData = nullptr;
    };
}