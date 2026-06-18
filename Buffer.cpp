#include "Buffer.hpp"

using Gfx::Buffer;

Buffer::Buffer(const vk::BufferCreateInfo& createInfo, vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory):
	m_createInfo(std::move(createInfo)),
	m_buffer(std::move(buffer)), 
	m_bufferMemory(std::move(bufferMemory))
{
}

void Buffer::map() {
	m_mappedData = m_bufferMemory.mapMemory(0, m_createInfo.size);
}

void Buffer::unmap() {
	m_bufferMemory.unmapMemory();
	m_mappedData = nullptr;
}