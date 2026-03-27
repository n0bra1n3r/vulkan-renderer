#include "Buffer.hpp"

using Gfx::Buffer;

Buffer::Buffer(vk::raii::Buffer&& buffer, vk::raii::DeviceMemory&& bufferMemory, vk::DeviceSize size): 
	m_buffer(std::move(buffer)), 
	m_bufferMemory(std::move(bufferMemory)), 
	m_size(size)
{
}

void* Buffer::map() const {
	return m_bufferMemory.mapMemory(0, m_size);
}

void Buffer::unmap() const {
	m_bufferMemory.unmapMemory();
}