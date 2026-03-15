#include "Buffer.hpp"

void* Gfx::Buffer::map() const { 
	return m_bufferMemory.mapMemory(0, m_size); 
}

void Gfx::Buffer::unmap() const { 
	m_bufferMemory.unmapMemory(); 
}
