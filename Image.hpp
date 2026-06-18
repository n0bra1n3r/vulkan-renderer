#pragma once

#include "RHI.hpp"

namespace Gfx
{
    class Image
    {
    private:
        friend class RHI;

        Image(const vk::ImageCreateInfo& createInfo, vk::raii::Image&& image, vk::raii::DeviceMemory&& bufferMemory, vk::raii::ImageView&& imageView);

    public:
        Image(nullptr_t):
            m_createInfo({}),
            m_image(nullptr), 
            m_bufferMemory(nullptr), 
			m_imageView(nullptr)
        {}

        Image() = delete;

        operator vk::Image() const { return *m_image; }
        vk::Image operator*() const { return *m_image; }

        const vk::ImageCreateInfo& getCreateInfo() const { return m_createInfo; }
		const vk::raii::ImageView& getImageView() const { return m_imageView; }

    private:
        vk::ImageCreateInfo m_createInfo;
        vk::raii::Image m_image;
        vk::raii::DeviceMemory m_bufferMemory;
        vk::raii::ImageView m_imageView;
    };
}