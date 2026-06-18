#pragma once

#include <vulkan/vulkan_raii.hpp>

namespace Gfx
{
    class Image
    {
    private:
        friend class RHI;

        Image(
            const vk::ImageCreateInfo& createInfo, 
            std::vector<vk::raii::Image>&& images, 
            std::vector<vk::raii::DeviceMemory>&& imageMemories, 
            std::vector<vk::raii::ImageView>&& imageViews);

    public:
        Image(nullptr_t) {}

        Image() = delete;

        const vk::ImageCreateInfo& getCreateInfo() const { return m_createInfo; }
		const vk::ImageView& getImageView(int index) const { return *m_imageViews[index]; }
        const std::vector<vk::Image>& getImages() const { return m_rawImages; }

    private:
        vk::ImageCreateInfo m_createInfo;
        std::vector<vk::raii::Image> m_images;
        std::vector<vk::raii::DeviceMemory> m_imageMemories;
        std::vector<vk::raii::ImageView> m_imageViews;
        std::vector<vk::Image> m_rawImages;
    };
}