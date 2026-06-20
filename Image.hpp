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
        const vk::Image& getImage(int index) const { return *m_images[index]; }
        size_t getImageCount() const { return m_images.size(); }

    private:
        vk::ImageCreateInfo m_createInfo;
        std::vector<vk::raii::Image> m_images;
        std::vector<vk::raii::DeviceMemory> m_imageMemories;
        std::vector<vk::raii::ImageView> m_imageViews;
    };

    class Sampler
    {
    private:
        friend class RHI;

        Sampler(const vk::SamplerCreateInfo& createInfo, vk::raii::Sampler&& sampler);

    public:
        Sampler(nullptr_t) : m_sampler(nullptr) {}

        Sampler() = delete;

        const vk::SamplerCreateInfo& getCreateInfo() const { return m_createInfo; }
        const vk::Sampler& getSampler() const { return *m_sampler; }

    private:
        vk::SamplerCreateInfo m_createInfo;
        vk::raii::Sampler m_sampler;
    };
}