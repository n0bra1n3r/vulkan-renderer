#pragma once

#include "RenderGraph.hpp"

class Gfx
{
public:
	Gfx() = default;
    Gfx(const Gfx&) = delete;

	void init(const std::string& appName, const std::vector<const char*>& extensions, void* window);

	const vk::raii::Device& getDevice() const { return device; }
	const vk::raii::PhysicalDevice& getPhysicalDevice() const { return physicalDevice; }
	const vk::raii::Queue& getGraphicsQueue() const { return graphicsQueue; }
    vk::Format getSwapChainSurfaceFormat() const { return swapChainSurfaceFormat.format; }
    vk::Extent2D getSwapChainExtent() const { return swapChainExtent; }
	uint8_t getMaxFramesInFlight() const { return maxFramesInFlight; }
	const vk::raii::ImageView& getSwapChainImageView(size_t index) const { return swapChainImageViews[index]; }
	const vk::raii::CommandPool& getCommandPool() const { return commandPool; }
	const std::unique_ptr<RenderGraph>& getRenderGraph() const { return renderGraph; }

private:
	void createInstance(const std::string& appName, const std::vector<const char*>& extensions);
    void createSurface(void* window);
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain(void* window);
    void createImageViews();
    void createCommandPool();

    void initRenderGraph();

private:
    vk::raii::Context context{};
    vk::raii::Instance instance = nullptr;
    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device device = nullptr;
    uint32_t graphicsFamily = 0, presentFamily = 0;
    vk::raii::Queue graphicsQueue = nullptr;
    vk::raii::Queue presentQueue = nullptr;
    vk::raii::SurfaceKHR surface = nullptr;
    vk::SurfaceFormatKHR swapChainSurfaceFormat{};
    vk::Extent2D swapChainExtent{};
    vk::raii::SwapchainKHR swapChain = nullptr;
    std::vector<vk::Image> swapChainImages{};
    std::vector<vk::raii::ImageView> swapChainImageViews{};
	uint8_t maxFramesInFlight;
    vk::raii::CommandPool commandPool = nullptr;

    std::unique_ptr<RenderGraph> renderGraph;
};

