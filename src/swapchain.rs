use crate::{Device, Instance, Surface};
use ash::vk;
use scope_guard::scope_guard;
use std::{ops::Deref, sync::Arc};

pub const FRAMES_IN_FLIGHT_COUNT: usize = 2;

pub struct Swapchain<'allocator, 'window> {
    device: Arc<Device<'allocator>>,
    surface: Arc<Surface<'allocator, 'window>>,

    width: u32,
    height: u32,
    swapchain: vk::SwapchainKHR,
    swapchain_funcs: ash::khr::swapchain::Device,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,

    command_pool: vk::CommandPool,

    frame_counter: usize,
    aquired_image: [vk::Semaphore; FRAMES_IN_FLIGHT_COUNT],
    command_buffers: [vk::CommandBuffer; FRAMES_IN_FLIGHT_COUNT],
    render_finished: [vk::Semaphore; FRAMES_IN_FLIGHT_COUNT],
    render_finished_fences: [vk::Fence; FRAMES_IN_FLIGHT_COUNT],
    finished_presenting: [vk::Fence; FRAMES_IN_FLIGHT_COUNT],
}

impl<'allocator, 'window> Swapchain<'allocator, 'window> {
    pub fn new(
        device: Arc<Device<'allocator>>,
        surface: Arc<Surface<'allocator, 'window>>,
    ) -> Self {
        assert!(Arc::ptr_eq(device.instance(), surface.instance()));

        let swapchain_funcs = ash::khr::swapchain::Device::new(device.instance(), &device);

        let capabilities = unsafe {
            surface.get_physical_device_surface_capabilities(
                device.physical_device(),
                surface.handle(),
            )
        }
        .unwrap();

        let graphics_queue_family_index = device.graphics_queue_family_index();

        let width = capabilities.min_image_extent.width;
        let height = capabilities.min_image_extent.height;
        let swapchain_create_info = swapchain_create_info(
            surface.handle(),
            vk::Extent2D { width, height },
            &graphics_queue_family_index,
            vk::SwapchainKHR::null(),
        );

        let swapchain = scope_guard!(
            |swapchain| unsafe { swapchain_funcs.destroy_swapchain(swapchain, device.allocator()) },
            unsafe { swapchain_funcs.create_swapchain(&swapchain_create_info, device.allocator()) }
                .unwrap()
        );

        let images = unsafe { swapchain_funcs.get_swapchain_images(*swapchain) }.unwrap();

        let mut image_views = scope_guard!(
            |image_views| {
                for image_view in image_views {
                    unsafe { device.destroy_image_view(image_view, device.allocator()) };
                }
            },
            Vec::with_capacity(images.len())
        );
        for &image in &images {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_create_info.image_format)
                .components(vk::ComponentMapping::default())
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(vk::REMAINING_MIP_LEVELS)
                        .base_array_layer(0)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS),
                );

            let image_view =
                unsafe { device.create_image_view(&image_view_create_info, device.allocator()) }
                    .unwrap();
            image_views.push(image_view);
        }

        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(graphics_queue_family_index);

        let command_pool = scope_guard!(
            |command_pool| unsafe { device.destroy_command_pool(command_pool, device.allocator()) },
            unsafe { device.create_command_pool(&command_pool_create_info, device.allocator()) }
                .unwrap()
        );

        let aquired_image = scope_guard!(
            |aquired_image| {
                for semaphore in aquired_image {
                    unsafe { device.destroy_semaphore(semaphore, device.allocator()) };
                }
            },
            std::array::from_fn(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_create_info, device.allocator()) }
                    .unwrap()
            })
        );

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT_COUNT as _);
        let command_buffers =
            unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }
                .unwrap()
                .try_into()
                .unwrap();

        let render_finished = scope_guard!(
            |render_finished| {
                for semaphore in render_finished {
                    unsafe { device.destroy_semaphore(semaphore, device.allocator()) };
                }
            },
            std::array::from_fn(|_| {
                let semaphore_create_info = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_create_info, device.allocator()) }
                    .unwrap()
            })
        );

        let render_finished_fences = scope_guard!(
            |render_finished| {
                for fence in render_finished {
                    unsafe { device.destroy_fence(fence, device.allocator()) };
                }
            },
            std::array::from_fn(|_| {
                let fence_create_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_create_info, device.allocator()) }.unwrap()
            })
        );

        let finished_presenting = scope_guard!(
            |finished_presenting| {
                for fence in finished_presenting {
                    unsafe { device.destroy_fence(fence, device.allocator()) };
                }
            },
            std::array::from_fn(|_| {
                let fence_create_info =
                    vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
                unsafe { device.create_fence(&fence_create_info, device.allocator()) }.unwrap()
            })
        );

        Self {
            surface,

            width,
            height,
            swapchain: swapchain.into_inner(),
            swapchain_funcs,

            images,
            image_views: image_views.into_inner(),

            command_pool: command_pool.into_inner(),

            frame_counter: 0,
            aquired_image: aquired_image.into_inner(),
            command_buffers,
            render_finished: render_finished.into_inner(),
            render_finished_fences: render_finished_fences.into_inner(),
            finished_presenting: finished_presenting.into_inner(),

            device,
        }
    }

    pub fn instance(&self) -> &Arc<Instance<'allocator>> {
        self.device.instance()
    }

    pub fn allocator(&self) -> Option<&vk::AllocationCallbacks<'allocator>> {
        self.device.allocator()
    }

    pub fn device(&self) -> &Arc<Device<'allocator>> {
        &self.device
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.swapchain
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn resize(&mut self, mut width: u32, mut height: u32) {
        if width == 0 || height == 0 || (width == self.width && height == self.height) {
            return;
        }

        unsafe {
            self.device
                .wait_for_fences(&self.render_finished_fences, true, u64::MAX)
        }
        .unwrap();
        unsafe {
            self.device
                .wait_for_fences(&self.finished_presenting, true, u64::MAX)
        }
        .unwrap();

        let capabilities = unsafe {
            self.surface.get_physical_device_surface_capabilities(
                self.device.physical_device(),
                self.surface.handle(),
            )
        }
        .unwrap();

        let graphics_queue_family_index = self.device.graphics_queue_family_index();

        width = width.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );
        height = height.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );
        let swapchain_create_info = swapchain_create_info(
            self.surface.handle(),
            vk::Extent2D { width, height },
            &graphics_queue_family_index,
            self.swapchain,
        );

        let old_swapchain = core::mem::replace(
            &mut self.swapchain,
            unsafe {
                self.swapchain_funcs
                    .create_swapchain(&swapchain_create_info, self.device.allocator())
            }
            .unwrap(),
        );
        unsafe { self.destroy_swapchain(old_swapchain, self.allocator()) };

        self.width = width;
        self.height = height;

        self.images.clear();
        for image_view in self.image_views.drain(..) {
            unsafe {
                self.device
                    .destroy_image_view(image_view, self.device.allocator());
            }
        }

        self.images = unsafe { self.get_swapchain_images(self.swapchain) }.unwrap();
        for &image in &self.images {
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(swapchain_create_info.image_format)
                .components(vk::ComponentMapping::default())
                .subresource_range(make_subresource_range(vk::ImageAspectFlags::COLOR));

            let image_view = unsafe {
                self.device
                    .create_image_view(&image_view_create_info, self.device.allocator())
            }
            .unwrap();
            self.image_views.push(image_view);
        }
    }

    pub fn try_next_frame<'a>(
        &mut self,
        f: impl FnOnce(
            vk::CommandBuffer,
            &mut vk::ImageLayout,
            vk::Image,
            vk::ImageView,
            usize,
        ) -> RenderSync<'a>,
    ) -> RenderResult {
        let frame_index = self.frame_counter;

        match unsafe {
            self.device
                .wait_for_fences(&[self.render_finished_fences[frame_index]], true, 0)
        } {
            Err(vk::Result::TIMEOUT) => return RenderResult::NotReady,
            e => e.unwrap(),
        }
        match unsafe {
            self.device
                .wait_for_fences(&[self.finished_presenting[frame_index]], true, 0)
        } {
            Err(vk::Result::TIMEOUT) => return RenderResult::NotReady,
            e => e.unwrap(),
        }

        let (image_index, mut suboptimal) = match unsafe {
            self.acquire_next_image(
                self.swapchain,
                0,
                self.aquired_image[frame_index],
                vk::Fence::null(),
            )
        } {
            Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => return RenderResult::NotReady,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return RenderResult::OutOfDate,
            e => e.unwrap(),
        };

        self.frame_counter = (self.frame_counter + 1) % FRAMES_IN_FLIGHT_COUNT;

        unsafe {
            self.device.reset_command_buffer(
                self.command_buffers[frame_index],
                vk::CommandBufferResetFlags::empty(),
            )
        }
        .unwrap();

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe {
            self.device.begin_command_buffer(
                self.command_buffers[frame_index],
                &command_buffer_begin_info,
            )
        }
        .unwrap();

        let mut image_layout = vk::ImageLayout::UNDEFINED;
        let RenderSync {
            wait_sempahore_info: user_wait_semaphore_info,
            signal_sempahore_info: user_signal_semaphore_info,
        } = f(
            self.command_buffers[frame_index],
            &mut image_layout,
            self.images[image_index as usize],
            self.image_views[image_index as usize],
            frame_index,
        );

        unsafe {
            transition_image(
                &self.device,
                self.command_buffers[frame_index],
                self.images[image_index as usize],
                &mut image_layout,
                vk::ImageLayout::PRESENT_SRC_KHR,
            );
        }
        unsafe {
            self.device
                .end_command_buffer(self.command_buffers[frame_index])
        }
        .unwrap();

        {
            unsafe {
                self.device
                    .reset_fences(&[self.render_finished_fences[frame_index]])
            }
            .unwrap();

            let command_infos = [vk::CommandBufferSubmitInfo::default()
                .command_buffer(self.command_buffers[frame_index])];

            let acquire_wait_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.aquired_image[frame_index])
                .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT);
            let render_finished_signal_info = vk::SemaphoreSubmitInfo::default()
                .semaphore(self.render_finished[frame_index])
                .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS);
            let render_finished_timeline_signal_info = self.device.signal_timeline_submit_info();

            let wait_infos = match user_wait_semaphore_info {
                Some(user_wait_info) => &[acquire_wait_info, user_wait_info] as &[_],
                None => &[acquire_wait_info] as &[_],
            };
            let signal_infos = match user_signal_semaphore_info {
                Some(user_signal_info) => &[
                    render_finished_signal_info,
                    render_finished_timeline_signal_info,
                    user_signal_info,
                ] as &[_],
                None => &[
                    render_finished_signal_info,
                    render_finished_timeline_signal_info,
                ] as &[_],
            };

            self.device
                .with_graphics_queue(|graphics_queue| unsafe {
                    self.device.queue_submit2(
                        graphics_queue,
                        &[vk::SubmitInfo2::default()
                            .command_buffer_infos(&command_infos)
                            .wait_semaphore_infos(wait_infos)
                            .signal_semaphore_infos(signal_infos)],
                        self.render_finished_fences[frame_index],
                    )
                })
                .unwrap();
        }

        {
            unsafe {
                self.device
                    .reset_fences(&[self.finished_presenting[frame_index]])
            }
            .unwrap();

            let mut result = vk::Result::SUCCESS;
            let mut present_finished_fences = vk::SwapchainPresentFenceInfoEXT::default().fences(
                core::slice::from_ref(&self.finished_presenting[frame_index]),
            );
            let present_info = vk::PresentInfoKHR::default()
                .push_next(&mut present_finished_fences)
                .wait_semaphores(core::slice::from_ref(&self.render_finished[frame_index]))
                .swapchains(core::slice::from_ref(&self.swapchain))
                .image_indices(core::slice::from_ref(&image_index))
                .results(core::slice::from_mut(&mut result));

            suboptimal |= match self.device.with_graphics_queue(|graphics_queue| unsafe {
                self.queue_present(graphics_queue, &present_info)
            }) {
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return RenderResult::OutOfDate;
                }
                result => result.unwrap(),
            };
            result.result().unwrap();
        }

        if suboptimal {
            RenderResult::Suboptimal
        } else {
            RenderResult::Success
        }
    }
}

pub struct RenderSync<'a> {
    pub wait_sempahore_info: Option<vk::SemaphoreSubmitInfo<'a>>,
    pub signal_sempahore_info: Option<vk::SemaphoreSubmitInfo<'a>>,
}

pub enum RenderResult {
    NotReady,
    OutOfDate,
    Suboptimal,
    Success,
}

impl Deref for Swapchain<'_, '_> {
    type Target = ash::khr::swapchain::Device;

    fn deref(&self) -> &Self::Target {
        &self.swapchain_funcs
    }
}

impl Drop for Swapchain<'_, '_> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wait_for_fences(&self.render_finished_fences, true, u64::MAX)
        }
        .unwrap();
        unsafe {
            self.device
                .wait_for_fences(&self.finished_presenting, true, u64::MAX)
        }
        .unwrap();

        for &semaphore in &self.aquired_image {
            unsafe { self.device.destroy_semaphore(semaphore, self.allocator()) };
        }
        for &semaphore in &self.render_finished {
            unsafe { self.device.destroy_semaphore(semaphore, self.allocator()) };
        }
        for &fence in &self.render_finished_fences {
            unsafe { self.device.destroy_fence(fence, self.allocator()) };
        }
        for &fence in &self.finished_presenting {
            unsafe { self.device.destroy_fence(fence, self.allocator()) };
        }

        unsafe {
            self.device
                .destroy_command_pool(self.command_pool, self.allocator());
        }

        for &image_view in &self.image_views {
            unsafe { self.device.destroy_image_view(image_view, self.allocator()) };
        }

        unsafe { self.destroy_swapchain(self.swapchain, self.allocator()) };
    }
}

fn swapchain_create_info<'a>(
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    queue_family_index: &'a u32,
    old_swapchain: vk::SwapchainKHR,
) -> vk::SwapchainCreateInfoKHR<'a> {
    vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(3)
        .image_format(vk::Format::B8G8R8A8_UNORM)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .queue_family_indices(core::slice::from_ref(queue_family_index))
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::MAILBOX)
        .clipped(true)
        .old_swapchain(old_swapchain)
}

pub fn make_subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(vk::REMAINING_MIP_LEVELS)
        .base_array_layer(0)
        .layer_count(vk::REMAINING_ARRAY_LAYERS)
}

/// # Safety
/// See [Device::cmd_pipeline_barrier2](ash::device::Device::cmd_pipeline_barrier2)
pub unsafe fn transition_image(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    current_layout: &mut vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(*current_layout)
        .new_layout(new_layout)
        .subresource_range(make_subresource_range(
            if new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            },
        ))
        .image(image);

    let dependency_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&image_barrier));

    unsafe { device.cmd_pipeline_barrier2(command_buffer, &dependency_info) };
    *current_layout = new_layout;
}
