use ash::vk;
use bytemuck::NoUninit;
use gpu_allocator::MemoryLocation;
use scope_guard::scope_guard;
use std::sync::Arc;
use triangle_based_rendering::{
    Buffer, Device, Instance, RenderResult, RenderSync, ResourceToDestroy, Surface, Swapchain,
    transition_image,
};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowAttributes,
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let window = {
        let attributes = WindowAttributes::default().with_title("NonEuclidean Renderer");
        #[expect(deprecated)]
        event_loop.create_window(attributes).unwrap()
    };

    let entry = unsafe { ash::Entry::load() }.unwrap();

    let instance = Arc::new(unsafe { Instance::new(entry, None) });
    let surface = Arc::new(Surface::new(instance.clone(), &window));

    let device = Arc::new(Device::new(instance.clone()));
    let mut swapchain = Swapchain::new(device.clone(), surface);

    let mut buffer = Buffer::new(
        device.clone(),
        "Test Buffer",
        MemoryLocation::CpuToGpu,
        128,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    );

    {
        let floats = bytemuck::cast_slice_mut::<u8, f32>(unsafe { buffer.get_mapped_mut() }.unwrap());
        floats[0] = 0.5;
    }

    let shader_create_info = vk::ShaderModuleCreateInfo::default().code(
        const {
            #[repr(C)]
            struct Aligned<T: ?Sized> {
                align: [u32; 0],
                bytes: T,
            }

            const BYTES: &Aligned<[u8]> = &Aligned {
                align: [],
                bytes: *include_bytes!(concat!(env!("OUT_DIR"), "/shaders/full_screen_quad.spv")),
            };

            assert!(BYTES.bytes.len().is_multiple_of(4));
            unsafe {
                core::slice::from_raw_parts(
                    BYTES.bytes.as_ptr().cast::<u32>(),
                    BYTES.bytes.len() / 4,
                )
            }
        },
    );
    let shader = scope_guard!(
        |shader| unsafe {
            device.schedule_destroy_resource(
                device.current_timeline_counter(),
                ResourceToDestroy::ShaderModule(shader),
            );
        },
        unsafe { device.create_shader_module(&shader_create_info, device.allocator()) }.unwrap()
    );

    #[derive(Clone, Copy, NoUninit)]
    #[repr(C)]
    struct PushConstants {
        buffer: vk::DeviceAddress,
    }

    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(size_of::<PushConstants>() as _);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(core::slice::from_ref(&push_constant_range));

    let pipeline_layout = scope_guard!(
        |pipeline_layout| unsafe {
            device.schedule_destroy_resource(
                device.current_timeline_counter(),
                ResourceToDestroy::PipelineLayout(pipeline_layout),
            );
        },
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, device.allocator()) }
            .unwrap()
    );

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(*shader)
            .name(c"vertex"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(*shader)
            .name(c"fragment"),
    ];
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let mut rendering_create_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(&[vk::Format::B8G8R8A8_UNORM]);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(core::slice::from_ref(&blend_attachment));
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
        .push_next(&mut rendering_create_info)
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .layout(*pipeline_layout);

    let pipline = scope_guard!(
        |pipeline| unsafe {
            device.schedule_destroy_resource(
                device.current_timeline_counter(),
                ResourceToDestroy::Pipeline(pipeline),
            );
        },
        unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[pipeline_create_info],
                device.allocator(),
            )
        }
        .unwrap()[0]
    );

    let render = |command_buffer: vk::CommandBuffer,
                  image_layout: &mut vk::ImageLayout,
                  width: u32,
                  height: u32,
                  image: vk::Image,
                  image_view: vk::ImageView,
                  #[expect(unused)] frame_index: usize| {
        unsafe {
            transition_image(
                &device,
                command_buffer,
                image,
                image_layout,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );
        }

        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(image_view)
            .image_layout(*image_layout)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [1.0, 0.0, 1.0, 1.0],
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D { width, height },
            })
            .layer_count(1)
            .color_attachments(core::slice::from_ref(&color_attachment_info));
        unsafe { device.cmd_begin_rendering(command_buffer, &rendering_info) };

        let viewport = vk::Viewport::default()
            .x(0.0)
            .y(height as f32)
            .width(width as _)
            .height(-(height as f32));
        unsafe { device.cmd_set_viewport(command_buffer, 0, &[viewport]) };

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
        };
        unsafe { device.cmd_set_scissor(command_buffer, 0, &[scissor]) };

        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, *pipline);
            device.cmd_push_constants(
                command_buffer,
                *pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&PushConstants {
                    buffer: buffer.device_address(),
                }),
            );
            device.cmd_draw(command_buffer, 4, 1, 0, 0);
        }

        unsafe { device.cmd_end_rendering(command_buffer) };

        RenderSync {
            wait_sempahore_info: None,
            signal_sempahore_info: None,
        }
    };

    let run = |event: Event<()>, event_loop: &ActiveEventLoop| match event {
        Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => event_loop.exit(),

            WindowEvent::Resized(size) => {
                device.destroy_resources();

                swapchain.resize(size.width, size.height);
                swapchain.try_next_frame(render);
            }

            _ => {}
        },

        Event::AboutToWait => {
            device.destroy_resources();

            match swapchain.try_next_frame(render) {
                RenderResult::NotReady => {}
                RenderResult::OutOfDate | RenderResult::Suboptimal => {
                    let size = window.inner_size();
                    swapchain.resize(size.width, size.height);
                }
                RenderResult::Success => {}
            }
        }

        _ => {}
    };
    #[expect(deprecated)]
    event_loop.run(run).unwrap();
}
