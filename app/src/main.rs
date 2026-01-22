use ash::vk;
use bytemuck::NoUninit;
use gpu_allocator::MemoryLocation;
use rendering::{
    Buffer, Device, Instance, RenderResult, RenderSync, ResourceToDestroy, Shader, Surface,
    Swapchain, include_spirv, transition_image,
};
use scope_guard::scope_guard;
use std::{sync::Arc, time::Instant};
use winit::{
    event::{Event, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::WindowAttributes,
};

#[derive(Clone, Copy, NoUninit)]
#[repr(C)]
struct Triangle {
    // ax is 0
    // ay is 0
    bx: f32,
    // by is 0
    cx: f32,
    cy: f32,

    _padding1: u32,

    edge_triangles: [u32; 3],
    edge_indices: [u8; 3],

    _padding2: u8,
}

#[derive(Clone, Copy, NoUninit)]
#[repr(C)]
struct Position {
    offset_x: f32,
    offset_y: f32,
    triangle_index: u32,
}

#[derive(Clone, Copy, NoUninit)]
#[repr(C)]
struct PushConstants {
    triangles: vk::DeviceAddress,
    start_position: Position,
    aspect: f32,
}

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

    let triangles = [
        Triangle {
            bx: 2.0,
            cx: 1.0,
            cy: 2.0,

            edge_triangles: [1, 1, 1],
            edge_indices: [0, 1, 2],

            _padding1: 0,
            _padding2: 0,
        },
        Triangle {
            bx: 2.0,
            cx: 1.0,
            cy: 2.0,

            edge_triangles: [0, 0, 0],
            edge_indices: [0, 1, 2],

            _padding1: 0,
            _padding2: 0,
        },
    ];

    let mut triangles_buffer = Buffer::new(
        device.clone(),
        "Triangles Buffer",
        MemoryLocation::CpuToGpu,
        size_of_val::<[_]>(&triangles) as _,
        vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        false,
    );

    {
        let triangles_buffer = unsafe { triangles_buffer.get_mapped_mut() }.unwrap();
        triangles_buffer.copy_from_slice(bytemuck::cast_slice(&triangles));
    }

    let shader = unsafe {
        Shader::new(
            device.clone(),
            include_spirv!(concat!(env!("OUT_DIR"), "/shaders/full_screen_quad.spv")),
        )
    };

    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
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
            .module(shader.handle())
            .name(c"vertex"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader.handle())
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

    let pipeline = scope_guard!(
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

    drop(shader);

    let mut position = Position {
        offset_x: 0.5,
        offset_y: 0.5,
        triangle_index: 0,
    };

    let mut last_time = Instant::now();
    let mut dt = 0.0;
    let mut w_pressed = false;
    let mut s_pressed = false;
    let mut a_pressed = false;
    let mut d_pressed = false;
    let run = |event: Event<()>, event_loop: &ActiveEventLoop| match event {
        Event::NewEvents(_) => {
            let time = Instant::now();
            dt = (time - last_time).as_secs_f32();
            last_time = time;
        }

        Event::WindowEvent { window_id, event } if window_id == window.id() => match event {
            WindowEvent::CloseRequested | WindowEvent::Destroyed => event_loop.exit(),

            WindowEvent::Resized(size) => {
                device.destroy_resources();

                swapchain.resize(size.width, size.height);
                swapchain.try_next_frame(
                    |command_buffer: vk::CommandBuffer,
                     image_layout: &mut vk::ImageLayout,
                     width: u32,
                     height: u32,
                     image: vk::Image,
                     image_view: vk::ImageView,
                     frame_index: usize| {
                        unsafe {
                            render(
                                &device,
                                *pipeline_layout,
                                *pipeline,
                                &triangles_buffer,
                                command_buffer,
                                image_layout,
                                width,
                                height,
                                image,
                                image_view,
                                frame_index,
                                position,
                            )
                        }
                    },
                );
            }

            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                is_synthetic: _,
            } => match code {
                KeyCode::KeyW => w_pressed = state.is_pressed(),
                KeyCode::KeyS => s_pressed = state.is_pressed(),
                KeyCode::KeyA => a_pressed = state.is_pressed(),
                KeyCode::KeyD => d_pressed = state.is_pressed(),
                _ => {}
            },

            _ => {}
        },

        Event::AboutToWait => {
            device.destroy_resources();

            let speed = 1.0;
            if w_pressed {
                position.offset_y += speed * dt;
            }
            if s_pressed {
                position.offset_y -= speed * dt;
            }
            if a_pressed {
                position.offset_x -= speed * dt;
            }
            if d_pressed {
                position.offset_x += speed * dt;
            }

            match swapchain.try_next_frame(
                |command_buffer: vk::CommandBuffer,
                 image_layout: &mut vk::ImageLayout,
                 width: u32,
                 height: u32,
                 image: vk::Image,
                 image_view: vk::ImageView,
                 frame_index: usize| {
                    unsafe {
                        render(
                            &device,
                            *pipeline_layout,
                            *pipeline,
                            &triangles_buffer,
                            command_buffer,
                            image_layout,
                            width,
                            height,
                            image,
                            image_view,
                            frame_index,
                            position,
                        )
                    }
                },
            ) {
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

#[expect(clippy::too_many_arguments)]
unsafe fn render<'a>(
    device: &Device<'_>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    triangles_buffer: &Buffer,
    command_buffer: vk::CommandBuffer,
    image_layout: &mut vk::ImageLayout,
    width: u32,
    height: u32,
    image: vk::Image,
    image_view: vk::ImageView,
    #[expect(unused)] frame_index: usize,
    position: Position,
) -> RenderSync<'a> {
    unsafe {
        transition_image(
            device,
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
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            0,
            bytemuck::bytes_of(&PushConstants {
                triangles: triangles_buffer.device_address(),
                start_position: position,
                aspect: width as f32 / height as f32,
            }),
        );
        device.cmd_draw(command_buffer, 4, 1, 0, 0);
    }

    unsafe { device.cmd_end_rendering(command_buffer) };

    RenderSync {
        wait_sempahore_info: None,
        signal_sempahore_info: None,
    }
}
