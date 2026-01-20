use ash::vk;
use gpu_allocator::MemoryLocation;
use std::sync::Arc;
use triangle_based_rendering::{
    Buffer, Device, Instance, RenderResult, RenderSync, Surface, Swapchain, make_subresource_range,
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
        let attributes = WindowAttributes::default().with_title("Renderer");
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
        vk::BufferUsageFlags::STORAGE_BUFFER,
        false,
    );

    unsafe { buffer.get_mapped_mut() }.unwrap().fill(0);

    let render = |command_buffer: vk::CommandBuffer,
                  image_layout: &mut vk::ImageLayout,
                  image: vk::Image,
                  #[expect(unused)] image_view: vk::ImageView,
                  #[expect(unused)] frame_index: usize| {
        unsafe {
            transition_image(
                &device,
                command_buffer,
                image,
                image_layout,
                vk::ImageLayout::GENERAL,
            );
            device.cmd_clear_color_image(
                command_buffer,
                image,
                *image_layout,
                &vk::ClearColorValue {
                    float32: [1.0, 0.0, 0.0, 1.0],
                },
                &[make_subresource_range(vk::ImageAspectFlags::COLOR)],
            );
        }

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
