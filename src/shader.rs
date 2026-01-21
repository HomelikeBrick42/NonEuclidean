use crate::{Device, Instance, ResourceToDestroy};
use ash::vk;
use std::sync::Arc;

pub struct Shader<'allocator> {
    device: Arc<Device<'allocator>>,
    shader: vk::ShaderModule,
}

impl<'allocator> Shader<'allocator> {
    /// # Safety
    /// `spirv_code` must be valid SPIR-V code
    pub unsafe fn new(device: Arc<Device<'allocator>>, spirv_code: &[u32]) -> Self {
        let create_info = vk::ShaderModuleCreateInfo::default().code(spirv_code);
        let shader =
            unsafe { device.create_shader_module(&create_info, device.allocator()) }.unwrap();
        Self { device, shader }
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

    pub fn handle(&self) -> vk::ShaderModule {
        self.shader
    }
}

impl Drop for Shader<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.schedule_destroy_resource(
                self.device.current_timeline_counter(),
                ResourceToDestroy::ShaderModule(self.shader),
            );
        }
    }
}

#[macro_export]
macro_rules! include_spirv {
    ($($path:tt)*) => {
        const {
            #[repr(C)]
            struct Aligned<T: ?Sized> {
                align: [u32; 0],
                bytes: T,
            }

            const BYTES: &Aligned<[u8]> = &Aligned {
                align: [],
                bytes: *include_bytes!($($path)*),
            };

            assert!(BYTES.bytes.len().is_multiple_of(4));
            unsafe {
                core::slice::from_raw_parts(
                    BYTES.bytes.as_ptr().cast::<u32>(),
                    BYTES.bytes.len() / 4,
                )
            }
        }
    };
}
