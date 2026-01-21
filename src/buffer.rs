use crate::{Device, Instance, ResourceToDestroy};
use ash::vk;
use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme},
};
use scope_guard::scope_guard;
use std::{mem::ManuallyDrop, ptr::NonNull, sync::Arc};

pub struct Buffer<'allocator> {
    device: Arc<Device<'allocator>>,
    buffer: vk::Buffer,
    allocation: ManuallyDrop<Allocation>,
}

impl<'allocator> Buffer<'allocator> {
    pub fn new(
        device: Arc<Device<'allocator>>,
        name: &str,
        location: MemoryLocation,
        size: u64,
        usage: vk::BufferUsageFlags,
        dedicated_allocation: bool,
    ) -> Self {
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = scope_guard!(
            |buffer| unsafe { device.destroy_buffer(buffer, device.allocator()) },
            unsafe { device.create_buffer(&buffer_create_info, device.allocator()) }.unwrap()
        );
        let requirements = unsafe { device.get_buffer_memory_requirements(*buffer) };

        let allocation = scope_guard!(
            |allocation| device
                .with_allocator(|allocator| allocator.free(allocation))
                .unwrap(),
            device
                .with_allocator(|allocator| {
                    allocator.allocate(&AllocationCreateDesc {
                        name,
                        requirements,
                        location,
                        linear: true,
                        allocation_scheme: if dedicated_allocation {
                            AllocationScheme::DedicatedBuffer(*buffer)
                        } else {
                            AllocationScheme::GpuAllocatorManaged
                        },
                    })
                })
                .unwrap()
        );

        unsafe { device.bind_buffer_memory(*buffer, allocation.memory(), allocation.offset()) }
            .unwrap();

        Self {
            buffer: buffer.into_inner(),
            allocation: ManuallyDrop::new(allocation.into_inner()),
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

    pub fn handle(&self) -> vk::Buffer {
        self.buffer
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        unsafe { self.allocation.memory() }
    }

    pub fn offset(&self) -> u64 {
        self.allocation.offset()
    }

    pub fn size(&self) -> u64 {
        self.allocation.size()
    }

    /// # Safety
    /// This buffer must have been created with [vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS]
    pub unsafe fn device_address(&self) -> vk::DeviceAddress {
        let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.device.get_buffer_device_address(&device_address_info) }
    }

    pub fn as_ptr(&self) -> Option<NonNull<()>> {
        self.allocation.mapped_ptr().map(|ptr| ptr.cast())
    }

    /// # Safety
    /// The GPU must not be writing to this buffer, to avoid data races
    pub unsafe fn get_mapped(&self) -> Option<&[u8]> {
        self.allocation.mapped_slice()
    }

    /// # Safety
    /// The buffer must not be in use by the GPU, to avoid data races
    pub unsafe fn get_mapped_mut(&mut self) -> Option<&mut [u8]> {
        self.allocation.mapped_slice_mut()
    }
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.schedule_destroy_resource(
                self.device.current_timeline_counter(),
                ResourceToDestroy::Buffer(self.buffer, ManuallyDrop::take(&mut self.allocation)),
            );
        }
    }
}
