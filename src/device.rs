use crate::Instance;
use ash::vk::{self, Handle};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use scope_guard::scope_guard;
use std::{
    collections::VecDeque,
    ffi::CStr,
    mem::ManuallyDrop,
    ops::Deref,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

pub enum ResourceToDestroy {
    ImageView(vk::ImageView),
    Semaphore(vk::Semaphore),
    Fence(vk::Fence),
    Buffer(vk::Buffer, Allocation),
}

pub struct Device<'allocator> {
    instance: Arc<Instance<'allocator>>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    graphics_queue_family_index: u32,
    graphics_queue: Mutex<vk::Queue>,
    timeline_counter: AtomicU64,
    timeline_semaphore: vk::Semaphore,
    resources_to_destroy: Mutex<VecDeque<(u64, ResourceToDestroy)>>,
    allocator: ManuallyDrop<Mutex<Allocator>>,
}

impl<'allocator> Device<'allocator> {
    pub fn new(instance: Arc<Instance<'allocator>>) -> Self {
        let required_version = vk::API_VERSION_1_3;
        let required_extensions: [&CStr; _] =
            [vk::KHR_SWAPCHAIN_NAME, vk::EXT_SWAPCHAIN_MAINTENANCE1_NAME];

        let device_features = vk::PhysicalDeviceFeatures::default();
        let mut device_features11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut device_features12 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .timeline_semaphore(true)
            .buffer_device_address(true)
            .scalar_block_layout(true);
        let mut device_features13 = vk::PhysicalDeviceVulkan13Features::default()
            .synchronization2(true)
            .dynamic_rendering(true);

        let mut swapchain_maintenance1_features =
            vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                .swapchain_maintenance1(true);

        let mut device_features2 = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut swapchain_maintenance1_features)
            .push_next(&mut device_features13)
            .push_next(&mut device_features12)
            .push_next(&mut device_features11)
            .features(device_features);

        let (physical_device, graphics_queue_family_index) = {
            let mut chosen_physical_device = vk::PhysicalDevice::null();
            let mut chosen_graphics_queue_family_index = vk::QUEUE_FAMILY_IGNORED;

            let physical_devices = unsafe { instance.enumerate_physical_devices() }.unwrap();
            'search: for physical_device in physical_devices {
                let properties =
                    unsafe { instance.get_physical_device_properties(physical_device) };

                let name = properties.device_name_as_c_str().unwrap().to_string_lossy();
                println!("Checking physical device '{name}'");

                if properties.api_version < required_version {
                    println!(
                        "Expected at least physical device version {}.{}.{}.{} but got version {}.{}.{}.{}, skipping this physical device",
                        vk::api_version_variant(required_version),
                        vk::api_version_major(required_version),
                        vk::api_version_minor(required_version),
                        vk::api_version_patch(required_version),
                        vk::api_version_variant(properties.api_version),
                        vk::api_version_major(properties.api_version),
                        vk::api_version_minor(properties.api_version),
                        vk::api_version_patch(properties.api_version),
                    );
                    continue 'search;
                }

                {
                    let extensions =
                        unsafe { instance.enumerate_device_extension_properties(physical_device) }
                            .unwrap();
                    'checks: for required_extension in required_extensions {
                        for extension in &extensions {
                            let Ok(extension) = extension.extension_name_as_c_str() else {
                                continue;
                            };
                            if required_extension == extension {
                                continue 'checks;
                            }
                        }

                        let required_extension_name = required_extension.to_string_lossy();
                        println!(
                            "Unable to find vulkan device extension '{required_extension_name}', skipping this physical device"
                        );
                        continue 'search;
                    }
                }

                let mut graphics_queue_family_index = vk::QUEUE_FAMILY_IGNORED;
                {
                    let queue_families = unsafe {
                        instance.get_physical_device_queue_family_properties(physical_device)
                    };
                    for (i, queue_family) in queue_families.into_iter().enumerate() {
                        if queue_family
                            .queue_flags
                            .contains(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE)
                        {
                            graphics_queue_family_index = i as _;
                            break;
                        }
                    }
                }
                if graphics_queue_family_index == vk::QUEUE_FAMILY_IGNORED {
                    println!(
                        "Unable to find suitable graphics queue family, skipping this physical device"
                    );
                    continue 'search;
                }

                chosen_physical_device = physical_device;
                chosen_graphics_queue_family_index = graphics_queue_family_index;
                println!("Chose physical device '{name}'");
                break 'search;
            }

            if chosen_physical_device.is_null() {
                panic!("Unable to find a suitable vulkan physical device");
            }
            (chosen_physical_device, chosen_graphics_queue_family_index)
        };

        let graphics_queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_queue_family_index)
            .queue_priorities(&[1.0]);
        let queue_create_infos = [graphics_queue_create_info];

        let required_extension_ptrs = required_extensions.map(|extension| extension.as_ptr());
        let device_create_info = vk::DeviceCreateInfo::default()
            .push_next(&mut device_features2)
            .queue_create_infos(&queue_create_infos)
            .enabled_extension_names(&required_extension_ptrs);

        let device = unsafe {
            instance.create_device(physical_device, &device_create_info, instance.allocator())
        }
        .unwrap();
        let cleanup = scope_guard!(|| unsafe { device.destroy_device(instance.allocator()) });

        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family_index, 0) };

        let timeline_counter = 0;

        let mut timline_semaphore_create_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(timeline_counter);
        let timeline_semaphore_create_info =
            vk::SemaphoreCreateInfo::default().push_next(&mut timline_semaphore_create_info);

        let timeline_semaphore = unsafe {
            device.create_semaphore(&timeline_semaphore_create_info, instance.allocator())
        }
        .unwrap();
        let cleanup = cleanup.stack(|()| unsafe {
            device.destroy_semaphore(timeline_semaphore, instance.allocator())
        });

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: (**instance).clone(),
            device: device.clone(),
            physical_device,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })
        .unwrap();

        cleanup.forget();
        Self {
            instance,
            physical_device,
            device,
            graphics_queue_family_index,
            graphics_queue: Mutex::new(graphics_queue),
            timeline_counter: AtomicU64::new(timeline_counter),
            timeline_semaphore,
            resources_to_destroy: Mutex::new(VecDeque::new()),
            allocator: ManuallyDrop::new(Mutex::new(allocator)),
        }
    }

    pub fn instance(&self) -> &Arc<Instance<'allocator>> {
        &self.instance
    }

    pub fn allocator(&self) -> Option<&vk::AllocationCallbacks<'allocator>> {
        self.instance.allocator()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }

    pub fn with_graphics_queue<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let graphics_queue = self.graphics_queue.lock();
        f(*graphics_queue)
    }

    pub fn current_timeline_counter(&self) -> u64 {
        self.timeline_counter.load(Ordering::Relaxed)
    }

    pub fn get_and_then_increment_timeline_counter(&self) -> u64 {
        self.timeline_counter.fetch_add(1, Ordering::Relaxed)
    }

    pub fn signal_timeline_submit_info(&self) -> vk::SemaphoreSubmitInfo<'_> {
        vk::SemaphoreSubmitInfo::default()
            .semaphore(self.timeline_semaphore)
            .value(self.get_and_then_increment_timeline_counter() + 1)
    }

    pub fn wait_for_counter(&self, counter: u64, timeout: u64) -> bool {
        debug_assert!(counter <= self.current_timeline_counter());

        let wait_info = vk::SemaphoreWaitInfo::default()
            .semaphores(core::slice::from_ref(&self.timeline_semaphore))
            .values(core::slice::from_ref(&counter));

        match unsafe { self.wait_semaphores(&wait_info, timeout) } {
            Ok(()) => true,
            Err(vk::Result::TIMEOUT) => false,
            e => {
                e.unwrap();
                false
            }
        }
    }

    /// # Safety
    /// `resource` must be valid to destroy after the timeline semaphore reaches `counter`
    pub unsafe fn schedule_destroy_resource(&self, counter: u64, resource: ResourceToDestroy) {
        debug_assert!(counter <= self.current_timeline_counter());

        let mut resources = self.resources_to_destroy.lock();
        let (Ok(index) | Err(index)) =
            resources.binary_search_by_key(&counter, |&(counter, _)| counter);
        resources.insert(index, (counter, resource));
    }

    pub fn destroy_resources(&self) {
        let mut resources = self.resources_to_destroy.lock();

        let current_counter =
            unsafe { self.get_semaphore_counter_value(self.timeline_semaphore) }.unwrap();

        let allocator = self.allocator();
        while let Some((_, resource)) =
            resources.pop_front_if(|&mut (required_counter, _)| required_counter <= current_counter)
        {
            match resource {
                ResourceToDestroy::ImageView(image_view) => {
                    unsafe { self.destroy_image_view(image_view, allocator) };
                }
                ResourceToDestroy::Semaphore(semaphore) => {
                    unsafe { self.destroy_semaphore(semaphore, allocator) };
                }
                ResourceToDestroy::Fence(fence) => {
                    unsafe { self.destroy_fence(fence, allocator) };
                }
                ResourceToDestroy::Buffer(buffer, allocation) => {
                    unsafe { self.destroy_buffer(buffer, self.allocator()) };
                    self.with_allocator(|allocator| allocator.free(allocation))
                        .unwrap();
                }
            }
        }
    }

    pub fn with_allocator<R>(&self, f: impl FnOnce(&mut Allocator) -> R) -> R {
        let mut allocator = self.allocator.lock();
        f(&mut allocator)
    }
}

impl Deref for Device<'_> {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Drop for Device<'_> {
    fn drop(&mut self) {
        unsafe { self.device_wait_idle() }.unwrap();

        self.destroy_resources();
        debug_assert!(self.resources_to_destroy.get_mut().is_empty());

        unsafe { self.destroy_semaphore(self.timeline_semaphore, self.allocator()) };

        unsafe { ManuallyDrop::drop(&mut self.allocator) };
        unsafe { self.destroy_device(self.allocator()) };
    }
}
