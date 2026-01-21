use ash::vk;
use scope_guard::scope_guard;
use std::{
    ffi::{CStr, c_void},
    ops::Deref,
};

pub struct Instance<'allocator> {
    entry: ash::Entry,
    allocator: Option<vk::AllocationCallbacks<'allocator>>,
    instance: ash::Instance,
}

impl<'allocator> Instance<'allocator> {
    /// # Safety
    /// `entry` must be valid
    /// `allocator` must be valid
    pub unsafe fn new(
        entry: ash::Entry,
        allocator: Option<vk::AllocationCallbacks<'allocator>>,
    ) -> Self {
        let required_version = vk::API_VERSION_1_3;
        let required_layers: [&CStr; _] = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation",
        ];
        let required_extensions: [&CStr; _] = [
            #[cfg(windows)]
            vk::KHR_WIN32_SURFACE_NAME,
            vk::KHR_SURFACE_NAME,
            vk::KHR_GET_SURFACE_CAPABILITIES2_NAME,
            vk::EXT_SURFACE_MAINTENANCE1_NAME,
            #[cfg(debug_assertions)]
            vk::EXT_DEBUG_UTILS_NAME,
        ];

        {
            let version = unsafe { entry.try_enumerate_instance_version() }
                .unwrap()
                .unwrap_or(vk::API_VERSION_1_0);
            if version < required_version {
                panic!(
                    "Expected at least vulkan api version {}.{}.{}.{} but got version {}.{}.{}.{}",
                    vk::api_version_variant(required_version),
                    vk::api_version_major(required_version),
                    vk::api_version_minor(required_version),
                    vk::api_version_patch(required_version),
                    vk::api_version_variant(version),
                    vk::api_version_major(version),
                    vk::api_version_minor(version),
                    vk::api_version_patch(version),
                );
            }
        }

        {
            let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap();
            'checks: for required_layer in required_layers {
                for layer in &layers {
                    let Ok(layer) = layer.layer_name_as_c_str() else {
                        continue;
                    };
                    if required_layer == layer {
                        continue 'checks;
                    }
                }

                let required_layer_name = required_layer.to_string_lossy();
                panic!("Unable to find vulkan layer '{required_layer_name}'");
            }
        }

        {
            let extensions =
                unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap();
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
                panic!("Unable to find vulkan extension '{required_extension_name}'");
            }
        }

        let application_info = vk::ApplicationInfo::default()
            .application_name(c"Renderer")
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(c"Renderer")
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(required_version);

        let required_layer_ptrs = required_layers.map(|layer| layer.as_ptr());
        let required_extension_ptrs = required_extensions.map(|extension| extension.as_ptr());
        let mut instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .enabled_layer_names(&required_layer_ptrs)
            .enabled_extension_names(&required_extension_ptrs);

        unsafe extern "system" fn debug_message_callback(
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
            message_types: vk::DebugUtilsMessageTypeFlagsEXT,
            p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
            #[expect(unused)] p_user_data: *mut c_void,
        ) -> vk::Bool32 {
            let message = unsafe {
                (*p_callback_data)
                    .message_as_c_str()
                    .unwrap_or(c"")
                    .to_string_lossy()
            };
            eprintln!("{message_severity:?} {message_types:?} {message}");
            vk::FALSE
        }

        let mut debug_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(debug_message_callback));
        if cfg!(debug_assertions) {
            instance_create_info = instance_create_info.push_next(&mut debug_messenger_create_info);
        }

        let instance =
            unsafe { entry.create_instance(&instance_create_info, allocator.as_ref()) }.unwrap();
        let cleanup = scope_guard!(|| unsafe { instance.destroy_instance(allocator.as_ref()) });

        cleanup.forget();
        Self {
            entry,
            allocator,
            instance,
        }
    }

    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    pub fn allocator(&self) -> Option<&vk::AllocationCallbacks<'allocator>> {
        self.allocator.as_ref()
    }
}

impl Deref for Instance<'_> {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Drop for Instance<'_> {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(self.allocator()) };
    }
}
