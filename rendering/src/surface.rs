use crate::Instance;
use ash::vk;
use scope_guard::scope_guard;
use std::{ops::Deref, sync::Arc};
use winit::raw_window_handle::{
    HasDisplayHandle, HasWindowHandle, RawWindowHandle, Win32WindowHandle,
};

pub struct Surface<'allocator, 'window> {
    instance: Arc<Instance<'allocator>>,
    #[expect(unused)]
    window: Box<dyn 'window + Send + Sync>,
    surface: vk::SurfaceKHR,
    surface_funcs: ash::khr::surface::Instance,
}

impl<'allocator, 'window> Surface<'allocator, 'window> {
    pub fn new(
        instance: Arc<Instance<'allocator>>,
        window: impl 'window + HasWindowHandle + HasDisplayHandle + Send + Sync,
    ) -> Self {
        let surface_funcs = ash::khr::surface::Instance::new(instance.entry(), &instance);

        let surface = match window.window_handle().unwrap().as_raw() {
            RawWindowHandle::Win32(Win32WindowHandle {
                hwnd, hinstance, ..
            }) => {
                let win32_funcs =
                    ash::khr::win32_surface::Instance::new(instance.entry(), &instance);

                let surface_create_info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(hinstance.map_or(0, |hinstance| hinstance.get()))
                    .hwnd(hwnd.get());

                unsafe {
                    win32_funcs.create_win32_surface(&surface_create_info, instance.allocator())
                }
                .unwrap()
            }

            _ => panic!("Unsupported platform"),
        };
        let cleanup = scope_guard!(|| unsafe {
            surface_funcs.destroy_surface(surface, instance.allocator())
        });

        cleanup.forget();
        Self {
            instance,
            window: Box::new(window),
            surface,
            surface_funcs,
        }
    }

    pub fn instance(&self) -> &Arc<Instance<'allocator>> {
        &self.instance
    }

    pub fn allocator(&self) -> Option<&vk::AllocationCallbacks<'allocator>> {
        self.instance.allocator()
    }

    pub fn handle(&self) -> vk::SurfaceKHR {
        self.surface
    }
}

impl Deref for Surface<'_, '_> {
    type Target = ash::khr::surface::Instance;

    fn deref(&self) -> &Self::Target {
        &self.surface_funcs
    }
}

impl Drop for Surface<'_, '_> {
    fn drop(&mut self) {
        unsafe { self.destroy_surface(self.surface, self.allocator()) };
    }
}
