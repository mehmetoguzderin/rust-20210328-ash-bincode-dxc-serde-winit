use ash::{
    version::{DeviceV1_0, EntryV1_0, InstanceV1_0},
    vk,
};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::error::Error;
use std::ffi::{CStr, CString};
use std::io::Cursor;
use std::mem::size_of;
use std::result::Result;
use std::time::Instant;
use winit::{
    dpi::PhysicalSize,
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
struct CSInput {
    width: u32,
    height: u32,
    time: f32,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

fn main() -> Result<(), Box<dyn Error>> {
    let application_name = {
        let application_name = "rust 20210328 ash bincode dxc serde winit";
        (
            application_name,
            CString::new("rust 20210328 ash bincode dxc serde winit")?,
        )
    };
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(application_name.0)
        .with_inner_size(PhysicalSize::new(720, 720))
        .with_resizable(false)
        .build(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    let layer_names = [CString::new("VK_LAYER_KHRONOS_validation")?];
    let layers_names_raw: Vec<*const i8> = layer_names
        .iter()
        .map(|raw_name| raw_name.as_ptr())
        .collect();

    let surface_extensions = ash_window::enumerate_required_extensions(&window)?;
    let mut extension_names_raw = surface_extensions
        .iter()
        .map(|ext| ext.as_ptr())
        .collect::<Vec<_>>();
    extension_names_raw.push(ash::extensions::ext::DebugUtils::name().as_ptr());

    let application_info = vk::ApplicationInfo::builder()
        .application_name(&application_name.1)
        .application_version(0)
        .engine_name(&application_name.1)
        .engine_version(0)
        .api_version(vk::make_version(1, 1, 0))
        .build();

    let instance_info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers_names_raw)
        .enabled_extension_names(&extension_names_raw)
        .build();

    let instance = unsafe { entry.create_instance(&instance_info, None) }?;

    let debug_utils_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_debug_callback))
        .build();

    let debug_utils_loader = ash::extensions::ext::DebugUtils::new(&entry, &instance);
    let debug_utils_messenger = unsafe {
        debug_utils_loader.create_debug_utils_messenger(&debug_utils_messenger_info, None)
    }?;
    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;
    let pdevices = unsafe { instance.enumerate_physical_devices() }?;
    let surface_loader = ash::extensions::khr::Surface::new(&entry, &instance);
    let (pdevice, queue_family_index) = pdevices
        .iter()
        .map(|pdevice| {
            unsafe { instance.get_physical_device_queue_family_properties(*pdevice) }
                .iter()
                .enumerate()
                .filter_map(|(index, ref info)| {
                    let supports_graphic_and_surface =
                        info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && unsafe {
                                surface_loader.get_physical_device_surface_support(
                                    *pdevice,
                                    index as u32,
                                    surface,
                                )
                            }
                            .unwrap();
                    if supports_graphic_and_surface {
                        Some((*pdevice, index))
                    } else {
                        None
                    }
                })
                .next()
        })
        .flatten()
        .next()
        .ok_or("")?;
    #[cfg(not(target_os = "macos"))]
    let physical_device_properties = unsafe { instance.get_physical_device_properties(pdevice) };
    let queue_family_index = queue_family_index as u32;
    let device_extension_names_raw = [ash::extensions::khr::Swapchain::name().as_ptr()];
    let features = vk::PhysicalDeviceFeatures {
        shader_clip_distance: 1,
        ..Default::default()
    };
    let priorities = [1.0];

    let device_queue_info = [vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&priorities)
        .build()];

    let device_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&device_queue_info)
        .enabled_extension_names(&device_extension_names_raw)
        .enabled_features(&features)
        .build();

    let device = unsafe { instance.create_device(pdevice, &device_info, None) }?;

    let present_queue = unsafe { device.get_device_queue(queue_family_index as u32, 0) };

    let surface_formats =
        unsafe { surface_loader.get_physical_device_surface_formats(pdevice, surface) }?;

    let surface_format = surface_formats
        .iter()
        .filter(|surface_format| {
            let format_properties = unsafe {
                instance.get_physical_device_format_properties(pdevice, surface_format.format)
            };
            format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
                && surface_format.format == vk::Format::B8G8R8A8_UNORM
                || surface_format.format == vk::Format::R8G8B8A8_UNORM
        })
        .collect::<Vec<_>>()[0];

    let surface_capabilities =
        unsafe { surface_loader.get_physical_device_surface_capabilities(pdevice, surface) }?;
    let mut desired_image_count = surface_capabilities.min_image_count + 1;
    if surface_capabilities.max_image_count > 0
        && desired_image_count > surface_capabilities.max_image_count
    {
        desired_image_count = surface_capabilities.max_image_count;
    }
    let surface_resolution = match surface_capabilities.current_extent.width {
        std::u32::MAX => vk::Extent2D {
            width: window.inner_size().width,
            height: window.inner_size().height,
        },
        _ => vk::Extent2D {
            width: surface_capabilities.current_extent.width,
            height: surface_capabilities.current_extent.height,
        },
    };
    let pre_transform = if surface_capabilities
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        surface_capabilities.current_transform
    };
    let present_modes =
        unsafe { surface_loader.get_physical_device_surface_present_modes(pdevice, surface) }?;
    let present_mode = present_modes
        .iter()
        .cloned()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);
    let swapchain_loader = ash::extensions::khr::Swapchain::new(&instance, &device);

    let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(desired_image_count)
        .image_color_space(surface_format.color_space)
        .image_format(surface_format.format)
        .image_extent(surface_resolution)
        .image_usage(vk::ImageUsageFlags::STORAGE)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(pre_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_array_layers(1)
        .build();

    let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_info, None) }?;

    let command_pool_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index)
        .build();

    let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }?;

    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .build();

    let command_buffers =
        unsafe { device.allocate_command_buffers(&command_buffer_allocate_info) }?;
    let command_buffer = command_buffers[0];

    let present_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;
    let present_image_views: Vec<vk::ImageView> = present_images
        .iter()
        .map(|&image| {
            let image_view_info = vk::ImageViewCreateInfo::builder()
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(surface_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image(image)
                .build();
            unsafe { device.create_image_view(&image_view_info, None) }.unwrap()
        })
        .collect();

    let fence_info = vk::FenceCreateInfo::builder()
        .flags(vk::FenceCreateFlags::SIGNALED)
        .build();

    let draw_commands_reuse_fence = unsafe { device.create_fence(&fence_info, None) }?;

    let semaphore_info = vk::SemaphoreCreateInfo::default();

    let present_complete_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }?;
    let rendering_complete_semaphore = unsafe { device.create_semaphore(&semaphore_info, None) }?;

    let mut compute_spv_file = Cursor::new(&include_bytes!("../gpu/main.hlsl.spv")[..]);

    let spv = ash::util::read_spv(&mut compute_spv_file)?;
    let shader_module_info = vk::ShaderModuleCreateInfo::builder().code(&spv).build();

    let shader_module = unsafe { device.create_shader_module(&shader_module_info, None) }?;

    let shader_entry_name = CString::new("main")?;

    let descriptor_set_layout_bindings = [vk::DescriptorSetLayoutBinding::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .build()];

    let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(&descriptor_set_layout_bindings)
        .build();

    let descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }?;

    let descriptor_pool_sizes = [vk::DescriptorPoolSize::builder()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(present_image_views.len() as u32)
        .build()];
    let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(present_image_views.len() as u32)
        .pool_sizes(&descriptor_pool_sizes)
        .build();

    let descriptor_pool = unsafe { device.create_descriptor_pool(&descriptor_pool_info, None) }?;

    let descriptor_set_layouts = present_image_views
        .iter()
        .map(|_| descriptor_set_layout)
        .collect::<Vec<_>>();

    let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&descriptor_set_layouts)
        .build();

    let descriptor_sets =
        unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }?;

    let descriptor_image_infos = present_image_views
        .iter()
        .map(|&present_image_view| {
            [vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::GENERAL)
                .image_view(present_image_view)
                .build()]
        })
        .collect::<Vec<_>>();

    let write_descriptor_sets = descriptor_image_infos
        .iter()
        .enumerate()
        .map(|(i, descriptor_image_info)| {
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets[i])
                .image_info(descriptor_image_info)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .build()
        })
        .collect::<Vec<_>>();

    unsafe { device.update_descriptor_sets(&write_descriptor_sets, &[]) };

    let push_constant_ranges = [vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(size_of::<CSInput>() as u32)
        .build()];

    let descriptor_set_layouts = [descriptor_set_layout];

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
        .push_constant_ranges(&push_constant_ranges)
        .set_layouts(&descriptor_set_layouts)
        .build();

    let pipeline_layout =
        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }?;

    let pipeline_shader_stage_create_info = vk::PipelineShaderStageCreateInfo {
        s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
        module: shader_module,
        p_name: shader_entry_name.as_ptr(),
        stage: vk::ShaderStageFlags::COMPUTE,
        ..Default::default()
    };

    let compute_pipeline_create_info = vk::ComputePipelineCreateInfo::builder()
        .stage(pipeline_shader_stage_create_info)
        .layout(pipeline_layout)
        .build();

    let compute_pipelines = unsafe {
        device.create_compute_pipelines(
            vk::PipelineCache::null(),
            &[compute_pipeline_create_info],
            None,
        )
    }
    .unwrap();

    let compute_pipeline = compute_pipelines[0];

    let instant = Instant::now();

    let query_pools: Vec<vk::QueryPool> = present_image_views
        .iter()
        .map(|_| {
            let query_pool_create_info = vk::QueryPoolCreateInfo::builder()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2)
                .build();
            unsafe { device.create_query_pool(&query_pool_create_info, None) }.unwrap()
        })
        .collect();

    let mut run = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawRequested(window_id) if window_id == window.id() && run => {
                let (present_index, _) = unsafe {
                    swapchain_loader.acquire_next_image(
                        swapchain,
                        std::u64::MAX,
                        present_complete_semaphore,
                        vk::Fence::null(),
                    )
                }
                .unwrap();

                unsafe {
                    device
                        .wait_for_fences(&[draw_commands_reuse_fence], true, std::u64::MAX)
                        .expect("Wait for fence failed.")
                };

                unsafe {
                    device
                        .reset_fences(&[draw_commands_reuse_fence])
                        .expect("Reset fences failed.")
                };

                unsafe {
                    device
                        .reset_command_buffer(
                            command_buffer,
                            vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                        )
                        .expect("Reset command buffer failed.")
                };

                let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                    .build();

                unsafe {
                    device
                        .begin_command_buffer(command_buffer, &command_buffer_begin_info)
                        .expect("Begin commandbuffer")
                };
                unsafe {
                    device.cmd_reset_query_pool(
                        command_buffer,
                        query_pools[present_index as usize],
                        0,
                        2,
                    )
                };
                let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                    .image(present_images[present_index as usize])
                    .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    )
                    .build();
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    )
                };
                unsafe {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        pipeline_layout,
                        0,
                        &[descriptor_sets[present_index as usize]],
                        &[],
                    )
                };
                unsafe {
                    device.cmd_bind_pipeline(
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        compute_pipeline,
                    )
                };
                let push_constants = bincode::serialize(&CSInput {
                    width: surface_resolution.width as u32,
                    height: surface_resolution.height as u32,
                    time: instant.elapsed().as_secs_f32(),
                })
                .unwrap();
                unsafe {
                    device.cmd_push_constants(
                        command_buffer,
                        pipeline_layout,
                        vk::ShaderStageFlags::COMPUTE,
                        0,
                        &push_constants,
                    )
                };
                unsafe {
                    device.cmd_write_timestamp(
                        command_buffer,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        query_pools[present_index as usize],
                        0,
                    )
                };
                unsafe {
                    device.cmd_dispatch(
                        command_buffer,
                        surface_resolution.width as u32 / 8u32,
                        surface_resolution.height as u32 / 8u32,
                        1,
                    )
                };
                unsafe {
                    device.cmd_write_timestamp(
                        command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        query_pools[present_index as usize],
                        1,
                    )
                };
                let layout_transition_barriers = vk::ImageMemoryBarrier::builder()
                    .image(present_images[present_index as usize])
                    .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .layer_count(1)
                            .level_count(1)
                            .build(),
                    )
                    .build();
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    )
                };

                unsafe {
                    device
                        .end_command_buffer(command_buffer)
                        .expect("End commandbuffer")
                };

                let command_buffers = vec![command_buffer];

                let submit_info = vk::SubmitInfo::builder()
                    .wait_semaphores(&[present_complete_semaphore])
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&[rendering_complete_semaphore])
                    .build();

                unsafe {
                    device
                        .queue_submit(present_queue, &[submit_info], draw_commands_reuse_fence)
                        .expect("queue submit failed.")
                };
                let wait_semaphors = [rendering_complete_semaphore];
                let swapchains = [swapchain];
                let image_indices = [present_index];
                let present_info_khr = vk::PresentInfoKHR::builder()
                    .wait_semaphores(&wait_semaphors)
                    .swapchains(&swapchains)
                    .image_indices(&image_indices)
                    .build();
                unsafe { swapchain_loader.queue_present(present_queue, &present_info_khr) }
                    .unwrap();
                unsafe { device.device_wait_idle() }.unwrap();
                let mut duration = [0u64; 2];
                unsafe {
                    device.get_query_pool_results(
                        query_pools[present_index as usize],
                        0,
                        2,
                        &mut duration,
                        vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                    )
                }
                .unwrap();
                #[cfg(not(target_os = "macos"))]
                println!(
                    "{:#?} microseconds",
                    (duration[1] - duration[0]) as f64
                        * physical_device_properties.limits.timestamp_period as f64
                        / 1000.0
                );
            }
            Event::WindowEvent { event, window_id } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    _ => (),
                };
            }
            _ => (),
        }
        if *control_flow == ControlFlow::Exit {
            run = false;
            unsafe { device.device_wait_idle() }.unwrap();
            for &query_pool in &query_pools {
                unsafe { device.destroy_query_pool(query_pool, None) };
            }
            unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
            unsafe { device.destroy_descriptor_pool(descriptor_pool, None) };
            unsafe { device.destroy_pipeline(compute_pipeline, None) };
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
            unsafe { device.destroy_shader_module(shader_module, None) };
            unsafe { device.destroy_semaphore(present_complete_semaphore, None) };
            unsafe { device.destroy_semaphore(rendering_complete_semaphore, None) };
            unsafe { device.destroy_fence(draw_commands_reuse_fence, None) };
            for &image_view in &present_image_views {
                unsafe { device.destroy_image_view(image_view, None) };
            }
            unsafe { device.destroy_command_pool(command_pool, None) };
            unsafe { swapchain_loader.destroy_swapchain(swapchain, None) };
            unsafe { device.destroy_device(None) };
            unsafe { surface_loader.destroy_surface(surface, None) };
            unsafe {
                debug_utils_loader.destroy_debug_utils_messenger(debug_utils_messenger, None)
            };
            unsafe { instance.destroy_instance(None) };
        } else {
            window.request_redraw();
        }
    });
}
