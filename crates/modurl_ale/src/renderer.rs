use minifb::{Window, WindowOptions};

pub(crate) struct Renderer {
    window: Option<Window>,
    buffer: Vec<u32>,
    width: usize,
    height: usize,
}

impl Renderer {
    pub(crate) fn new(width: usize, height: usize, title: &str, render_to_window: bool) -> Self {
        let window = if render_to_window {
            Some(
                Window::new(title, width, height, WindowOptions::default()).expect("create window"),
            )
        } else {
            None
        };

        Self {
            window,
            buffer: vec![0; width * height],
            width,
            height,
        }
    }

    pub(crate) fn set_buffer(&mut self, buffer: &[u32]) {
        self.buffer.copy_from_slice(buffer);
    }

    pub(crate) fn present(&mut self) {
        if let Some(window) = &mut self.window {
            window
                .update_with_buffer(&self.buffer, self.width, self.height)
                .unwrap();
        }
    }
}
