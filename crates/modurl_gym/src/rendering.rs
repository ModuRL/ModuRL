use minifb::{Window, WindowOptions};

pub struct Renderer {
    window: Window,
    buffer: Vec<u32>, // ARGB8888
    w: usize,
    h: usize,
}

impl Renderer {
    pub fn new(w: usize, h: usize, title: &str) -> Self {
        let window = Window::new(title, w, h, WindowOptions::default()).expect("create window");

        Self {
            window,
            buffer: vec![0; w * h],
            w,
            h,
        }
    }

    pub fn clear(&mut self, color: u32) {
        self.buffer.fill(color);
    }

    pub fn rect(&mut self, x: usize, y: usize, rw: usize, rh: usize, color: u32) {
        let x_end = (x + rw).min(self.w);
        let y_end = (y + rh).min(self.h);

        for row in y..y_end {
            let start = row * self.w + x;
            let end = row * self.w + x_end;
            self.buffer[start..end].fill(color);
        }
    }

    pub fn present(&mut self) {
        self.window
            .update_with_buffer(&self.buffer, self.w, self.h)
            .unwrap();
    }

    pub fn is_open(&self) -> bool {
        self.window.is_open()
    }

    pub fn quad(
        &mut self,
        p1: (f32, f32),
        p2: (f32, f32),
        p3: (f32, f32),
        p4: (f32, f32),
        color: u32,
    ) {
        let min_x = p1.0.min(p2.0.min(p3.0.min(p4.0))).max(0.0) as usize;
        let max_x = p1.0.max(p2.0.max(p3.0.max(p4.0))).min(self.w as f32 - 1.0) as usize;
        let min_y = p1.1.min(p2.1.min(p3.1.min(p4.1))).max(0.0) as usize;
        let max_y = p1.1.max(p2.1.max(p3.1.max(p4.1))).min(self.h as f32 - 1.0) as usize;

        if min_x >= max_x || min_y >= max_y {
            return;
        }

        // Only iterate over the bounding box, not the entire screen
        for row in min_y..=max_y {
            for col in min_x..=max_x {
                let px = col as f32 + 0.5;
                let py = row as f32 + 0.5;

                let b1 = (px - p2.0) * (p1.1 - p2.1) - (p1.0 - p2.0) * (py - p2.1) < 0.0;
                let b2 = (px - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (py - p3.1) < 0.0;
                let b3 = (px - p4.0) * (p3.1 - p4.1) - (p3.0 - p4.0) * (py - p4.1) < 0.0;
                let b4 = (px - p1.0) * (p4.1 - p1.1) - (p4.0 - p1.0) * (py - p1.1) < 0.0;

                if (b1 == b2) && (b2 == b3) && (b3 == b4) {
                    self.buffer[row * self.w + col] = color;
                }
            }
        }
    }

    pub fn draw_circle(&mut self, center_x: usize, center_y: usize, radius: usize, color: u32) {
        let radius_sq = (radius * radius) as isize;
        let x_start = center_x.saturating_sub(radius).max(0);
        let x_end = (center_x + radius).min(self.w - 1);
        let y_start = center_y.saturating_sub(radius).max(0);
        let y_end = (center_y + radius).min(self.h - 1);

        for y in y_start..=y_end {
            for x in x_start..=x_end {
                let dx = x as isize - center_x as isize;
                let dy = y as isize - center_y as isize;
                if dx * dx + dy * dy <= radius_sq {
                    self.buffer[y * self.w + x] = color;
                }
            }
        }
    }

    pub fn get_width(&self) -> usize {
        self.w
    }
    pub fn get_height(&self) -> usize {
        self.h
    }

    /// Draw a flag with pole at the specified position
    ///
    /// # Arguments
    /// * `flag_x` - X position of the flag pole
    /// * `flag_y_bottom` - Y position of the bottom of the flag pole
    /// * `pole_height` - Height of the flag pole
    /// * `flag_width` - Width of the triangular flag
    /// * `flag_height` - Height of the triangular flag  
    /// * `pole_color` - Color of the flag pole (e.g., 0x000000 for black)
    /// * `flag_color` - Color of the flag (e.g., 0xCCCC00 for yellow)
    pub fn draw_flag(
        &mut self,
        flag_x: usize,
        flag_y_bottom: usize,
        pole_height: usize,
        flag_width: usize,
        flag_height: usize,
        pole_color: u32,
        flag_color: u32,
    ) {
        let flag_y_top = flag_y_bottom.saturating_sub(pole_height);

        // Draw flag pole (vertical line)
        for y in flag_y_top..=flag_y_bottom {
            if flag_x < self.w && y < self.h {
                self.rect(flag_x, y, 2, 1, pole_color);
            }
        }

        // Draw flag (triangle) with point vertically centered
        // Fill the triangular flag area
        for y_offset in 0..flag_height {
            let distance_from_center = ((y_offset as i32) - (flag_height as i32 / 2)).abs();
            let width_at_y = flag_width as i32 * (flag_height as i32 / 2 - distance_from_center)
                / (flag_height as i32 / 2);

            if width_at_y > 0 {
                for x_offset in 0..width_at_y {
                    let px = flag_x + 2 + x_offset as usize; // Start after the pole
                    let py = flag_y_top + y_offset;
                    if px < self.w && py < self.h {
                        self.rect(px, py, 1, 1, flag_color);
                    }
                }
            }
        }
    }
}
