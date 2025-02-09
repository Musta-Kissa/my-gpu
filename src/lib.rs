#![allow(unused_variables, dead_code)]
#[cfg(test)]
mod test;

#[macro_use]
extern crate my_math;

use core::panic;
use std::any::Any;

use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Mul;
use std::ops::Add;

use my_math::matrix::Matrix;
use my_math::vec::IVec2;
use my_math::vec::Vec3;
use my_math::vec::Vec4;

pub const ALPHA_FULL: u32 = 0b1111_1111 << 24;
pub const ALPHA_HALF: u32 = 0b0111_1111 << 24;
pub const WHITE: u32 = 0b1111_1111 << 24 ^ !0u32;
pub const BLACK: u32 = 0u32;
pub const RED: u32 = ((1u32 << 9) - 1) << 16;
pub const RED_TRANSPARENT: u32 = ALPHA_HALF | ((1u32 << 9) - 1) << 16;
pub const GREEN: u32 = ((1u32 << 9) - 1) << 8;
pub const BLUE: u32 = (1u32 << 9) - 1;
pub const LIGHT_BLUE: u32 = 126 << 16 | 172 << 8 | 242;
pub const MAGENTA: u32 = RED | BLUE;
pub const YELLOW: u32 = GREEN | RED;

// The order is reversed in memory
#[derive(Clone, Copy)]
struct ColorChanels {
    b: u8,
    g: u8,
    r: u8,
    a: u8,
}
#[derive(Clone, Copy)]
union Color {
    col: u32,
    ch: ColorChanels,
}
impl Mul<f64> for Color {
    type Output = Color;
    fn mul(self, rhs: f64) -> Self::Output {
        unsafe {
            let a = self.ch.a;
            let r = (self.ch.r as f64 * rhs).round() as u8;
            let g = (self.ch.g as f64 * rhs).round() as u8;
            let b = (self.ch.b as f64 * rhs).round() as u8;
            Color {
                ch: ColorChanels {
                    a,
                    r,g,b
                }
            }
        }
    } 
}
impl Add<Color> for Color {
    type Output = Color;
    fn add(self, rhs: Color) -> Self::Output {
        unsafe {
            Color {
                ch: ColorChanels {
                    a: self.ch.a,
                    r: self.ch.r + rhs.ch.r,
                    g: self.ch.g + rhs.ch.g,
                    b: self.ch.b + rhs.ch.b,
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct TriangleV3 {
    vert: [Vec3; 3],
    col: [Color; 3],
}
impl TriangleV3 {
    fn new(p1: Vec3, p2: Vec3, p3: Vec3, c1: Color, c2: Color, c3: Color) -> Self {
        TriangleV3 {
            vert: [p1, p2, p3],
            col: [c1, c2, c3],
        }
    }
}
#[derive(Clone, Copy)]
struct TriangleV4 {
    vert: [Vec4; 3],
    col: [Color; 3],
}
impl TriangleV4 {
    fn new(p1: Vec4, p2: Vec4, p3: Vec4, c1: Color, c2: Color, c3: Color) -> Self {
        TriangleV4 {
            vert: [p1, p2, p3],
            col: [c1, c2, c3],
        }
    }
    /// Returns true if the triangle should be rendered
    fn winding_order(&self) -> bool {
        let edge12 = self.vert[0].to_vec3() - self.vert[1].to_vec3();
        let edge13 = self.vert[2].to_vec3() - self.vert[1].to_vec3();
        let winding_order = edge12.cross(edge13);
        if winding_order.z.is_sign_positive() {
            return true;
        } else {
            return false;
        }
    }
}
impl Mul<TriangleV4> for Matrix<4, 4> {
    type Output = TriangleV4;

    fn mul(self, rhs: TriangleV4) -> Self::Output {
        let mut out = rhs;
        out.vert[0] = self * rhs.vert[0];
        out.vert[1] = self * rhs.vert[1];
        out.vert[2] = self * rhs.vert[2];
        out
    }
}

pub struct Binds(Vec<*mut dyn Any>);

impl Binds {
    pub fn cast_ref<'a, T: 'static>(&'a self, idx: usize) -> Option<&'a T> {
        unsafe { (*self.0[idx]).downcast_ref::<T>() }
    }
}

impl Deref for Binds {
    type Target = Vec<*mut dyn Any>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Binds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub trait ClipPos {
    fn clip_pos(&self) -> Vec4;
}
pub trait VertexPos {
    fn vertex_pos(&self) -> Vec3;
}

#[derive(Copy, Clone)]
pub struct SurfaceConfig {
    pub width: usize,
    pub height: usize,
}
#[derive(Copy, Clone)]
pub struct Config {
    pub surface_cofig: SurfaceConfig,
}
pub struct Gpu<VertexIn, VertexOut> {
    config: Config,

    surface: *mut u32,
    binds: Binds,

    zbuffer: Vec<f64>,

    vertex_shader: fn(&VertexIn, &mut Binds) -> VertexOut,
    fragment_shader: fn(&VertexOut, &mut Binds) -> u32,
}

#[allow(dead_code)]
impl<VertexIn: VertexPos, VertexOut: ClipPos> Gpu<VertexIn, VertexOut> {
    // The vertex out must have a field of clip_pos
    pub fn new(
        config: Config,
        surface: *mut u32,
        binds: Binds,
        vertex_shader: fn(&VertexIn, &mut Binds) -> VertexOut,
        fragment_shader: fn(&VertexOut, &mut Binds) -> u32,
    ) -> Self {
        Gpu {
            config,
            surface,
            binds,
            zbuffer: vec![1.0f64; config.surface_cofig.width * config.surface_cofig.height],
            vertex_shader,
            fragment_shader,
        }
    }
    fn pixel_fits(&self, pixel: IVec2) -> bool {
        !(pixel.y < 0
            || pixel.x < 0
            || pixel.y >= self.config.surface_cofig.height as i64
            || pixel.x >= self.config.surface_cofig.width as i64)
    }

    fn set_pixel(&mut self, pixel: IVec2, color: u32) {
        if !self.pixel_fits(pixel) {
            return;
        }
        unsafe {
            //*self.surface.add(pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize) =
            //*self.surface.add(pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize) | color;
            *self
                .surface
                .add(pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize) = color;
        }
    }
    fn check_z(&mut self, pos: IVec2, z: f64) -> bool {
        if !self.pixel_fits(pos) {
            return false;
        }
        self.zbuffer[pos.y as usize * self.config.surface_cofig.width + pos.x as usize] > z
    }
    fn set_pixel_z(&mut self, pixel_pos: IVec2, color: u32, z: f64) {
        if !self.pixel_fits(pixel_pos) {
            return;
        }
        // !( a > b) not the same as: a <= b
        // NANI??
        // doesnt work:
        //if (self.zbuffer[pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize] <= z){
        // this works:
        let p = pixel_pos.y as usize * self.config.surface_cofig.width + pixel_pos.x as usize;
        if !(self.zbuffer[p] > z) {
            return;
        }
        unsafe {
            let pixel = self
                .surface
                .add(pixel_pos.y as usize * self.config.surface_cofig.width + pixel_pos.x as usize);
            let bg_color_r = (*pixel << 8) >> 24;
            let bg_color_g = (*pixel << 16) >> 24;
            let bg_color_b = (*pixel << 24) >> 24;
            let alpha = 1.0 - (color >> 24) as f64 / 0b1111_1111 as f64;
            let color_r = (color << 8) >> 24;
            let color_g = (color << 16) >> 24;
            let color_b = (color << 24) >> 24;

            //let bg_alpha = 1.0 - (*pixel >> 24) as f64 / 0b1111_1111 as f64;
            //let alpha0 = alpha + bg_alpha * (1. - alpha);
            //let out_color_r = ((alpha * color_r as f64 + (1. - alpha) * bg_color_r as f64 * bg_alpha as f64)/alpha0).round() as u32;
            //let out_color_g = ((alpha * color_g as f64 + (1. - alpha) * bg_color_g as f64 * bg_alpha as f64)/alpha0).round() as u32;
            //let out_color_b = ((alpha * color_b as f64 + (1. - alpha) * bg_color_b as f64 * bg_alpha as f64)/alpha0).round() as u32;

            let out_color_r =
                (alpha * color_r as f64 + (1. - alpha) * bg_color_r as f64).round() as u32;
            let out_color_g =
                (alpha * color_g as f64 + (1. - alpha) * bg_color_g as f64).round() as u32;
            let out_color_b =
                (alpha * color_b as f64 + (1. - alpha) * bg_color_b as f64).round() as u32;

            *pixel = out_color_r << 16 | out_color_g << 8 | out_color_b;
        }
        self.zbuffer[p] = z;
    }

    fn line(&mut self, start: &IVec2, end: &IVec2, color: u32) {
        let d_y: i64 = (end.y - start.y).abs();
        let d_x: i64 = (end.x - start.x).abs();

        let s_x: i64 = if start.x < end.x { 1_i64 } else { -1_i64 };
        let s_y: i64 = if start.y < end.y { 1_i64 } else { -1_i64 };

        let mut curr_y = start.y;
        let mut curr_x = start.x;

        let mut err = d_x - d_y;

        while !(curr_y == end.y && curr_x == end.x) {
            self.set_pixel(ivec2!(curr_x, curr_y), color);

            let e2 = err * 2;
            if e2 > -d_y {
                err -= d_y;
                curr_x += s_x;
            }
            if e2 < d_x {
                err += d_x;
                curr_y += s_y;
            }
        }
    }
    fn clip_to_screen_v3(&self, pos: Vec4) -> Vec3 {
        // already devided when clipping
        //let pos = pos / pos.w;
        let s_x = (pos.x + 1.) * self.config.surface_cofig.width as f64 / 2.;
        let s_y = (-pos.y + 1.) * self.config.surface_cofig.height as f64 / 2.;
        vec3!(s_x, s_y, pos.z)
    }
    fn clip_to_screen(&self, pos: Vec4) -> IVec2 {
        let pos = pos / pos.w;
        let s_x: i64 = ((pos.x + 1.) * self.config.surface_cofig.width as f64 / 2.).floor() as i64;
        let s_y: i64 =
            ((-pos.y + 1.) * self.config.surface_cofig.height as f64 / 2.).floor() as i64;
        ivec2!(s_x, s_y)
    }

    pub fn clear(&mut self, color: u32) {
        let surface_len = self.config.surface_cofig.height * self.config.surface_cofig.width;
        for i in 0..surface_len {
            unsafe {
                *self.surface.add(i) = color;
            }
            self.zbuffer[i] = 1.0;
        }
    }
    /// Rasterize the triangle with the scanline algorithm
    fn fill_triangle_z_f(&mut self, p1: Vec3, p2: Vec3, p3: Vec3, c1: Color, c2: Color, c3: Color) {
        let mut points = vec![p1, p2, p3];
        points.sort_by(|a, b| (a.y).total_cmp(&b.y));
        let p1 = points[0];
        let p2 = points[1];
        let p3 = points[2];

        if p2.y == p3.y {
            self.fill_flat_bottom(p1, p2, p3, c1, c2, c3);
        } else if p1.y == p2.y {
            self.fill_flat_top(p1, p2, p3, c1, c2, c3);
        } else {
            let xslope = (p3.x - p1.x) / (p3.y - p1.y);
            let x = xslope * (p2.y - p1.y) + p1.x;
            let zslope = (p3.z - p1.z) / (p3.y - p1.y);
            let z = zslope * (p2.y - p1.y) + p1.z;
            let y = p2.y;
            let d = vec3!(x, y, z);
            self.fill_flat_bottom(p1, p2, d, c1, c2, c3);
            self.fill_flat_top(d, p2, p3, c1, c2, c3);
        }
    }

    fn fill_flat_top(&mut self, p1: Vec3, p2: Vec3, p3: Vec3, c1: Color, c2: Color, c3: Color) {
        let mut xinc_13 = (p3.x - p1.x) / (p3.y - p1.y);
        let mut xinc_23 = (p3.x - p2.x) / (p3.y - p2.y);

        let mut x13 = p1.x + xinc_13 * (1. - p1.y.fract());
        let mut x23 = p2.x + xinc_23 * (1. - p2.y.fract());

        let mut zinc_13 = (p3.z - p1.z) / (p3.y - p1.y);
        let mut zinc_23 = (p3.z - p2.z) / (p3.y - p2.y);

        let mut zleft = p1.z + zinc_13 * (1. - p1.y.fract());
        let mut zright = p2.z + zinc_23 * (1. - p2.y.fract());

        if x13 > x23 {
            std::mem::swap(&mut x13, &mut x23);
            std::mem::swap(&mut xinc_13, &mut xinc_23);

            std::mem::swap(&mut zleft, &mut zright);
            std::mem::swap(&mut zinc_13, &mut zinc_23);
        }

        let determinant = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);

        for y in p1.y.floor() as i64..p3.y.floor() as i64 {
            let zinc = (zright - zleft) / (x23 - x13);
            let mut z = zleft + zinc * (1. - x13.fract());

            for x in x13.ceil() as i64..x23.ceil() as i64 {
                let lambda1 = ((p2.y - p3.y) * (x as f64 - p3.x) + (p3.x - p2.x) * (y as f64 - p3.y)) / determinant;
                let lambda2 = ((p3.y - p1.y) * (x as f64 - p3.x) + (p1.x - p3.x) * (y as f64 - p3.y)) / determinant;
                let lambda3 = 1. - lambda1 - lambda2;
                unsafe {
                    let r = (c1.ch.r as f64 * lambda1 + c2.ch.r as f64 * lambda2 + c3.ch.r as f64 * lambda3).round() as u32;
                    let g = (c1.ch.g as f64 * lambda1 + c2.ch.g as f64 * lambda2 + c3.ch.g as f64 * lambda3).round() as u32;
                    let b = (c1.ch.b as f64 * lambda1 + c2.ch.b as f64 * lambda2 + c3.ch.b as f64 * lambda3).round() as u32;
                    let alpha = c1.col >> 24;
                    let color = alpha << 24 | r << 16 | g << 8 | b;
                    self.set_pixel_z(ivec2!(x, y), color, z);
                }
                z += zinc;
            }
            x23 += xinc_23;
            x13 += xinc_13;

            zleft += zinc_13;
            zright += zinc_23;
        }
    }
    fn fill_flat_bottom(&mut self, p1: Vec3, p2: Vec3, p3: Vec3, c1: Color, c2: Color, c3: Color) {
        let mut xinc_12 = (p2.x - p1.x) / (p2.y - p1.y);
        let mut xinc_13 = (p3.x - p1.x) / (p3.y - p1.y);

        let mut x12 = p1.x + xinc_12 * (1. - p1.y.fract());
        let mut x13 = p1.x + xinc_13 * (1. - p1.y.fract());

        let mut zinc_12 = (p2.z - p1.z) / (p2.y - p1.y);
        let mut zinc_13 = (p3.z - p1.z) / (p3.y - p1.y);

        let mut zleft = p1.z + zinc_12 * (1. - p1.y.fract());
        let mut zright = p1.z + zinc_13 * (1. - p1.y.fract());

        if x12 > x13 {
            std::mem::swap(&mut x12, &mut x13);
            std::mem::swap(&mut xinc_13, &mut xinc_12);

            std::mem::swap(&mut zleft, &mut zright);
            std::mem::swap(&mut zinc_12, &mut zinc_13);
        }

        let determinant = (p2.y - p3.y) * (p1.x - p3.x) + (p3.x - p2.x) * (p1.y - p3.y);

        for y in p1.y.floor() as i64..p2.y.floor() as i64 {
            let zinc = (zright - zleft) / (x13 - x12);
            let mut z = zleft + zinc * (1. - x12.fract());

            for x in x12.ceil() as i64..x13.ceil() as i64 {
                let lambda1 = ((p2.y - p3.y) * (x as f64 - p3.x) + (p3.x - p2.x) * (y as f64 - p3.y)) / determinant;
                let lambda2 = ((p3.y - p1.y) * (x as f64 - p3.x) + (p1.x - p3.x) * (y as f64 - p3.y)) / determinant;
                let lambda3 = 1. - lambda1 - lambda2;
                unsafe {
                    let r = (c1.ch.r as f64 * lambda1 + c2.ch.r as f64 * lambda2 + c3.ch.r as f64 * lambda3).round() as u32;
                    let g = (c1.ch.g as f64 * lambda1 + c2.ch.g as f64 * lambda2 + c3.ch.g as f64 * lambda3).round() as u32;
                    let b = (c1.ch.b as f64 * lambda1 + c2.ch.b as f64 * lambda2 + c3.ch.b as f64 * lambda3).round() as u32;
                    let alpha = c1.col >> 24;
                    let color = alpha << 24 | r << 16 | g << 8 | b;
                    self.set_pixel_z(ivec2!(x, y), color, z);
                }
                z += zinc;
            }
            x13 += xinc_13;
            x12 += xinc_12;

            zleft += zinc_12;
            zright += zinc_13;
        }
    }

    fn draw_triangle_clip_z(
        &mut self,
        p1: Vec4, p2: Vec4, p3: Vec4,
        c1: Color, c2: Color, c3: Color,
    ) {
        let p1 = self.clip_to_screen_v3(p1);
        let p2 = self.clip_to_screen_v3(p2);
        let p3 = self.clip_to_screen_v3(p3);

        let edge12 = p2 - p1;
        let edge13 = p3 - p1;
        let winding_order = edge12.cross(edge13);
        if winding_order.z.is_sign_negative() {
            //self.fill_triangle_z(p1, p2, p3, RED);
            return;
        }

        self.fill_triangle_z_f(p1, p2, p3, c1, c2, c3);
    }

    fn draw_triangle_clip_wire(&mut self, p1: Vec4, p2: Vec4, p3: Vec4, color: u32) {
        let p1 = self.clip_to_screen(p1);
        let p2 = self.clip_to_screen(p2);
        let p3 = self.clip_to_screen(p3);

        let edge12 = p2 - p1;
        let edge13 = p3 - p1;
        let winding_order = edge12.cross(edge13);
        if winding_order.is_negative() {
            return;
        }

        self.line(&p1, &p2, color);
        self.line(&p2, &p3, color);
        self.line(&p3, &p1, color);
    }

    pub fn draw_indexed(&mut self, vertex_buffer: &[VertexIn], index_buffer: &[u32], sort: bool) {
        struct Sort {
            indecies: [u32; 3],
            weight: f64,
        }
        let normals = &[
            vec3!(0., 0., 1.),
            vec3!(0., 1., 0.),
            vec3!(0., -1., 0.),
            vec3!(1., 0., 0.),
            vec3!(-1., 0., 0.),
        ];
        let plane_p = &[
            vec3!(0., 0., 0.),
            vec3!(0., -1., 0.),
            vec3!(0., 1., 0.),
            vec3!(-1., 0., 0.),
            vec3!(1., 0., 0.),
        ];
        let vert_size = vertex_buffer.len() as u32;
        let mut triangles: Vec<TriangleV4> = Vec::with_capacity(index_buffer.len() / 3);

        for t in index_buffer.chunks_exact(3) {
            if t[0] >= vert_size || t[1] >= vert_size || t[2] >= vert_size {
                return;
            }
            //println!("GPU::draw_indexed : drawing triangle {} {} {}",t[0],t[1],t[2]);

            let p1 = &vertex_buffer[t[0] as usize];
            let p2 = &vertex_buffer[t[1] as usize];
            let p3 = &vertex_buffer[t[2] as usize];

            let vertex_shader = self.vertex_shader;

            let p1 = vertex_shader(p1, &mut self.binds);
            let p2 = vertex_shader(p2, &mut self.binds);
            let p3 = vertex_shader(p3, &mut self.binds);

            let pos1 = p1.clip_pos();
            let pos2 = p2.clip_pos();
            let pos3 = p3.clip_pos();

            let fragment_shader = self.fragment_shader;

            let color1 = Color {
                col: fragment_shader(&p1, &mut self.binds),
            };
            let color2 = Color {
                col: fragment_shader(&p2, &mut self.binds),
            };
            let color3 = Color {
                col: fragment_shader(&p3, &mut self.binds),
            };

            let mut triangle_from_clip = vec![TriangleV4 {
                vert: [pos1, pos2, pos3],
                col: [color1, color2, color3],
            }];

            // Clip triangles to the screen
            for i in 0..normals.len() {
                let mut out: Vec<TriangleV4> = Vec::new();
                for t in &triangle_from_clip {
                    out.append(&mut clip_triangle_on_plane(normals[i], plane_p[i], *t));
                }
                triangle_from_clip = out;
            }
            triangles.append(&mut triangle_from_clip);
        }
        if sort {
            let sorted_indeces = get_z_sorted_indeces(&mut triangles);
            for (_, i) in sorted_indeces {
                self.draw_triangle_clip_z(
                    triangles[i].vert[0],
                    triangles[i].vert[1],
                    triangles[i].vert[2],
                    triangles[i].col[0],
                    triangles[i].col[1],
                    triangles[i].col[2],
                );
                // FOR WIREFRAME
                //self.draw_triangle_clip_wire(triangles[i][0], triangles[i][1], triangles[i][2], RED);
            }
        } else {
            for t in triangles {
                self.draw_triangle_clip_z(
                    t.vert[0], t.vert[1], t.vert[2], t.col[0], t.col[1], t.col[2],
                );
            }
        }
    }
}
fn get_z_sorted_indeces(triangles: &mut Vec<TriangleV4>) -> Vec<(f64, usize)> {
    let mut weights: Vec<(f64, usize)> = Vec::with_capacity(triangles.len());
    for (i, t) in triangles.iter().enumerate() {
        // Not dividing by w because already did that when clippling and w == 1
        let weight = (t.vert[0].z + t.vert[1].z + t.vert[2].z) / 3.;
        weights.push((weight, i));
    }
    weights.sort_by(|a, b| (b.0).total_cmp(&a.0));
    weights
}
fn clip_triangle_on_plane(
    plane_norm: Vec3,
    plane_p: Vec3,
    og_triangle: TriangleV4,
) -> Vec<TriangleV4> {
    let mut inside_index_list: Vec<usize> = Vec::new();
    let mut outside_index_list: Vec<usize> = Vec::new();

    let mut triangle = og_triangle;
    triangle.vert[0] = triangle.vert[0] / triangle.vert[0].w;
    triangle.vert[1] = triangle.vert[1] / triangle.vert[1].w;
    triangle.vert[2] = triangle.vert[2] / triangle.vert[2].w;

    // TODO: get rid of this
    if (1. - triangle.vert[0].y.abs()).abs() < 0.000_000_1
        && (triangle.vert[0].y != 1. || triangle.vert[0].y != -1.)
    {
        triangle.vert[0].y = 1. * triangle.vert[0].y.signum();
    }
    if (1. - triangle.vert[1].y.abs()).abs() < 0.000_000_1
        && (triangle.vert[1].y != 1. || triangle.vert[1].y != -1.)
    {
        triangle.vert[1].y = 1. * triangle.vert[1].y.signum();
    }
    if (1. - triangle.vert[2].y.abs()).abs() < 0.000_000_1
        && (triangle.vert[2].y != 1. || triangle.vert[0].y != -1.)
    {
        triangle.vert[2].y = 1. * triangle.vert[2].y.signum();
    }
    ////////////////////////

    for i in 0..3 {
        let plane_to_point = og_triangle.vert[i].to_vec3() - plane_p;
        let dot_point = plane_norm.dot(plane_to_point);

        if dot_point < 0. {
            outside_index_list.push(i);
        } else {
            inside_index_list.push(i);
        }
    }
    match inside_index_list.len() {
        0 => {
            return Vec::new();
        }
        1 => {
            #[rustfmt::skip]
            let (intersect_1,ratio_1) = line_plane_intersect( plane_p, plane_norm,
                triangle.vert[inside_index_list[0]].to_vec3(),
                triangle.vert[outside_index_list[0]].to_vec3(),
            );

            #[rustfmt::skip]
            let (intersect_2,ratio_2) = line_plane_intersect( plane_p, plane_norm,
                triangle.vert[inside_index_list[0]].to_vec3(),
                triangle.vert[outside_index_list[1]].to_vec3(),
            );
            // if ratio == 0 then the color at the intersect is the color of the start vertex
            let color_intersect1 = triangle.col[outside_index_list[0]] * ratio_1 + triangle.col[inside_index_list[0]] * (1. - ratio_1);
            let color_intersect2 = triangle.col[outside_index_list[1]] * ratio_2 + triangle.col[inside_index_list[0]] * (1. - ratio_2);

            let mut out_triangle = TriangleV4::new(
                triangle.vert[inside_index_list[0]],
                intersect_1.to_vec4(1.),
                intersect_2.to_vec4(1.),
                og_triangle.col[0],
                og_triangle.col[1],
                og_triangle.col[2],
                //triangle.col[inside_index_list[0]],
                //color_intersect1,
                //color_intersect2,
            );
            // If the index is 1 the out trianles will have a reverse winding order so i have to
            // account for that; if theres a better way to avoid this pls let me know
            if inside_index_list[0] == 1 {
                out_triangle.vert.swap(1, 2);
            }
            return vec![out_triangle];
        }
        2 => {
            #[rustfmt::skip]
            let (intersect_1,ratio_1) = line_plane_intersect( plane_p, plane_norm,
                triangle.vert[inside_index_list[0]].to_vec3(),
                triangle.vert[outside_index_list[0]].to_vec3(),
            );

            #[rustfmt::skip]
            let (intersect_2,ratio_2) = line_plane_intersect( plane_p, plane_norm,
                triangle.vert[inside_index_list[1]].to_vec3(),
                triangle.vert[outside_index_list[0]].to_vec3(),
            );

            let mut out_triangle_1 = TriangleV4::new(
                triangle.vert[inside_index_list[0]],
                triangle.vert[inside_index_list[1]],
                intersect_1.to_vec4(1.),
                og_triangle.col[0],
                og_triangle.col[1],
                og_triangle.col[2],
            );
            let mut out_triangle_2 = TriangleV4::new(
                intersect_2.to_vec4(1.),
                intersect_1.to_vec4(1.),
                triangle.vert[inside_index_list[1]],
                og_triangle.col[0],
                og_triangle.col[1],
                og_triangle.col[2],
            );
            // If the index is 1 the out trianles will have a reverse winding order so i have to
            // account for that; if theres a better way to avoid this pls let me know
            if outside_index_list[0] == 1 {
                out_triangle_1.vert.swap(1, 2);
                out_triangle_2.vert.swap(1, 2);
            }
            return vec![out_triangle_1, out_triangle_2];
        }
        3 => {
            return vec![triangle];
        }
        _ => panic!(),
    }
}

fn line_plane_intersect(plnae_p: Vec3, plane_n: Vec3, start: Vec3, end: Vec3) -> (Vec3,f64) {
    let line_d = end - start;
    let plane_d = plane_n.dot(plnae_p);

    let t = (plane_d - plane_n.dot(start)) / plane_n.dot(line_d);
    let line_t = line_d * t;
    let intersect = start + line_t;
    (intersect, line_t.mag()/line_d.mag())
}
