#[cfg(test)]
mod test;

#[macro_use]
extern crate my_math;

use std::any::Any;

use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Mul;

use my_math::matrix::Matrix;
use my_math::vec::IVec2;
use my_math::vec::Vec3;
use my_math::vec::Vec4;

pub const WHITE: u32 = !0u32;
pub const BLACK: u32 = 0u32;
pub const RED: u32 = ((1u32 << 9) - 1) << 16;
pub const GREEN: u32 = ((1u32 << 9) - 1) << 8;
pub const BLUE: u32 = ((1u32 << 9) - 1) << 0;
pub const MAGENTA: u32 = RED | BLUE;


#[derive(Clone, Copy)]
struct Triangle{
    vert:[Vec4; 3],
    col: u32
}
impl Triangle {
    fn new(p1: Vec4, p2: Vec4, p3: Vec4,color:u32) -> Self {
        Triangle { vert: [p1, p2, p3], col: color}
    }
    /// Returns true if the triangle should be rendered
    fn winding_order(&self) -> bool {
        let edge12 = self[0].to_vec3() - self[1].to_vec3();
        let edge13 = self[2].to_vec3() - self[1].to_vec3();
        let winding_order = edge12.cross(edge13);
        if winding_order.z.is_sign_positive() {
            return true;
        } else {
            return false;
        }
    }
}
impl Mul<Triangle> for Matrix<4, 4> {
    type Output = Triangle;

    fn mul(self, rhs: Triangle) -> Self::Output {
        let mut out = rhs;
        out[0] = self * rhs[0];
        out[1] = self * rhs[1];
        out[2] = self * rhs[2];
        out
    }
}
impl Deref for Triangle {
    type Target = [Vec4; 3];
    fn deref(&self) -> &Self::Target {
        return &self.vert;
    }
}
impl DerefMut for Triangle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return &mut self.vert;
    }
}

pub struct Binds(Vec<*mut dyn Any>);

impl Binds {
    pub fn cast_ref<'a, T: 'static>(&'a self, idx: usize) -> Option<&'a T> {
        unsafe {
            (*self.0[idx]).downcast_ref::<T>()
        }
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

pub struct SurfaceConfig {
    pub width: usize,
    pub height: usize,
}
pub struct Config {
    pub surface_cofig: SurfaceConfig,
}
pub struct Gpu<VertexIn,VertexOut> {
    config: Config,

    surface: *mut u32,
    binds: Binds,

    vertex_shader: fn(&VertexIn, &mut Binds) -> VertexOut,
    fragment_shader: fn(&VertexOut, &mut Binds) -> u32,
}

#[allow(dead_code)]
impl<VertexIn,VertexOut: ClipPos> Gpu<VertexIn,VertexOut> {
    // The vertex out must have a field of clip_pos
    pub fn new(
        config: Config,
        surface: *mut u32,
        binds: Binds,
        vertex_shader: fn(&VertexIn, &mut Binds) -> VertexOut,
        fragment_shader: fn(&VertexOut,&mut Binds) -> u32,
    ) -> Self {
        Gpu {
            config,
            surface,
            binds,
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
        }
    }

    /// draws a filled triangle using the scanline fill algorithm
    fn fill_triangle(&mut self, p1: IVec2, p2: IVec2, p3: IVec2, color: u32) {
        let mut points = vec![p1, p2, p3];
        points.sort_by(|a, b| (a.y).cmp(&b.y));
        let p1 = points[0];
        let p2 = points[1];
        let p3 = points[2];

        if p2.y == p3.y {
            let slope12 = (p1.x - p2.x) as f64 / (p1.y - p2.y) as f64;
            let slope13 = (p1.x - p3.x) as f64 / (p1.y - p3.y) as f64;

            for y in p1.y..=p3.y {
                let mut x1 = (slope12 * (y - p2.y) as f64).round() as i64 + p2.x;
                let mut x2 = (slope13 * (y - p3.y) as f64).round() as i64 + p3.x;

                if x1 > x2 {
                    std::mem::swap(&mut x1, &mut x2);
                }

                for x in x1..=x2 {
                    self.set_pixel(ivec2!(x, y), color);
                }
            }
        } else if p1.y == p2.y {
            let slope13 = (p1.x - p3.x) as f64 / (p1.y - p3.y) as f64;
            let slope23 = (p2.x - p3.x) as f64 / (p2.y - p3.y) as f64;

            for y in p1.y..=p3.y {
                let mut x1 = (slope13 * (y - p3.y) as f64).round() as i64 + p3.x;
                let mut x2 = (slope23 * (y - p2.y) as f64).round() as i64 + p2.x;

                if x1 > x2 {
                    std::mem::swap(&mut x1, &mut x2);
                }

                for x in x1..=x2 {
                    self.set_pixel(ivec2!(x, y), color);
                }
            }
        } else {
            let slope = (p1.x - p3.x) as f64 / (p1.y - p3.y) as f64;
            let d_y = p2.y;
            let d_x = (slope * (p2.y - p3.y) as f64).ceil() as i64 + p3.x;
            let d = ivec2!(d_x, d_y);
            self.fill_triangle(p1, p2, d, color);
            self.fill_triangle(d, p2, p3, color);
        }
    }
    fn draw_triangle_clip(&mut self, p1: Vec4, p2: Vec4, p3: Vec4, color: u32) {
        let p1 = self.clip_to_screen(p1);
        let p2 = self.clip_to_screen(p2);
        let p3 = self.clip_to_screen(p3);

        let edge12 = p2 - p1;
        let edge13 = p3 - p1;
        let winding_order = edge12.cross(edge13);
        if winding_order.is_negative() {
            //self.fill_triangle(p1, p2, p3, RED);
            return;
        }

        self.fill_triangle(p1, p2, p3, color);
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

    pub fn draw_indexed(&mut self, vertex_buffer: &[VertexIn], index_buffer: &[u32]) {
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
        for t in index_buffer.chunks_exact(3) {
            let vert_size = vertex_buffer.len() as u32;
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

            let color1 = fragment_shader(&p1, &mut self.binds);
            //let color2 = fragment_shader(&p2, &mut self.binds);
            //let color3 = fragment_shader(&p3, &mut self.binds);

            let mut triangles = vec![Triangle::new(pos1, pos2, pos3,color1)];

            // Clip triangles to the screen
            for i in 0..normals.len() {
                let mut out: Vec<Triangle> = Vec::new();
                for t in &triangles {
                    out.append(&mut clip_triangle_on_plane(
                        normals[i], plane_p[i], *t,
                    ));
                }
                triangles = out;
            }

            for triangle in &triangles {
                self.draw_triangle_clip(triangle[0], triangle[1], triangle[2], triangle.col);
                // FOR WIREFRAME 
                // self.draw_triangle_clip_wire(triangle[0], triangle[1], triangle[2], triangle.col);
            }
        }
    }
}
fn clip_triangle_on_plane(
        plane_norm: Vec3,
        plane_p: Vec3,
        og_triangle: Triangle,
    ) -> Vec<Triangle> {
        let mut inside_index_list: Vec<usize> = Vec::new();
        let mut outside_index_list: Vec<usize> = Vec::new();

        let mut new_triangle = og_triangle;
        new_triangle[0] = new_triangle[0] / new_triangle[0].w;
        new_triangle[1] = new_triangle[1] / new_triangle[1].w;
        new_triangle[2] = new_triangle[2] / new_triangle[2].w;

        for i in 0..3 {
            let plane_to_point = og_triangle[i].to_vec3() - plane_p;
            let dot_point = plane_norm.dot(plane_to_point);

            if dot_point < 0. {
                //if og_triangle[i].z < 0. {
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
                let intersect_1 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[0]].to_vec3(),
                    new_triangle[outside_index_list[0]].to_vec3(),
                ).to_vec4(1.);

                //println!("intersect_1: {:?}",intersect_1);

                #[rustfmt::skip]
                let intersect_2 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[0]].to_vec3(),
                    new_triangle[outside_index_list[1]].to_vec3(),
                ).to_vec4(1.);
                //println!("intersect_2: {:?}",intersect_2);

                let mut out_triangle =
                    Triangle::new(new_triangle[inside_index_list[0]], intersect_1, intersect_2,og_triangle.col);
                // If the index is 1 the out trianles will have a reverse winding order so i have to
                // account for that; if theres a better way to avoid this pls let me know
                if inside_index_list[0] == 1 {
                    unsafe { core::ptr::swap(&mut out_triangle[1], &mut out_triangle[2]) }
                }
                return vec![out_triangle];
            }
            2 => {
                #[rustfmt::skip]
                let intersect_1 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[0]].to_vec3(),
                    new_triangle[outside_index_list[0]].to_vec3(),
                ).to_vec4(1.);

                #[rustfmt::skip]
                let intersect_2 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[1]].to_vec3(),
                    new_triangle[outside_index_list[0]].to_vec3(),
                ).to_vec4(1.);

                let mut out_triangle_1 = Triangle::new(
                    new_triangle[inside_index_list[0]],
                    new_triangle[inside_index_list[1]],
                    intersect_1,
                    og_triangle.col
                );
                let mut out_triangle_2 =
                    Triangle::new(intersect_2, intersect_1, new_triangle[inside_index_list[1]],og_triangle.col);
                // If the index is 1 the out trianles will have a reverse winding order so i have to
                // account for that; if theres a better way to avoid this pls let me know
                if outside_index_list[0] == 1 {
                    unsafe {
                        core::ptr::swap(&mut out_triangle_1[1], &mut out_triangle_1[2]);
                        core::ptr::swap(&mut out_triangle_2[1], &mut out_triangle_2[2]);
                    }
                }
                return vec![out_triangle_1, out_triangle_2];
            }
            3 => {
                return vec![new_triangle];
            }
            _ => panic!(),
        }
    }


fn line_plane_intersect(plnae_p: Vec3, plane_n: Vec3, start: Vec3, end: Vec3) -> Vec3 {
    let line_d = end - start;
    let plane_d = plane_n.dot(plnae_p);

    let t = (plane_d - plane_n.dot(start)) / plane_n.dot(line_d);

    let intersect = start + line_d * t;
    intersect
}

