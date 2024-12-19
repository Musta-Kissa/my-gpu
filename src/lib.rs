#[cfg(test)]
mod test;

#[macro_use]
extern crate my_math;

use std::cmp::max;
use std::cmp::min;
use std::any::Any;

use std::ops::Deref;
use std::ops::DerefMut;
use std::ops::Mul;

use my_math::vec::IVec2;
use my_math::vec::Vec3;
use my_math::vec::Vec4;
use my_math::matrix::Matrix;

const WHITE: u32 = !0u32;

fn line_plane_intersect(plnae_p: Vec3, plane_n: Vec3, start: Vec3, end: Vec3) -> Vec3 {
    let line_d = end - start;
    let plane_d = plane_n.dot(plnae_p);

    let t = (plane_d - plane_n.dot(start)) / plane_n.dot(line_d);

    let intersect = start + line_d * t;
    intersect
}


#[derive(Clone, Copy)]
struct Triangle([Vec4; 3]);
impl Triangle {
    fn new(p1: Vec4, p2: Vec4, p3: Vec4) -> Self {
        Triangle { 0: [p1, p2, p3] }
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
        return &self.0;
    }
}
impl DerefMut for Triangle {
    fn deref_mut(&mut self) -> &mut Self::Target {
        return &mut self.0;
    }
}

pub struct Binds(Vec<Box<dyn Any>>);

impl Binds {
    pub fn cast_ref<'a,T :'static>(&'a self,idx: usize) -> &'a T {
        self.0[idx].downcast_ref::<T>().unwrap()
    }
}

impl Deref for Binds {
    type Target = Vec<Box<dyn Any>>;

    fn deref(&self) -> &Self::Target {
       &self.0 
    }
}
impl DerefMut for Binds {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0    
    }
}

pub struct SurfaceConfig {
    pub width: usize,
    pub height: usize,
}
pub struct Config {
    pub surface_cofig: SurfaceConfig,
}
pub struct Gpu<'s> {
    config: Config,

    surface: *mut u32,
    binds: Binds,

    vertex_shader: fn([f64;3],&mut Binds) -> Vec4,

    vertex_buffer: &'s [[f64;3]],
    index_buffer: Option<&'s [u16]>,
}

impl<'s> Gpu<'s> {
    pub fn new(config: Config, surface: *mut u32,binds: Binds, vertex_shader: fn([f64;3],&mut Binds) -> Vec4, vertex_buffer: &'s [[f64;3]],  index_buffer: Option<&'s [u16]>) -> Self {
        Gpu {
            config,
            surface,
            binds,
            vertex_shader,
            vertex_buffer,
            index_buffer,
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
            *self.surface.add(pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize) = 
                *self.surface.add(pixel.y as usize * self.config.surface_cofig.width + pixel.x as usize) | color;
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
        let s_y: i64 = ((-pos.y + 1.) * self.config.surface_cofig.height as f64 / 2.).floor() as i64;
        ivec2!(s_x, s_y)
    }

    pub fn clear(&mut self, color: u32) {
        let surface_len = self.config.surface_cofig.height * self.config.surface_cofig.width; 
        for i in 0..surface_len{
            unsafe {
                *self.surface.add(i) = color;
            }
        }
    }
    fn is_in_triangle( p1: IVec2, p2: IVec2, p3: IVec2, p_x: i64, p_y: i64) -> bool {
        let cross_12 = (p_x - p1.x) * (p2.y - p1.y) - (p_y - p1.y) * (p2.x - p1.x);
        let cross_23 = (p_x - p2.x) * (p3.y - p2.y) - (p_y - p2.y) * (p3.x - p2.x);
        let cross_31 = (p_x - p3.x) * (p1.y - p3.y) - (p_y - p3.y) * (p1.x - p3.x);

        let all_pos = cross_12.is_positive() && cross_23.is_positive() && cross_31.is_positive();
        let all_neg = cross_12.is_negative() && cross_23.is_negative() && cross_31.is_negative();

        all_pos ^ all_neg
    }

    fn draw_triangle_clip(&mut self, p1: Vec4, p2: Vec4, p3: Vec4) {
        let p1 = self.clip_to_screen(p1);
        let p2 = self.clip_to_screen(p2);
        let p3 = self.clip_to_screen(p3);

        let edge12 = p2 - p1;
        let edge13 = p3 - p1;
        let winding_order = edge12.cross(edge13);
        if winding_order.is_negative() {
            return;
        } 

        let right = max(max(p1.x, p2.x),p3.x);
        let left = min(min(p1.x, p2.x),p3.x);

        let top = max(max(p1.y, p2.y),p3.y);
        let bottom = min(min(p1.y, p2.y),p3.y);

        for x in left..right {
            for y in bottom..top {
                if Self::is_in_triangle(p1,p2,p3,x,y) {
                    self.set_pixel(ivec2!(x,y), WHITE);
                }
            }
        }
    }

    pub fn draw(&mut self) {
        for t in self.vertex_buffer.chunks_exact(3) {
            self.draw_triangle_clip(
                vec4!(t[0][0] , t[0][1], t[0][2],1.0),
                vec4!(t[1][0] , t[1][1], t[1][2],1.0),
                vec4!(t[2][0] , t[2][1], t[2][2],1.0),
            );
        }
    }
    fn clip_triangle_on_plane2(plane_norm: Vec3, plane_p: Vec3, og_triangle: Triangle) -> Vec<Triangle> {
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
                ).to_vec4();

                //println!("intersect_1: {:?}",intersect_1);

                #[rustfmt::skip]
                let intersect_2 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[0]].to_vec3(),
                    new_triangle[outside_index_list[1]].to_vec3(),
                ).to_vec4();
                //println!("intersect_2: {:?}",intersect_2);

                let mut out_triangle =
                    Triangle::new(new_triangle[inside_index_list[0]], intersect_1, intersect_2);
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
                ).to_vec4();
                //println!("intersect_1: {:?}",intersect_1);

                #[rustfmt::skip]
                let intersect_2 = line_plane_intersect( plane_p, plane_norm,
                    new_triangle[inside_index_list[1]].to_vec3(),
                    new_triangle[outside_index_list[0]].to_vec3(),
                ).to_vec4();
                //println!("intersect_2: {:?}",intersect_2);

                let mut out_triangle_1 = Triangle::new(
                    new_triangle[inside_index_list[0]],
                    new_triangle[inside_index_list[1]],
                    intersect_1,
                );
                let mut out_triangle_2 =
                    Triangle::new(intersect_2, intersect_1, new_triangle[inside_index_list[1]]);
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

    pub fn draw_indexed(&mut self) {
        let normals = &[ 
            vec3!(0., 0., 1.),
            vec3!(0., 1., 0.), 
            vec3!(0.,-1.,0.),
            vec3!(1.,0.,0.),
            vec3!(-1.,0.,0.),
        ];
        let plane_p = &[ 
            vec3!(0., 0., 0.),
            vec3!(0., -1., 0.), 
            vec3!(0.,1.,0.) ,
            vec3!(-1.,0.,0.) ,
            vec3!(1.,0.,0.) ,
        ];
        for t in self.index_buffer.unwrap().chunks_exact(3) {
            let p1 = self.vertex_buffer[t[0] as usize];
            let p2 = self.vertex_buffer[t[1] as usize];
            let p3 = self.vertex_buffer[t[2] as usize];

            let vertex_shader = self.vertex_shader;

            let p1 = vertex_shader(p1,&mut self.binds);
            let p2 = vertex_shader(p2,&mut self.binds);
            let p3 = vertex_shader(p3,&mut self.binds);
            
            let mut triangles = vec![Triangle::new(p1, p2, p3)];

            // Clip triangles to the screen 
            for i in 0..normals.len(){
                let mut out:Vec<Triangle> = Vec::new();
                for t in &triangles {
                out.append(&mut Self::clip_triangle_on_plane2(
                        normals[i],
                        plane_p[i],
                        *t,
                   ));
                }
                triangles = out;
            }

            for triangle in triangles {
                self.draw_triangle_clip(triangle[0], triangle[1], triangle[2]);
            }
        }
    }
}
