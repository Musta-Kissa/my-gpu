use crate::SurfaceConfig;
use crate::RED;
use crate::WHITE;
use crate::GREEN;
use crate::MAGENTA;
use minifb;
use my_math::matrix::Matrix;

use crate as my_gpu;
use std::any::Any;

struct Window {
    window: minifb::Window,
    framebuffer: Framebuffer,
}
impl Window {
    fn new(name: &str, width: usize, height: usize) -> Self {
        let mut window =
            minifb::Window::new(name, width, height, minifb::WindowOptions::default()).unwrap();

        window.set_target_fps(60);

        let framebuffer = Framebuffer::new(width, height);
        Self {
            window,
            framebuffer,
        }
    }
    fn is_open(&self) -> bool {
        self.window.is_open()
    }
    fn update(&mut self) {
        self.window.update()
    }
    fn display(&mut self) {
        let _ = self.window.update_with_buffer(
            &self.framebuffer.data,
            self.framebuffer.width,
            self.framebuffer.height,
        );
    }
}
struct Framebuffer {
    data: Vec<u32>,
    width: usize,
    height: usize,
}
impl Framebuffer {
    fn new(width: usize, height: usize) -> Self {
        let data = vec![0; width * height];
        Self {
            data,
            width,
            height,
        }
    }
}

use my_math::vec::IVec2;

fn is_in_triangle(p1: IVec2, p2: IVec2, p3: IVec2, point: IVec2) -> bool {
    let cross_12 = (point.x - p1.x) * (p2.y - p1.y) - (point.y - p1.y) * (p2.x - p1.x);
    let cross_23 = (point.x - p2.x) * (p3.y - p2.y) - (point.y - p2.y) * (p3.x - p2.x);
    let cross_31 = (point.x - p3.x) * (p1.y - p3.y) - (point.y - p3.y) * (p1.x - p3.x);

    let all_pos = cross_12.is_positive() && cross_23.is_positive() && cross_31.is_positive();
    let all_neg = cross_12.is_negative() && cross_23.is_negative() && cross_31.is_negative();

    all_pos ^ all_neg
}

use my_math::vec::Vec3;
use my_math::vec::Vec4;

struct Camera {
    pos: Vec3,
    up: Vec3,
    dir: Vec3,
    speed: f64,
    near: f64,
    far: f64,
    fov: f64,
}
impl Camera {
    fn right(&self) -> Vec3 {
        let dir = self.up.cross(self.dir);
        dir.norm()
    }
    fn left(&self) -> Vec3 {
        -1. * self.right()
    }
    fn forward(&self) -> Vec3 {
        let dir = self.right().cross(self.up);
        dir.norm()
    }
    fn back(&self) -> Vec3 {
        -1. * self.forward()
    }
}


fn gen_cube_mesh(pos:Vec3,color:u32) -> Vec<VertexIn> {
    let verts = [
            [pos.x+1.,pos.y+0.,pos.z+0.],
            [pos.x+1.,pos.y+1.,pos.z+0.],
            [pos.x+1.,pos.y+1.,pos.z+1.],
            [pos.x+1.,pos.y+0.,pos.z+1.],

            [pos.x+0.,pos.y+0.,pos.z+0.],
            [pos.x+0.,pos.y+0.,pos.z+1.],
            [pos.x+0.,pos.y+1.,pos.z+1.],
            [pos.x+0.,pos.y+1.,pos.z+0.],

            [pos.x+1.,pos.y+0.,pos.z+1.],
            [pos.x+1.,pos.y+1.,pos.z+1.],
            [pos.x+0.,pos.y+1.,pos.z+1.],
            [pos.x+0.,pos.y+0.,pos.z+1.],

            [pos.x+1.,pos.y+0.,pos.z+0.],
            [pos.x+0.,pos.y+0.,pos.z+0.],
            [pos.x+0.,pos.y+1.,pos.z+0.],
            [pos.x+1.,pos.y+1.,pos.z+0.],

            [pos.x+0.,pos.y+1.,pos.z+0.],
            [pos.x+0.,pos.y+1.,pos.z+1.],
            [pos.x+1.,pos.y+1.,pos.z+1.],
            [pos.x+1.,pos.y+1.,pos.z+0.],

            [pos.x+0.,pos.y+0.,pos.z+0.],
            [pos.x+1.,pos.y+0.,pos.z+0.],
            [pos.x+1.,pos.y+0.,pos.z+1.],
            [pos.x+0.,pos.y+0.,pos.z+1.],
    ];
    let normals = [
            [ 1.,0.,0. ],
            [ -1.,0.,0. ],
            [ 0.,0.,1. ],
            [ 0.,0.,-1. ],
            [ 0.,1.,0. ],
            [ 0.,-1.,0. ],
            ];
    let mut out = Vec::new();
    for (i,quad) in verts.chunks_exact(4).enumerate() {
        for j in 0..4 {
            out.push(VertexIn {
                pos: Vec3::from_slice(&quad[j]),
                color: color,
                norm: Vec3::from_slice(&normals[i]),
            })
        }
    }
    out
}

use self::my_gpu::Binds;
use self::my_gpu::ClipPos;

pub struct VertexIn {
    pos: Vec3,
    color: u32,
    norm: Vec3,
}
pub struct VertexOut {
    clip_pos: Vec4,
    color: u32,
    norm: Vec3,
}
impl ClipPos for VertexOut {
    fn clip_pos(&self) -> Vec4 {
        self.clip_pos
    }
}

fn vertex(vert:&VertexIn , binds: &mut Binds) -> VertexOut {
    let bind0: Matrix<4,4> = *binds.cast_ref(0).unwrap();

    let out = VertexOut {
        clip_pos: bind0 * vert.pos.to_vec4(1.0),
        color: vert.color,
        norm: vert.norm,
    };
    out
}

fn fragment(vert:&VertexOut, binds: &mut Binds) -> u32 {
    let light_dir:Vec3 = *binds.cast_ref(1).unwrap();

    let ambient = 0b0000_0100;

    let dot_light = light_dir.norm().dot(vert.norm);
    let dim_ratio = (dot_light + 1.)/2.;

    let r_col = (( (vert.color << 8) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let g_col = (( (vert.color << 16) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let b_col = (( (vert.color << 24) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let mut out = 0u32;
    out = out | (r_col << 16);
    out = out | (g_col << 8);
    out = out | (b_col << 0);
    out
}

    
pub fn gen_cube_indeces(vert_len: usize) -> Vec<u32> {
    let mut indices: Vec<u32> = Vec::new();
    indices.reserve_exact(vert_len * 10 / 4);
    //clockwise winding
    for i in 0..(vert_len as u32) / 4 {
        indices.extend([
            0 + 4 * i,
            1 + 4 * i,
            2 + 4 * i,
            2 + 4 * i,
            3 + 4 * i,
            0 + 4 * i,
        ]);
    }
    indices
}


#[test]
fn main() {
    let mut window = Window::new("minifb", 600 * 16 / 9, 600);
    let surface_config = SurfaceConfig {
        width: window.framebuffer.width,
        height: window.framebuffer.height,
    };

    let mut camera = Camera {
        up: vec3!(0., 1., 0.),
        pos: vec3!(0., 0., -1.),
        dir: vec3!(0., 0., 1.),
        speed: 0.05,
        near: 0.1,
        far: 100.,
        fov: 60.,
    };

    let mut cam_trans_mat = my_math::matrix::look_at_lh(camera.pos, camera.pos + camera.dir, camera.up);
    let mut proj = my_math::matrix::proj_mat_wgpu(camera.fov, 16. / 9., camera.near, camera.far);
    let mut view_proj = proj * cam_trans_mat;

    let config = my_gpu::Config {
        surface_cofig: surface_config,
    };
    let mut light_dir = vec3!(-5.,-2.,-4.);

    let mut binds: Binds = Binds(Vec::new());

    binds.push(&mut view_proj);
    binds.push(&mut light_dir);

    let mut verts = gen_cube_mesh(vec3!(0.,0.,0.),WHITE);
    let mut verts_sun = gen_cube_mesh(light_dir,MAGENTA);
    let mut verts_green= gen_cube_mesh(vec3!(light_dir.x * -1.,light_dir.y * -1.,light_dir.z),GREEN);
    let mut indices = gen_cube_indeces(verts.len());
    let mut indices_sun = gen_cube_indeces(verts_sun.len() + verts.len());
    let mut indices_green = gen_cube_indeces(verts_green.len() + verts_sun.len() + verts.len());

    verts.append(&mut verts_sun);
    verts.append(&mut verts_green);
    indices.append(&mut indices_sun);
    indices.append(&mut indices_green);

    let mut gpu = my_gpu::Gpu::new(
        config,
        window.framebuffer.data.as_mut_ptr(),
        binds,
        vertex,
        fragment,
    );

    'draw_loop: while window.is_open() {
        gpu.clear(my_gpu::BLUE);
        gpu.draw_indexed(&verts,&indices);

        for key in window.window.get_keys() {
            use minifb::Key;
            match key {
                Key::Space => camera.pos = camera.pos + camera.speed * camera.up, // Up
                Key::LeftShift => camera.pos = camera.pos - camera.speed * camera.up, // Down

                Key::W => camera.pos = camera.pos + camera.speed * camera.forward(),
                Key::S => camera.pos = camera.pos + camera.speed * camera.back(),

                Key::D => camera.pos = camera.pos + camera.speed * camera.right(),
                Key::A => camera.pos = camera.pos + camera.speed * camera.left(),

                Key::K => {
                    camera.speed = camera.speed * 1.05;
                    println!("speed: {}", camera.speed);
                }
                Key::J => {
                    camera.speed = camera.speed * 0.95;
                    println!("speed: {}", camera.speed);
                }

                Key::O => camera.fov += 1.,
                Key::P => camera.fov -= 1.,

                Key::G => {
                    println!("camera pos:{:?}", camera.pos);
                }
                Key::H => {
                    light_dir.rot_quat(1.,vec3!(-1.,0.,1.));
                }
                Key::R => {
                    camera.up = vec3!(0.,1.,0.); 
                }

                Key::Escape => break 'draw_loop,

                Key::Left => camera.dir.rot_quat(-1., camera.up), //Yaw Left
                Key::Right => camera.dir.rot_quat(1., camera.up), //Yaw Right
                Key::E => camera.up.rot_quat(-1., camera.dir),    //Roll Right
                Key::Q => camera.up.rot_quat(1., camera.dir),     //Roll Left
                Key::Up => {
                    // clamp the angle
                    if camera.dir.dot(camera.up) > 0.9 {
                        continue;
                    }
                    //Pitch Up
                    camera.dir.rot_quat(1., camera.dir.cross(camera.up).norm());
                }
                Key::Down => {
                    // clamp the angle
                    if camera.dir.dot(camera.up) < -0.9 {
                        continue;
                    }
                    //Pitch Down
                    camera.dir.rot_quat(-1., camera.dir.cross(camera.up).norm());
                }

                _ => (),
            }
        }
        if !window.window.get_keys().is_empty() {
            cam_trans_mat = my_math::matrix::look_at_lh(camera.pos, camera.pos + camera.dir, camera.up);
            proj = my_math::matrix::proj_mat_wgpu(camera.fov, 16. / 9., camera.near, camera.far);
            view_proj = proj * cam_trans_mat;

            verts = gen_cube_mesh(vec3!(0.,0.,0.),WHITE);
            verts_sun = gen_cube_mesh(light_dir,MAGENTA);
            verts_green= gen_cube_mesh(vec3!(light_dir.x * -1.,light_dir.y * -1.,light_dir.z),GREEN);
            indices = gen_cube_indeces(verts.len());
            indices_sun = gen_cube_indeces(verts_sun.len() + verts.len());
            indices_green = gen_cube_indeces(verts_green.len() + verts_sun.len() + verts.len());

            verts.append(&mut verts_sun);
            verts.append(&mut verts_green);
            indices.append(&mut indices_sun);
            indices.append(&mut indices_green);
        }

        window.display();
    }
}
