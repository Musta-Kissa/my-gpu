use crate::SurfaceConfig;
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
#[rustfmt::skip]
pub const VERTICES: &[[f64; 3]] = &[
    [-0.5, -0.5,  0.5 + 3.],
    [-0.5,  0.5,  0.5 + 3.],
    [ 0.5, -0.5,  0.5 + 3.],
    [ 0.5,  0.5,  0.5 + 3.],
    [-0.5, -0.5, -0.5 + 3.],
    [-0.5,  0.5, -0.5 + 3.],
    [ 0.5, -0.5, -0.5 + 3.],
    [ 0.5,  0.5, -0.5 + 3.],

    [-0.5, -0.5, -0.5 + 1.],
    [-0.5,  0.5, -0.5 + 1.],
    [ 0.5, -0.5, -0.5 + 1.],
];

#[rustfmt::skip]
pub const INDICES: &[u16] = &[
    0, 2, 1,  1, 2, 3, 
    6, 4, 7,  7, 4, 5, 
    4, 0, 5,  5, 0, 1, 
    2, 6, 3,  3, 6, 7, 
    4, 6, 0,  0, 6, 2, 
    1, 3, 5,  5, 3, 7, 
    8, 9, 10,
];

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

use self::my_gpu::Binds;

fn vertex(vert: [f64; 3], binds: &mut Binds) -> Vec4 {
    let bind0: Matrix<4,4> = *binds.cast_ref(0);

    let out = Vec4::from_slice(&vert);
    bind0 * out
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
        pos: vec3!(0.004214, 0., -2.006215412),
        dir: vec3!(0., 0., 1.),
        speed: 0.05,
        near: 0.1,
        far: 100.,
        fov: 60.,
    };

    let mut cam_trans_mat =
        my_math::matrix::look_at_lh(camera.pos, camera.pos + camera.dir, camera.up);
    let mut proj = my_math::matrix::proj_mat_wgpu(camera.fov, 16. / 9., camera.near, camera.far);
    let mut view_proj = proj * cam_trans_mat;

    let config = my_gpu::Config {
        surface_cofig: surface_config,
    };

    let mut binds: Binds = Binds( Vec::new() );
    binds.push(Box::new(view_proj));

    let mut gpu = my_gpu::Gpu::new(
        config,
        window.framebuffer.data.as_mut_ptr(),
        binds,
        vertex,
        VERTICES,
        Some(INDICES),
    );

    'draw_loop: while window.is_open() {
        gpu.clear(0u32);
        gpu.draw_indexed();

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
            gpu.binds[0] = Box::new(view_proj);
        }

        window.display();
    }
}
