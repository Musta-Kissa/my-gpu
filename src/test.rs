use crate::SurfaceConfig;
use crate::VertexPos;
use crate::ALPHA_HALF;
use crate::LIGHT_BLUE;
use crate::RED;
use crate::RED_TRANSPARENT;
use crate::WHITE;
use crate::BLUE;
use crate::GREEN;
use crate::MAGENTA;
use crate::YELLOW;
use minifb;
use my_math::matrix::Matrix;
use std::time::Instant;

use crate as my_gpu;

struct Window {
    window: minifb::Window,
    framebuffer: Framebuffer,
}
impl Window {
    fn new(name: &str, width: usize, height: usize) -> Self {
        let mut window =
            minifb::Window::new(name, width, height, minifb::WindowOptions { resize: true , ..minifb::WindowOptions::default() } ).unwrap();

        //window.set_target_fps(60);

        let framebuffer = Framebuffer::new(width, height);
        Self {
            window,
            framebuffer,
        }
    }
    fn is_open(&self) -> bool {
        self.window.is_open()
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
impl VertexPos for VertexIn {
    fn vertex_pos(&self) -> Vec3 {
        self.pos
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

    let ambient = 0b0000_0111;

    let dot_light = light_dir.norm().dot(vert.norm);
    let dim_ratio = (dot_light + 1.)/2.;

    let r_col = (( (vert.color << 08) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let g_col = (( (vert.color << 16) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let b_col = (( (vert.color << 24) >> 24) as f64 * dim_ratio).round() as u32 | ambient;
    let mut out = 0u32;
    out = out | (r_col << 16);
    out = out | (g_col << 8);
    out = out | (b_col << 0);
    out = out | (vert.color >> 24) << 24;
    out
}

const RES: usize = 1080;
#[test]
fn main() {
    let mut window = Window::new("minifb", RES * 16 / 9, RES);
    let surface_config = SurfaceConfig {
        width: window.framebuffer.width,
        height: window.framebuffer.height,
    };

    let mut camera = Camera {
        up: vec3!(0., 1., 0.),
        //pos: Vec3 { x: 5.779785901190033, y: 4.8049999999999935, z: -9.057020099801605 },
        //dir: Vec3 { x: -0.4866490490059816, y: -0.23339999999999939, z: 0.873546408327325 },
        pos: vec3!(2.5,2.5,-5.),
        dir: vec3!(0.,0.,1.),
        speed: 0.005,
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

    let teapot_mesh1 = Mesh::from_obj("./teapot2.obj".into(),RED_TRANSPARENT,vec3!(1.,1.,-6.));
    let teapot_mesh2 = Mesh::from_obj("./teapot2.obj".into(),LIGHT_BLUE,vec3!(3.,2.,1.));
    let teapot_mesh3 = Mesh::from_obj("./teapot2.obj".into(),YELLOW,vec3!(1.,2.,-4.));
    let cube_mesh1 = Mesh::from_obj("./cube2.obj".into(),ALPHA_HALF | GREEN,vec3!(-5.,-2.,-3.));
    let mut cube_sun_mesh = Mesh::from_obj("./cube2.obj".into(),WHITE,light_dir);

    let mut gpu = my_gpu::Gpu::new(
        config,
        window.framebuffer.data.as_mut_ptr(),
        binds,
        vertex,
        fragment,
    );
    let test_tri_vert = &[
        VertexIn {
            pos: vec3!(0.,0.,0.),
            color: GREEN,
            norm: Vec3::UP,
        },
        VertexIn {
            pos: vec3!(2.5,5.,0.),
            color: RED,
            norm: Vec3::UP,
        },
        VertexIn {
            pos: vec3!(5.,0.,0.),
            color: BLUE,
            norm: Vec3::UP,
        },
    ];

    'draw_loop: while window.is_open() {
        let start = Instant::now();

        gpu.clear(my_gpu::LIGHT_BLUE);
        //gpu.draw_indexed(&cube_sun_mesh.verts,&cube_sun_mesh.indices,false);
        //
        //gpu.draw_indexed(&teapot_mesh2.verts,&teapot_mesh2.indices,false);
        //gpu.draw_indexed(&teapot_mesh3.verts,&teapot_mesh3.indices,false);
        gpu.draw_indexed(test_tri_vert,&vec![0,1,2],false);
//
        //gpu.draw_indexed(&cube_mesh1.verts,&cube_mesh1.indices,true);
        //gpu.draw_indexed(&teapot_mesh1.verts,&teapot_mesh1.indices,true);
        window.display();

        let d_t = start.elapsed().as_millis() as f64;
        window.window.set_title(&format!("{:.2}fps ({}ms)",1./(d_t / 1000.),d_t));

        for key in window.window.get_keys() {
            use minifb::Key;
            match key {
                Key::Space => camera.pos = camera.pos + camera.speed  * d_t * camera.up, // Up
                Key::LeftShift => camera.pos = camera.pos - camera.speed * d_t * camera.up, // Down

                Key::W => camera.pos = camera.pos + camera.speed * d_t * camera.forward(),
                Key::S => camera.pos = camera.pos + camera.speed * d_t * camera.back(),

                Key::D => camera.pos = camera.pos + camera.speed * d_t * camera.right(),
                Key::A => camera.pos = camera.pos + camera.speed * d_t * camera.left(),

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

                Key::C => {
                    println!("camera pos:{:?}", camera.pos);
                    println!("camera dir:{:?}", camera.dir);
                }
                Key::H => {
                    light_dir.rot_quat(1.,vec3!(-1.,0.,1.));
                }
                Key::R => {
                    camera.up = vec3!(0.,1.,0.); 
                }

                Key::Escape => break 'draw_loop,

                Key::Left => camera.dir.rot_quat(-1. * d_t / 16., camera.up), //Yaw Left
                Key::Right => camera.dir.rot_quat(1. * d_t / 16., camera.up), //Yaw Right
                Key::E => camera.up.rot_quat(-1. * d_t / 16., camera.dir),    //Roll Right
                Key::Q => camera.up.rot_quat(1. * d_t / 16., camera.dir),     //Roll Left
                Key::Up => {
                    // clamp the angle
                    if camera.dir.dot(camera.up) > 0.9 {
                        continue;
                    }
                    //Pitch Up
                    camera.dir.rot_quat(1. * d_t / 16., camera.dir.cross(camera.up).norm());
                }
                Key::Down => {
                    // clamp the angle
                    if camera.dir.dot(camera.up) < -0.9 {
                        continue;
                    }
                    //Pitch Down
                    camera.dir.rot_quat(-1. * d_t / 16., camera.dir.cross(camera.up).norm());
                }
                _ => (),
            }
        }
        if !window.window.get_keys().is_empty() {
            cam_trans_mat = my_math::matrix::look_at_lh(camera.pos, camera.pos + camera.dir, camera.up);
            proj = my_math::matrix::proj_mat_wgpu(camera.fov, 16. / 9., camera.near, camera.far);
            view_proj = proj * cam_trans_mat;
            cube_sun_mesh = Mesh::from_obj("./cube2.obj".to_string(),WHITE,light_dir); 
        }
    }
}

use std::fs::File;
use std::io::{BufReader,BufRead};
pub struct Mesh {
    verts: Vec<VertexIn>,
    indices: Vec<u32>,
}
impl Mesh {
    pub fn from_obj(path: String, color:u32,pos:Vec3) -> Mesh {
        let file = File::open(path).unwrap();  
        
        let reader = BufReader::new(file);

        let mut out_verts: Vec<VertexIn> = Vec::new();
        let mut out_indeces: Vec<u32> = Vec::new();

        let mut verts: Vec<Vec3> = Vec::new();
        let mut normals: Vec<Vec3> = Vec::new();

        for (line_num,line) in reader.lines().enumerate() {
            //print!("line {} ",line_num);
            let line = line.unwrap_or("".to_string());
            let line = line.trim();
            
            let tokens:Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() {
                //println!("empty");
                continue;
            }
            match tokens[0] {
                "v" => {
                    //println!("vert");
                    let x:f64 = tokens[1].to_string().parse().expect("couldnt parse x of vert");
                    let y:f64 = tokens[2].to_string().parse().expect("couldnt parse y of vert");
                    let z:f64 = tokens[3].to_string().parse().expect("couldnt parse z of vert");
                    verts.push(vec3!(x,y,z)+pos);
                }
                "vn" => {
                    //println!("vert normal");
                    let x:f64 = tokens[1].to_string().parse().expect("couldnt parse x of normal");
                    let y:f64 = tokens[2].to_string().parse().expect("couldnt parse y of normal");
                    let z:f64 = tokens[3].to_string().parse().expect("couldnt parse z of normal");
                    normals.push(vec3!(x,y,z));
                }
                "f" => {
                    //print!("face: ");
                    if tokens.len() == 4 {
                        //println!("triangle");
                        let idx_count = out_indeces.len() as u32;
                        out_indeces.append(&mut vec![0,1,2].iter().map(|i| idx_count + i).collect());
                        
                        let mut t_verts = Vec::new();
                        for i in 1..4 {
                            let indexes:Vec<&str> = tokens[i].split("/").collect();
                            let v_index:usize = indexes[0].parse().unwrap();
                            let v = verts[v_index -1];
                            t_verts.push(v);
                        }

                        let mut t_norms = Vec::new();

                        let indexes:Vec<&str> = tokens[1].split("/").collect();
                        let has_norms = indexes.get(2).is_some();

                        if has_norms {
                            for i in 1..4 {
                                let indexes:Vec<&str> = tokens[i].split("/").collect();
                                let n_index:usize = indexes[2].parse().unwrap();
                                let n = normals[n_index -1];
                                t_norms.push(n);
                            }
                        } else {
                            let norm = calculate_normals(t_verts[0],t_verts[1],t_verts[2]);
                            for i in 1..4 {
                                t_norms.push(norm);
                            }
                        }
                        out_verts.push(VertexIn {
                            pos: t_verts[0],
                            color: color,
                            norm: t_norms[0],
                        });
                        out_verts.push(VertexIn {
                            pos: t_verts[1],
                            color: color,
                            norm: t_norms[1],
                        });
                        out_verts.push(VertexIn {
                            pos: t_verts[2],
                            color: color,
                            norm: t_norms[2],
                        });
                    } else if tokens.len() > 4 {
                        //Triangle 1: Vertices 1, 2, 3
                        //Triangle 2: Vertices 1, 3, 4
                        //println!("quad");
                        let vert_count = out_verts.len() as u32;
                        out_indeces.append(&mut vec![0,1,2,0,2,3].iter().map(|i| vert_count + i).collect());

                        for i in 1..5 {
                            let indexes:Vec<&str> = tokens[i].split("/").collect();
                            let v_index:usize = indexes[0].parse().unwrap();
                            let n_index:usize = indexes[2].parse().unwrap();
                            let v = verts[v_index -1];
                            let n = normals[n_index -1];
                            let vertex = VertexIn {
                                pos: v,
                                color: color,
                                norm: n,
                            };
                            out_verts.push(vertex);
                        }
                    }
                }
                _ => (),
            }
        }
        Mesh {
            verts: out_verts,
            indices: out_indeces,
        }
    }
}
fn calculate_normals(p1:Vec3,p2:Vec3,p3:Vec3) -> Vec3 {
    let edge12 = p2 - p1;
    let edge13 = p3 - p1;
    let norm = edge12.cross(edge13).norm();
    norm
}
