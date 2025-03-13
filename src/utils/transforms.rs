/*
 * This file is part of the "ImageProducer" project.
 *
 * transforms.rs 主要是进行变换操作的函数
*/
use ndarray::{Array1, Array2, Axis};
use num_complex::Complex64;
use std::cell::RefCell;
use std::cmp::min;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::path;

#[derive(Eq, PartialEq, Hash, Clone, Copy)]
pub enum TransformType {
    DFT,
    DCT,
    Hadamard,
    Haar,
}
/// 傅立叶变换矩阵
pub struct Transform {
    pub n: usize,
    pub transform_type: TransformType,
    pub matrix: Array2<Complex64>,
}

/// 缓存所有变换矩阵
/// 缓存所有变换矩阵
pub struct TransUtil {
    // 使用键值对存储变换矩阵
    // 键是 (变换类型, 矩阵大小) 的元组
    // 值是对应的变换矩阵
    pub transforms: HashMap<(TransformType, usize), Transform>,
}

impl TransUtil {
    pub fn new() -> Self {
        Self {
            transforms: HashMap::new(),
        }
    }
    /// 获取指定类型和大小的变换矩阵
    /// 如果存在则直接返回，不存在则创建、缓存并返回
    pub fn get_transform(&mut self, transform_type: TransformType, size: usize) -> &Transform {
        let key = (transform_type, size);

        // 如果不存在则创建新的变换矩阵
        if !self.transforms.contains_key(&key) {
            let transform = Transform::new(size, transform_type);
            self.transforms.insert(key, transform);
        }

        // 返回引用
        self.transforms.get(&key).unwrap()
    }

    /// 一维变换, 传入信号和变换类型
    /// 通过信号的长度和变换类型获取对应的变换矩阵
    pub fn trans_1d(
        &mut self,
        signal: &Array1<Complex64>,
        transform_type: TransformType,
    ) -> Array1<Complex64> {
        let size = signal.len();
        let temp_transform = self.get_transform(transform_type, size);
        let result = signal.dot(&temp_transform.matrix);
        result
    }

    /// 二维变换, 传入信号和变换类型
    /// 通过信号的长度和变换类型获取对应的变换矩阵
    pub fn trans_2d(
        &mut self,
        signal: &Array2<Complex64>,
        transform_type: TransformType,
    ) -> Array2<Complex64> {
        let size_n = signal.len_of(Axis(0));
        let size_m = signal.len_of(Axis(1));

        // 使用代码块限制 self 的可变借用范围
        let matrix_n = { self.get_transform(transform_type, size_n).matrix.clone() };

        let matrix_m = { self.get_transform(transform_type, size_m).matrix.clone() };

        // 直接进行矩阵运算并返回结果
        matrix_n.dot(signal).dot(&matrix_m.t())
    }
}

impl Transform {
    /// 创建一维变换矩阵
    pub fn new(n: usize, _type: TransformType) -> Self {
        match _type {
            TransformType::DFT => {
                let mut matrix = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let angle = -2.0 * PI * (i as f64) * (j as f64) / (n as f64);
                        matrix[[i, j]] = Complex64::new(angle.cos(), angle.sin());
                    }
                }
                Self {
                    n,
                    transform_type: _type,
                    matrix,
                }
            }
            TransformType::DCT => {
                let mut matrix = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        let angle = PI * (j as f64) * (i as f64) / (n as f64);
                        matrix[[i, j]] = Complex64::new(angle.cos(), 0.0);
                    }
                }
                Self {
                    n,
                    transform_type: _type,
                    matrix,
                }
            }
            TransformType::Hadamard => {
                let mut matrix = Array2::zeros((n, n));
                for i in 0..n {
                    for j in 0..n {
                        matrix[[i, j]] = Complex64::new(
                            ((i & j).count_ones() as f64 / 2.0).cos(),
                            ((i & j).count_ones() as f64 / 2.0).sin(),
                        );
                    }
                }
                Self {
                    n,
                    transform_type: _type,
                    matrix,
                }
            }
            TransformType::Haar => {
                let mut matrix = Array2::zeros((n, n));
                let mut temp = Array1::zeros(n);
                temp[0] = 1.0;
                for i in 0..n {
                    for j in 0..n {
                        let mut k = 0;
                        let mut t = i;
                        while t > 0 {
                            k += t & 1;
                            t >>= 1;
                        }
                        matrix[[i, j]] = Complex64::new(temp[k], 0.0);
                    }
                }
                Self {
                    n,
                    transform_type: _type,
                    matrix,
                }
            }
        }
    }
}

/// 定义一个 trait 用于转换为复数数组
pub trait ToComplex {
    fn to_complex(&self) -> Array2<Complex64>;
}

/// 为 Array2<f64> 实现 ToComplex trait
impl ToComplex for Array2<f64> {
    fn to_complex(&self) -> Array2<Complex64> {
        self.mapv(|x| Complex64::new(x, 0.0))
    }
}

/// 为 Vec<Vec<T>> 实现 ToComplex trait
impl<T> ToComplex for Vec<Vec<T>>
where
    T: Into<f64> + Copy,
{
    fn to_complex(&self) -> Array2<Complex64> {
        let rows = self.len();
        let cols = if rows > 0 { self[0].len() } else { 0 };

        let mut result = Array2::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..min(cols, self[i].len()) {
                let val: f64 = self[i][j].into();
                result[[i, j]] = Complex64::new(val, 0.0);
            }
        }

        result
    }
}

/// 通用转换函数
pub fn to_complex<T: ToComplex>(data: &T) -> Array2<Complex64> {
    data.to_complex()
}

pub fn better_print(arr: &Array2<Complex64>) {
    let dim = arr.dim();
    for i in 0..arr.len_of(Axis(0)) {
        for j in 0..arr.len_of(Axis(1)) {
            print!("{:.2}+{:.2}i ", arr[[i, j]].re, arr[[i, j]].im);
        }
        println!();
    }
    println!(
        "shape: ({}, {}), size: {}",
        dim.0,
        dim.1,
        arr.len_of(Axis(0)) * arr.len_of(Axis(1))
    );
    println!("\n");
}

//绘制频谱图像,二维,输入为复数数组
// 正常计算下频谱图像无法被人眼识别
// 为了方便观察，将频谱图像的值取对数
pub fn plot_spectrum(arr: &Array2<Complex64>, path: &str) {
    let dim = arr.dim();
    let mut result = Array2::zeros((dim.0, dim.1));
    for i in 0..dim.0 {
        for j in 0..dim.1 {
            result[[i, j]] = arr[[i, j]].norm().log(10.0);
        }
    }
    let max = result.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min = result.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mut img = image::GrayImage::new(dim.1 as u32, dim.0 as u32);
    for i in 0..dim.0 {
        for j in 0..dim.1 {
            let val = (result[[i, j]] - min) / (max - min) * 255.0;
            img.put_pixel(j as u32, i as u32, image::Luma([val as u8]));
        }
    }
    img.save(path).unwrap();
}
