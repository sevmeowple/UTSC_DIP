#![allow(unused_imports)]
#![allow(dead_code)]
// use hw::hw2;
// use num_complex::Complex;
use ndarray::{array, Array2};
use num_complex::Complex64;
use utils::{binreader, transforms::{self as tf}};
// use image::codecs::bmp;

mod hw;
mod utils;

fn main() {
    hw::hw3::hw3_1();
    // hw::hw3::hw3_3();
    // let path = "./pics/exp/image.png";
    // let img = image::open(path).unwrap();
    // // 转换为灰度图像
    // let gray_img = img.to_luma8();

    // // 获取尺寸
    // let (width, height) = gray_img.dimensions();

    // // 创建二维矩阵来存储信号值
    // let mut signal_matrix: Vec<Vec<f64>> = Vec::with_capacity(height as usize);

    // // 填充矩阵
    // for y in 0..height {
    //     let mut row: Vec<f64> = Vec::with_capacity(width as usize);
    //     for x in 0..width {
    //         let pixel = gray_img.get_pixel(x, y);
    //         // 转换为标准化的亮度值 (0.0 到 1.0)
    //         let intensity = pixel[0] as f64 / 255.0;
    //         row.push(intensity);
    //     }
    //     signal_matrix.push(row);
    // }

    // println!("已创建 {}x{} 的二维信号矩阵", width, height);
    // let sig = binreader::bmp_reader("./pics/").unwrap();
    // let signal_matrix = sig.get_all_pixels();
    // let mut trans_util = tf::TransUtil::new();
    // let sig_complex = tf::to_complex(
    //     &signal_matrix
    // );
    // let dft_transform = trans_util.trans_2d(
    //     &sig_complex, tf::TransformType::DFT);
    // tf::better_print(&dft_transform);
    // tf::plot_spectrum(
    //     &dft_transform, 
    //     "./pics/exp/dft_spectrum.png"
    // );
    // let dct_transform = trans_util.trans_2d(
    //     &sig_complex, tf::TransformType::DCT);
    // tf::plot_spectrum(
    //     &dct_transform, 
    //     "./pics/exp/dct_spectrum.png"
    // )
    // let img = imgcodecs::imread("./pics/Lena.bmp", imgcodecs::IMREAD_COLOR)?;
    // highgui::imshow("Image", &img)?;
    // highgui::wait_key(0)?;
    // Ok(())

}
