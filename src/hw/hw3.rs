use ndarray::{array, Array2};
use num_complex::Complex64;

use crate::utils::binreader;
use crate::utils::repath;
use crate::utils::transforms as tf;
pub fn hw3_1() {
    let real_input = array![
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0]
    ];
    let input = tf::to_complex(&real_input);
    let mut trans_util = tf::TransUtil::new();
    let dft_result = trans_util.trans_2d(&input, tf::TransformType::DFT);
    let dct_res = trans_util.trans_2d(&input, tf::TransformType::DCT);
    let hadamard_res = trans_util.trans_2d(&input, tf::TransformType::Hadamard);
    let haar_res = trans_util.trans_2d(&input, tf::TransformType::Haar);
    tf::better_print(&dft_result);
    tf::better_print(&dct_res);
    tf::better_print(&hadamard_res);
    tf::better_print(&haar_res);
}

pub fn hw3_3() {
    let lena_bmp = binreader::bmp_reader("./pics/Lena.bmp").unwrap();
    let mx = lena_bmp.get_all_pixels();
    let mut trans_util = tf::TransUtil::new();
    let mx_trans = tf::to_complex(&mx);
    let dft_res = trans_util.trans_2d(&mx_trans, tf::TransformType::DFT);
    let dct_res = trans_util.trans_2d(&mx_trans, tf::TransformType::DCT);
    tf::plot_spectrum(&dft_res, "./pics/hw3/dft_spectrum.png");
    tf::plot_spectrum(&dct_res, "./pics/hw3/dct_spectrum.png");
}
