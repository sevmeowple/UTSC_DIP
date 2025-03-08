use crate::utils::binreader;
use std::io::Result;
use image::{self, GenericImageView};

// Step 1,read from (200,200) to (210,210)
pub fn hw2_1(bmp: &mut binreader::BMPRes) -> Result<Vec<u32>> {
    let mut res = Vec::new();
    for y in 200..210 {
        for x in 200..210 {
            res.push(bmp.get_certain_pixel(x, y).unwrap());
        }
    }
    Ok(res)
}
//用image库实现来检查是否hw2_1函数正确
pub fn hw2_1_check(path:&str) -> Result<Vec<u32>> {
    let bmp = image::open(path).unwrap();
    let mut res = Vec::new();
    for y in 200..210 {
        for x in 200..210 {
            let pixel = bmp.get_pixel(x, y);
            let mut pixel_u32 = 0;
            pixel_u32 |= (pixel[0] as u32) << 16;
            pixel_u32 |= (pixel[1] as u32) << 8;
            pixel_u32 |= pixel[2] as u32;
            res.push(pixel_u32);
        }
    }
    Ok(res)
}


pub fn hw2_1_print(res:Vec<u32>) -> Result<()> {
    // 每行输出10个像素
    for i in 0..res.len() {
        print!("\"{:08X}\", ", res[i]);
        if (i+1) % 10 == 0 {
            println!();
        }
    }
    Ok(())
}

// Step 2, colorize the first 256 rows with 255
// 具体实现请参考utils/binreader.rs中的colorize函数

// Step 3, change the color table
// 具体实现请参考utils/binreader.rs中的change_color_table函数