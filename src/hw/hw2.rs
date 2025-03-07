use crate::utils::binreader;
use std::io::Result;

// Step 1,read from (200,200) to (210,210)
pub fn hw2_1(bmp: &mut binreader::BMPRes) -> Result<Vec<u32>> {
    let mut res = Vec::new();
    for y in 200..211 {
        for x in 200..211 {
            res.push(bmp.get_certain_pixel(x, y).unwrap());
        }
    }

    Ok(res)
}

// Step 2, colorize the first 256 rows with 255
// 具体实现请参考utils/binreader.rs中的colorize函数

// Step 3, change the color table
// 具体实现请参考utils/binreader.rs中的change_color_table函数