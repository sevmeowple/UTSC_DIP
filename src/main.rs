use hw::hw2;
// use image::codecs::bmp;

mod utils;
mod hw;


fn main() {
    // println!("Hello, world!");
    let path = "./Lena.bmp";
    let mut bmp_res = utils::binreader::bmp_reader(path).unwrap();
    // bmp_res.print_self();
    // bmp_res.print_colors_table();
    // let pixel_read = hw2::hw2_1(&mut bmp_res).unwrap();
    // for i in pixel_read {
    //     println!("{:08X}", i);
    // }

    // bmp_res.colorize(255,255).unwrap();
    // bmp_res.save_to_file("./hw2_2.bmp").unwrap();

    bmp_res.change_color_table();
    bmp_res.save_to_file("./hw2_3_2.bmp").unwrap();
}
