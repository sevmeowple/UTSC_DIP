use hw::hw2;
// use image::codecs::bmp;

mod utils;
mod hw;


fn main() {
    // println!("Hello, world!");
    let path = "./pics/Lena.bmp";
    let mut bmp_res = utils::binreader::bmp_reader(path).unwrap();
    // bmp_res.print_self();
    // bmp_res.print_colors_table();
    let pixel_read = hw2::hw2_1(&mut bmp_res).unwrap();
    hw2::hw2_1_print(pixel_read).unwrap();
    println!("=====================");
    let pixel_read_check = hw2::hw2_1_check(path).unwrap();
    hw2::hw2_1_print(pixel_read_check).unwrap();

    // bmp_res.colorize(255,255).unwrap();
    // bmp_res.save_to_file("./hw2_2_fix1.bmp").unwrap();
    // let char_path = "./1.txt";
    // bmp_res.generate_char_image(char_path,64).unwrap();
    // bmp_res.change_color_table();
    // bmp_res.save_to_file("./hw2_3_2.bmp").unwrap();
}
