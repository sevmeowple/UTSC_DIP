use std::char;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;

#[derive(Default)]
pub struct BMPRes {
    // self
    bmp_buffer: Vec<u8>,
    // Size of File 2-4
    bf_size: u32,
    // Reserved1 6-2
    bf_reserved1: u16,
    // Reserved2 8-2
    bf_reserved2: u16,
    // Offset after file head 10-4
    bf_offbits: u32,
    // Size of bitmap info 14-4
    dib_size: u32,
    // Width of bitmap 18-4
    bi_width: u32,
    // Height of bitmap 22-4
    bi_height: u32,
    // Planes 26-2
    bi_planes: u16,
    // Bit count 28-2
    bi_bit_count: u16,
    // Compression 30-4
    bi_compression: u32,
    // Size of image 34-4
    bi_size_image: u32,
    // X pixels per meter 38-4
    bi_x_pels_per_meter: u32,
    // Y pixels per meter 42-4
    bi_y_pels_per_meter: u32,
    // Colors used 46-4
    bi_clr_used: u32,
    // Important colors 50-4
    bi_clr_important: u32,
    // Color table
    bmi_colors: Vec<u32>,
}

// basic impl基本信息读取的实现
impl BMPRes {
    pub fn get_bf_size(&mut self) -> u32 {
        // 通过buffer获取bf_size并更新
        self.bf_size = u32::from_le_bytes(self.bmp_buffer[2..6].try_into().unwrap());
        self.bf_size
    }
    pub fn get_bf_reserved1(&mut self) -> u16 {
        self.bf_reserved1 = u16::from_le_bytes(self.bmp_buffer[6..8].try_into().unwrap());
        self.bf_reserved1
    }
    pub fn get_bf_reserved2(&mut self) -> u16 {
        self.bf_reserved2 = u16::from_le_bytes(self.bmp_buffer[8..10].try_into().unwrap());
        self.bf_reserved2
    }

    pub fn get_bf_offbits(&mut self) -> u32 {
        self.bf_offbits = u32::from_le_bytes(self.bmp_buffer[10..14].try_into().unwrap());
        self.bf_offbits
    }

    pub fn get_dib_size(&mut self) -> u32 {
        self.dib_size = u32::from_le_bytes(self.bmp_buffer[14..18].try_into().unwrap());
        self.dib_size
    }

    pub fn get_bi_width(&mut self) -> u32 {
        self.bi_width = u32::from_le_bytes(self.bmp_buffer[18..22].try_into().unwrap());
        self.bi_width
    }

    pub fn get_bi_height(&mut self) -> u32 {
        self.bi_height = u32::from_le_bytes(self.bmp_buffer[22..26].try_into().unwrap());
        self.bi_height
    }

    pub fn get_bi_planes(&mut self) -> u16 {
        self.bi_planes = u16::from_le_bytes(self.bmp_buffer[26..28].try_into().unwrap());
        self.bi_planes
    }

    pub fn get_bi_bit_count(&mut self) -> u16 {
        self.bi_bit_count = u16::from_le_bytes(self.bmp_buffer[28..30].try_into().unwrap());
        self.bi_bit_count
    }

    pub fn get_bi_compression(&mut self) -> u32 {
        self.bi_compression = u32::from_le_bytes(self.bmp_buffer[30..34].try_into().unwrap());
        self.bi_compression
    }

    pub fn get_bi_size_image(&mut self) -> u32 {
        self.bi_size_image = u32::from_le_bytes(self.bmp_buffer[34..38].try_into().unwrap());
        self.bi_size_image
    }

    pub fn get_bi_x_pels_per_meter(&mut self) -> u32 {
        self.bi_x_pels_per_meter = u32::from_le_bytes(self.bmp_buffer[38..42].try_into().unwrap());
        self.bi_x_pels_per_meter
    }

    pub fn get_bi_y_pels_per_meter(&mut self) -> u32 {
        self.bi_y_pels_per_meter = u32::from_le_bytes(self.bmp_buffer[42..46].try_into().unwrap());
        self.bi_y_pels_per_meter
    }

    pub fn get_bi_clr_used(&mut self) -> u32 {
        self.bi_clr_used = u32::from_le_bytes(self.bmp_buffer[46..50].try_into().unwrap());
        self.bi_clr_used
    }

    pub fn get_bi_clr_important(&mut self) -> u32 {
        self.bi_clr_important = u32::from_le_bytes(self.bmp_buffer[50..54].try_into().unwrap());
        self.bi_clr_important
    }

    pub fn get_bmi_colors(&mut self) -> Vec<u32> {
        let mut color_table: Vec<u32> = Vec::new();
        let mut color_table_start = 54;
        let color_table_end = self.bf_offbits as usize;
        while color_table_start < color_table_end {
            color_table.push(u32::from_le_bytes(
                self.bmp_buffer[color_table_start..color_table_start + 4]
                    .try_into()
                    .unwrap(),
            ));
            color_table_start += 4;
        }
        self.bmi_colors = color_table;
        self.bmi_colors.clone()
    }
}

// advanced impl
impl BMPRes {
    pub fn print_self(&self) {
        println!("bf_size: {}", self.bf_size);
        println!("bf_reserved1: {}", self.bf_reserved1);
        println!("bf_reserved2: {}", self.bf_reserved2);
        println!("bf_offbits: {}", self.bf_offbits);
        println!("dib_size: {}", self.dib_size);
        println!("bi_width: {}", self.bi_width);
        println!("bi_height: {}", self.bi_height);
        println!("bi_planes: {}", self.bi_planes);
        println!("bi_bit_count: {}", self.bi_bit_count);
        println!("bi_compression: {}", self.bi_compression);
        println!("bi_size_image: {}", self.bi_size_image);
        println!("bi_x_pels_per_meter: {}", self.bi_x_pels_per_meter);
        println!("bi_y_pels_per_meter: {}", self.bi_y_pels_per_meter);
        println!("bi_clr_used: {}", self.bi_clr_used);
        println!("bi_clr_important: {}", self.bi_clr_important);
    }

    pub fn print_colors_table(&self) {
        for i in 0..self.bmi_colors.len() {
            println!("Color {}: 0x{:08X}", i, self.bmi_colors[i]);
        }
    }

    pub fn get_certain_pixel(&self, x: u32, y: u32) -> Result<u32, &str> {
        let mut pixel = 0;
        let width = self.bi_width;
        let height = self.bi_height;
        let bit_count = self.bi_bit_count;
        let offbits = self.bf_offbits;

        if x >= self.bi_width || y >= self.bi_height {
            return Err("坐标超出图像边界");
        }
        // 计算每个像素占用的字节数
        let bytes_per_pixel = bit_count as usize / 8;

        // 计算每行的字节数（包括填充）
        let row_bytes_without_padding = width as usize * bytes_per_pixel;
        let padding = (4 - (row_bytes_without_padding % 4)) % 4;
        let bytes_per_row = row_bytes_without_padding + padding;

        // 计算像素在缓冲区中的位置
        // BMP图像存储是从底部向上的，所以y坐标需要反转
        let this_pixel_start =
            offbits as usize + (height - y) as usize * bytes_per_row + x as usize * bytes_per_pixel;

        match bit_count {
            8 => {
                let color_index = self.bmp_buffer[this_pixel_start] as usize;
                // 检查索引是否在颜色表范围内
                if color_index < self.bmi_colors.len() {
                    pixel = self.bmi_colors[color_index];
                } else {
                    return Err("颜色索引超出颜色表范围");
                }
            }
            24 => {
                pixel = u32::from_le_bytes(
                    self.bmp_buffer[this_pixel_start..this_pixel_start + 3]
                        .try_into()
                        .unwrap(),
                );
            }
            _ => {
                println!("Unsupported bit count");
            }
        }

        if padding > 0 {
            println!("Warning: Padding detected When reading pixel({},{})", x, y);
        }

        Ok(pixel)
    }

    // 调色,将前n行颜色替换为指定颜色
    pub fn colorize(&mut self, color: u32, n: u32) -> Result<(), &str> {
        let width = self.bi_width;
        let height = self.bi_height;
        let bit_count = self.bi_bit_count;
        let offbits = self.bf_offbits;

        if n > height {
            return Err("n超出图像高度");
        }

        // 计算每个像素占用的字节数
        let bytes_per_pixel = bit_count as usize / 8;

        // 计算每行的字节数（包括填充）
        let row_bytes_without_padding = width as usize * bytes_per_pixel;
        let padding = (4 - (row_bytes_without_padding % 4)) % 4;
        let bytes_per_row = row_bytes_without_padding + padding;

        // 计算像素在缓冲区中的位置
        // BMP图像存储是从底部向上的，所以y坐标需要反转
        let mut this_pixel_start = offbits as usize + (height - n) as usize * bytes_per_row;

        for _ in 0..n {
            for _ in 0..width {
                match bit_count {
                    8 => {
                        self.bmp_buffer[this_pixel_start] = (color & 0xFF) as u8;
                    }
                    24 => {
                        self.bmp_buffer[this_pixel_start] = (color & 0xFF) as u8;
                        self.bmp_buffer[this_pixel_start + 1] = ((color >> 8) & 0xFF) as u8;
                        self.bmp_buffer[this_pixel_start + 2] = ((color >> 16) & 0xFF) as u8;
                    }
                    _ => {
                        println!("Unsupported bit count");
                    }
                }
                this_pixel_start += bytes_per_pixel;
            }
            this_pixel_start += padding;
        }
        Ok(())
    }

    // 保存到文件
    pub fn save_to_file(&self, path: &str) -> Result<(), io::Error> {
        let mut file = File::create(path)?;
        file.write_all(&self.bmp_buffer)?;
        Ok(())
    }
    // change color table
    // 方案,不动蓝色通道,红色通道和绿色通道分别均匀取16个值
    // 最后形成16*16种颜色
    pub fn change_color_table(&mut self) {
        let color_table_start = 54;
        let color_table_end = self.bf_offbits as usize;
        let mut color_table_bytes = Vec::new();

        for r in 0..16 {
            for g in 0..16 {
                // 按 BGRA 顺序构造颜色（这是 BMP 文件中的标准格式）
                let b = 0u8; // 蓝色 = 0
                let g = (g * 16) as u8; // 绿色 = 0,16,32,...,240
                let r = (r * 16) as u8; // 红色 = 0,16,32,...,240
                let a = 0u8; // Alpha = 0 (0xFF 为完全不透明)

                // 添加字节到颜色表，按 BGRA 顺序
                color_table_bytes.push(b);
                color_table_bytes.push(g);
                color_table_bytes.push(r);
                color_table_bytes.push(a);
            }
        }

        // 更新缓冲区中的颜色表
        self.bmp_buffer
            .splice(color_table_start..color_table_end, color_table_bytes);

        // 更新内部结构的颜色表
        self.get_bmi_colors();
    }

    // 压缩成n*n的生成字符图像并保存到txt文件
    // " " (空格) → "." → ":" → "-" → "=" → "+" → "*" → "#" → "%" → "@"
    pub fn generate_char_image(&self, path: &str, n: u32) -> Result<(), io::Error> {
        let char_map = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '@'];
        let mut char_image = String::new();

        let width = self.bi_width;
        let height = self.bi_height;
        let bit_count = self.bi_bit_count;
        let offbits = self.bf_offbits;

        // 确保n不为0，避免除零错误
        let zoom = width.max(1) / n.max(1);
        if zoom == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Scale factor too large",
            ));
        }

        // 计算每个像素占用的字节数
        let bytes_per_pixel = bit_count as usize / 8;

        // 计算每行的字节数（包括填充）
        let row_bytes_without_padding = width as usize * bytes_per_pixel;
        let padding = (4 - (row_bytes_without_padding % 4)) % 4;
        let bytes_per_row = row_bytes_without_padding + padding;

        // 处理每个缩放后的行
        for y in 0..n {
            // 计算在原图中对应的行范围
            let y_start = (height * y) / n;
            let y_end = (height * (y + 1)) / n;

            // 处理每个缩放后的列
            for x in 0..n {
                let x_start = (width * x) / n;
                let x_end = (width * (x + 1)) / n;

                let mut sum = 0;
                let mut count = 0;

                // 计算区域内像素的平均值
                for orig_y in y_start..y_end {
                    for orig_x in x_start..x_end {
                        // BMP存储是从底向上的，需要反转y坐标
                        let y_pos = height - 1 - orig_y;
                        let pixel_offset = offbits as usize
                            + (y_pos as usize * bytes_per_row)
                            + (orig_x as usize * bytes_per_pixel);

                        // 安全检查，确保不会越界
                        if pixel_offset + bytes_per_pixel <= self.bmp_buffer.len() {
                            match bit_count {
                                8 => {
                                    let color_index = self.bmp_buffer[pixel_offset] as usize;
                                    if color_index < self.bmi_colors.len() {
                                        // 从颜色表获取颜色并计算灰度
                                        let color = self.bmi_colors[color_index];
                                        let r = ((color >> 16) & 0xFF) as u32;
                                        let g = ((color >> 8) & 0xFF) as u32;
                                        let b = (color & 0xFF) as u32;
                                        sum += (r * 299 + g * 587 + b * 114) / 1000;
                                        count += 1;
                                    }
                                }
                                24 => {
                                    let b = self.bmp_buffer[pixel_offset] as u32;
                                    let g = self.bmp_buffer[pixel_offset + 1] as u32;
                                    let r = self.bmp_buffer[pixel_offset + 2] as u32;
                                    sum += (r * 299 + g * 587 + b * 114) / 1000;
                                    count += 1;
                                }
                                _ => {}
                            }
                        }
                    }
                }

                // 计算并添加对应的字符
                if count > 0 {
                    let avg = sum / count;
                    let char_index = (avg * char_map.len() as u32 / 256) as usize;
                    char_image.push(char_map[char_index.min(char_map.len() - 1)]);
                } else {
                    char_image.push(' ');
                }
            }
            char_image.push('\n');
        }

        let mut file = File::create(path)?;
        file.write_all(char_image.as_bytes())?;
        Ok(())
    }
}

pub fn bmp_reader(path: &str) -> Result<BMPRes, io::Error> {
    let mut bmp_res = BMPRes::default();
    // check if file exists
    if !Path::new(path).exists() {
        return Err(io::Error::new(io::ErrorKind::NotFound, "File not found"));
    }

    let mut file = File::open(path)?;
    let mut bmp_buffer = Vec::new();
    let byte_read_n = file.read_to_end(&mut bmp_buffer)?;
    if byte_read_n < 18 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BMP file"));
    }

    // check if 0-2 is BM
    if bmp_buffer[0] != 0x42 || bmp_buffer[1] != 0x4D {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Not a BMP file"));
    }

    // Produce BMPRes
    {
        // get buff
        let bmpbuffer = bmp_buffer.clone();

        bmp_res.bmp_buffer = bmpbuffer;
        // get file size
        bmp_res.bf_size = u32::from_le_bytes(bmp_buffer[2..6].try_into().unwrap());
        // get reserved1
        bmp_res.bf_reserved1 = u16::from_le_bytes(bmp_buffer[6..8].try_into().unwrap());
        // get reserved2
        bmp_res.bf_reserved2 = u16::from_le_bytes(bmp_buffer[8..10].try_into().unwrap());
        // get offset after file head
        bmp_res.bf_offbits = u32::from_le_bytes(bmp_buffer[10..14].try_into().unwrap());
        // get size of bitmap info
        bmp_res.dib_size = u32::from_le_bytes(bmp_buffer[14..18].try_into().unwrap());
        // get width of bitmap
        bmp_res.bi_width = u32::from_le_bytes(bmp_buffer[18..22].try_into().unwrap());
        // get height of bitmap
        bmp_res.bi_height = u32::from_le_bytes(bmp_buffer[22..26].try_into().unwrap());
        // get planes
        bmp_res.bi_planes = u16::from_le_bytes(bmp_buffer[26..28].try_into().unwrap());
        // get bit count
        bmp_res.bi_bit_count = u16::from_le_bytes(bmp_buffer[28..30].try_into().unwrap());
        // get compression
        bmp_res.bi_compression = u32::from_le_bytes(bmp_buffer[30..34].try_into().unwrap());
        // get size of image
        bmp_res.bi_size_image = u32::from_le_bytes(bmp_buffer[34..38].try_into().unwrap());
        // get x pixels per meter
        bmp_res.bi_x_pels_per_meter = u32::from_le_bytes(bmp_buffer[38..42].try_into().unwrap());
        // get y pixels per meter
        bmp_res.bi_y_pels_per_meter = u32::from_le_bytes(bmp_buffer[42..46].try_into().unwrap());
        // get colors used
        bmp_res.bi_clr_used = u32::from_le_bytes(bmp_buffer[46..50].try_into().unwrap());
        // get important colors
        bmp_res.bi_clr_important = u32::from_le_bytes(bmp_buffer[50..54].try_into().unwrap());
        // get color table
        let mut color_table: Vec<u32> = Vec::new();
        let mut color_table_start = 54;
        let color_table_end = bmp_res.bf_offbits as usize;
        while color_table_start < color_table_end {
            color_table.push(u32::from_le_bytes(
                bmp_buffer[color_table_start..color_table_start + 4]
                    .try_into()
                    .unwrap(),
            ));
            color_table_start += 4;
        }
        bmp_res.bmi_colors = color_table;
    }

    Ok(bmp_res)
}
