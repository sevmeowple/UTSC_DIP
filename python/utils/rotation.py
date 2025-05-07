import numpy as np


def Wx(x: np.float32, a: np.float32) -> np.float32:
    """
    Wx函数
    :param x: 输入值
    :return: Wx值
    """
    absx = abs(x)
    if absx <= 1:
        return (a + 2) * absx ** 3 - (a + 3) * absx ** 2 + 1
    elif absx < 2:
        return a * absx ** 3 - 5 * a * absx ** 2 + 8 * a * absx - 4 * a
    else:
        return 0


def table_init(num: int, a: np.float32) -> np.ndarray:
    """
    初始化查找表
    映射 对应 table[n] 为 Wx(n/num,a)的值

    :param num: 表的大小
    :return: 一维查找表
    """
    table = np.zeros(num, dtype=np.float32)
    for i in range(num):
        table[i] = Wx(i / num, a)
    return table


class Rotation:
    table: np.ndarray = None
    num: int = 256

    def __init__(self, a: np.float32 = -0.5, num: int = 256) -> None:
        """
        初始化Rotation类
        :param a: Wx函数的参数,默认为-0.5Catmull-Rom,
        也推荐使用-0.75 B-Spline
        :param num: 查找表的大小,默认为256
        """
        self.table = table_init(num, a)  # 初始化查找表
        self.num = num  # 设置查找表的大小

    def get_neighborhood_vectorized(self, image, coords_y, coords_x):
        """向量化获取邻域矩阵"""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        num_coords = len(coords_y)
        
        # 创建结果数组
        result = np.zeros((num_coords, 4, 4, channels), dtype=np.float32)
        
        # 对每个偏移量单独处理
        for i, y_offset in enumerate([-1, 0, 1, 2]):
            for j, x_offset in enumerate([-1, 0, 1, 2]):
                # 计算偏移后的坐标
                y_pos = coords_y + y_offset
                x_pos = coords_x + x_offset
                
                # 镜像边界处理
                y_pos = np.abs(y_pos)
                x_pos = np.abs(x_pos)
                
                mask_y = y_pos >= h
                y_pos[mask_y] = 2 * h - y_pos[mask_y] - 2
                
                mask_x = x_pos >= w
                x_pos[mask_x] = 2 * w - x_pos[mask_x] - 2
                
                # 确保在有效范围内
                y_pos = np.clip(y_pos, 0, h - 1)
                x_pos = np.clip(x_pos, 0, w - 1)
                
                # 获取像素值并存入结果
                if channels > 1:
                    result[:, i, j] = image[y_pos, x_pos]
                else:
                    result[:, i, j, 0] = image[y_pos, x_pos]
        
        return result


    def get_weight_vectorized(self, frac: np.ndarray):
        """计算双三次插值权重向量

        Args:
            frac: 小数部分坐标，形状为(new_h*new_w,)

        Returns:
            形状为(new_h*new_w, 4)的权重向量
        """
        # 将小数部分扩展为(n, 1)
        frac = frac.reshape(-1, 1)
        indices = np.array([-1, 0, 1, 2])
        distances = np.abs(frac + indices)  # 形状为(n, 4)

        # 将距离映射到查找表索引
        table_indices = (distances * (self.num - 1)).astype(np.int32)
        table_indices = np.clip(table_indices, 0, self.num - 1)

        # 从查找表获取权重
        weights = self.table[table_indices]  # 形状为(n, 4)

        return weights

    def reverse_map(self, angle: float, image: np.ndarray) -> np.ndarray:
        # 逆向映射
        h, w = image.shape[:2]
        # 确保输入图像处理正确的通道数
        if len(image.shape) == 2:
            # 如果是灰度图，扩展为3通道
            image_with_channels = np.zeros((h, w, 3), dtype=np.uint8)
            image_with_channels[:,:,0] = image
            image_with_channels[:,:,1] = image
            image_with_channels[:,:,2] = image
            image = image_with_channels
            channels = 1  # 记录原始通道数
        else:
            channels = image.shape[2]
        
        # 计算旋转后的图像的宽高
        new_w = int(abs(h * np.sin(angle)) + abs(w * np.cos(angle)))
        new_h = int(abs(h * np.cos(angle)) + abs(w * np.sin(angle)))
        
        # 创建结果数组，包含alpha通道
        result = np.zeros((new_h, new_w, 4), dtype=np.float32)
        
        # 计算旋转中心
        center_x = w // 2
        center_y = h // 2
        new_center_x = new_w // 2
        new_center_y = new_h // 2
        
        # 计算旋转矩阵
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        
        # 生成索引矩阵
        new_table = np.indices((new_h, new_w), dtype=np.float32)
        new_table[0] -= new_center_y
        new_table[1] -= new_center_x
        new_table = np.reshape(new_table, (2, -1))
        new_table = rotation_matrix @ new_table
        new_table = np.reshape(new_table, (2, new_h, new_w))
        new_table[0] += center_y
        new_table[1] += center_x
        
        # 分离整数和小数部分
        new_table_int = np.floor(new_table).astype(np.int32)
        new_table_frac = new_table - new_table_int
        
        # 创建掩码标识哪些像素在原图像边界内
        # 这里可以根据需要调整边界判断条件
        valid_mask = ((new_table[0] >= 0) & (new_table[0] < h) & 
                    (new_table[1] >= 0) & (new_table[1] < w))
        
        # 仅处理有效区域内的像素
        valid_coords = np.where(valid_mask)
        coords_y = new_table_int[0][valid_coords]
        coords_x = new_table_int[1][valid_coords]
        frac_y = new_table_frac[0][valid_coords]
        frac_x = new_table_frac[1][valid_coords]
        
        # 获取邻域和权重
        pixels = self.get_neighborhood_vectorized(image, coords_y, coords_x)
        weights_y = self.get_weight_vectorized(frac_y)
        weights_x = self.get_weight_vectorized(frac_x)
        
        # 重塑权重以便计算
        weights_y_reshaped = weights_y.reshape(-1, 1, 4)  # (n, 1, 4)
        weights_x_reshaped = weights_x.reshape(-1, 4, 1)  # (n, 4, 1)
        
        # 计算插值结果
        input_channels = image.shape[2] if len(image.shape) > 2 else 1
        temp_result = np.zeros((len(coords_y), input_channels), dtype=np.float32)
        
        # 对每个通道单独计算
        for c in range(input_channels):
            for i in range(4):
                for j in range(4):
                    # 计算每个像素的加权贡献
                    temp_result[:, c] += pixels[:, i, j, c] * weights_y_reshaped[:, 0, i] * weights_x_reshaped[:, j, 0]
        
        # 组装最终结果数组，包括alpha通道
        temp_full = np.zeros((len(coords_y), 4), dtype=np.float32)
        temp_full[:, :input_channels] = temp_result
        
        # 设置有效区域的alpha通道为255(完全不透明)
        temp_full[:, 3] = 255
        
        # 将计算结果填回原始形状的数组，无效区域保持透明(alpha=0)
        result[valid_coords[0], valid_coords[1]] = temp_full
        
        # 根据原始图像类型返回结果
        if channels == 1:
            # 如果是灰度图，只返回第一个通道和alpha通道
            return_result = np.zeros((new_h, new_w, 2), dtype=np.uint8)
            return_result[:, :, 0] = result[:, :, 0]
            return_result[:, :, 1] = result[:, :, 3]  # alpha通道
            return return_result
        else:
            return result.astype(np.uint8)

   