import logging
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum



class MeanFilterType:
    def MEAN_ONE(self) -> np.ndarray:
        return np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9.0
        
    def Gaussian(self) -> np.ndarray:
        return np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16.0
    
    def MEAN_5x5(self) -> np.ndarray:
        kernel = np.ones((5, 5), dtype=np.float32)
        return kernel / kernel.sum()
        
    def Gaussian_5x5(self) -> np.ndarray:
        kernel = np.array([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1]], dtype=np.float32)
        return kernel / kernel.sum()
    
    def Motion_Blur(self) -> np.ndarray:
        kernel = np.zeros((5, 5), dtype=np.float32)
        kernel[2, :] = 1.0 / 5.0
        return kernel
        
    def DirectionalBlur_45deg(self) -> np.ndarray:
        kernel = np.eye(5, dtype=np.float32)
        return kernel / kernel.sum()

class EdgeMode(Enum):
    ZERO = "constant"  # 边界外像素值为0
    REFLECT = "reflect"  # 边界外像素值为镜像
    REPLICATE = "replicate"  # 边界外像素值为边界像素值


class EnhanceUtils:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        pass

    # 均值滤波
    def MeanFilter(
        self,
        img: np.ndarray,
        kernel_size: int = 3,
        filter_type: np.ndarray = np.array(
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32
        )
        / 9.0,
        edge_mode: EdgeMode = EdgeMode.ZERO,
    ) -> np.ndarray:
        if kernel_size * kernel_size != filter_type.size:
            self.logger.error("均值滤波器大小不匹配")
            return img
        h, w = img.shape

        # 边缘处理
        pad = kernel_size // 2
        img_padded = np.pad(
            img,
            pad,
            mode=edge_mode.value,
        )

        rows = h * w
        cols = kernel_size * kernel_size
        # 计算卷积
        windows = np.lib.stride_tricks.sliding_window_view(
            img_padded, (kernel_size, kernel_size)
        )
        windows = windows.reshape(rows, cols)
        output = np.dot(windows, filter_type.flatten()).reshape(h, w)

        return output

    def MedianFilter(
        self,
        img: np.ndarray,
        kernel_size: int = 3,
        edge_mode: EdgeMode = EdgeMode.ZERO,
    ) -> np.ndarray:
        """
        中值滤波
        :param img: 输入图像
        :param kernel_size: 核大小，必须为奇数
        :param edge_mode: 边缘处理模式
        :return: 滤波后的图像
        """
        if kernel_size % 2 == 0:
            self.logger.error("中值滤波器大小必须为奇数")
            return img

        h, w = img.shape

        # 边缘处理
        pad = kernel_size // 2
        img_padded = np.pad(
            img,
            pad,
            mode=edge_mode.value,
        )

        # 使用sliding_window_view获取所有窗口
        windows = np.lib.stride_tricks.sliding_window_view(
            img_padded, (kernel_size, kernel_size)
        )

        # 重塑窗口以便于计算
        windows_reshaped = windows.reshape(h, w, kernel_size * kernel_size)

        # 对每个窗口内的像素排序并取中值
        output = np.median(windows_reshaped, axis=2).reshape(h, w)

        return output.astype(img.dtype)
