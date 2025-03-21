from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from enum import Enum
from dataclasses import dataclass
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class TransType(Enum):
    DFT = 1
    DCT = 2
    WAVELET = 3
    HAAR = 4
    HADAMARD = 5


class ColorMap(Enum):
    """颜色图枚举类"""

    GRAY = "gray"  # 灰度
    VIRIDIS = "viridis"  # 彩虹色
    JET = "jet"  # 喷射色
    PLASMA = "plasma"  # 等离子
    INFERNO = "inferno"  # 地狱火
    MAGMA = "magma"  # 岩浆
    HOT = "hot"  # 热度图
    COOL = "cool"  # 冷色调
    RAINBOW = "rainbow"  # 彩虹
    HSV = "hsv"  # HSV色彩空间
    CIVIDIS = "cividis"  # 色盲友好
    TURBO = "turbo"  # 增强版彩虹
    BLUES = "Blues"  # 蓝色系
    REDS = "Reds"  # 红色系


@dataclass
class TransMatrix:
    size: int
    type: TransType
    matrix: np.ndarray


class TransformUtils:
    def __init__(self):
        self.Matrixs: list[TransMatrix] = []
        self.logger = logging.getLogger(__name__)
        pass

    def getMatrix(self, size: int, type: TransType):
        for matrix in self.Matrixs:
            if matrix.size == size and matrix.type == type:
                return matrix
        match type:
            case TransType.DFT:
                matrix = self.__createDFTMatrix(size)
                return matrix
            case TransType.HADAMARD:
                matrix = self.__createHadamardMatrix(size)
                return matrix
            case TransType.DCT:
                matrix = self.__createDCTMatrix(size)
                return matrix
            case TransType.HAAR:
                matrix = self.__createHaarMatrix(size)
                return matrix
            case _:
                self.logger.warning(f"Matrix not found for size {size} and type {type}")
                return None
        # self.logger.warning(f"Matrix not found for size {size} and type {type}")
        return None

    def __createDFTMatrix(self, size: int):
        """创建离散傅里叶变换矩阵"""
        # 创建一个空矩阵
        DFT = np.zeros((size, size), dtype=complex)

        # 为每个元素计算值
        for i in range(size):
            for j in range(size):
                DFT[i, j] = np.exp(-2j * np.pi * i * j / size)

        return DFT

    def __createDCTMatrix(self, size: int):
        """创建离散余弦变换矩阵"""
        # 创建一个空矩阵
        DCT = np.zeros((size, size))

        # 为每个元素计算值
        # 最后乘以系数
        # u从0到size-1
        for u in range(size):
            for x in range(size):
                DCT[u, x] = np.cos(u * (2 * x + 1) * np.pi / (2 * size))

            DCT[u, :] *= np.sqrt(1 / size) if u == 0 else np.sqrt(2 / size)
            
        return DCT

    def __createHaarMatrix(self, size: int):
        """创建哈尔变换矩阵"""
        # 检查size是否是2的幂
        if not (size > 0 and (size & (size - 1)) == 0):
            self.logger.error(f"Size {size} is not a power of 2")
            return None

        # 创建1x1的哈尔矩阵
        H = np.array([[1]])

        # 基本变换块
        h_transform = np.array([1, 1]) / np.sqrt(2)
        I_transform = np.array([1, -1]) / np.sqrt(2)
        # 递归构建更大的矩阵
        n = 1
        while n < size:
            # 构建新的Haar矩阵，大小为2n x 2n
            # 形式为：
            # [ H_n⊗[1,1] ]
            # [ I_n⊗[1,-1] ]
            # 这里⊗表示克罗内克积，但我们用复制和缩放实现

            H_ = np.kron(H, h_transform)
            I_ = np.kron(np.eye(n), I_transform)
            H = np.vstack((H_, I_))
            n *= 2
        return H

    def __createHadamardMatrix(self, size: int):
        """创建哈达玛德矩阵"""
        # 检查size是否是2的幂
        if not (size > 0 and (size & (size - 1)) == 0):
            self.logger.error(f"Size {size} is not a power of 2")
            return None

        # 创建1x1的哈达玛德矩阵
        H = np.array([[1]])

        # 递归构建更大的矩阵
        n = 1
        while n < size:
            H = np.block([[H, H], [H, -H]])
            n *= 2

        # 归一化（可选）
        # H = H / np.sqrt(size)

        return H

    def __generateTransformBasisImages(self, size: int, transform_type: TransType):
        """
        为各种变换类型生成基图像

        参数:
            size: 变换尺寸
            transform_type: 变换类型

        返回:
            基图像列表
        """
        # 检查size是否为完全平方数
        img_size = int(np.sqrt(size))
        if img_size * img_size != size:
            self.logger.error(f"Size {size} is not a perfect square")
            return None

        # 根据变换类型生成基矩阵
        basis_matrix = self.getMatrix(size, transform_type)
        if basis_matrix is None:
            self.logger.error(f"Unsupported transform type: {transform_type}")
            return None

        # 创建存储基图像的数组
        basis_images = []

        # 将每一行重新排列成二维图像
        for i in range(size):
            # 如果是DFT，获取实部和虚部
            if transform_type == TransType.DFT:
                real_part = np.real(basis_matrix[i]).reshape(img_size, img_size)
                imag_part = np.imag(basis_matrix[i]).reshape(img_size, img_size)
                # 可以选择显示复数的模或实部
                basis_image = np.sqrt(real_part**2 + imag_part**2)  # 模
                # 或者只显示实部
                # basis_image = real_part
            else:
                basis_image = basis_matrix[i].reshape(img_size, img_size)

            basis_images.append(basis_image)

        return basis_images

    def plotSpectrum(
        self,
        matrix: np.ndarray,
        FD: bool = True,
        magnitude_cmap: ColorMap = ColorMap.GRAY,
        XW: bool = True,
        phase_cmap: ColorMap = ColorMap.GRAY,
        transform_type: TransType = TransType.DFT,
    ):
        match transform_type:
            case TransType.DFT:
                if FD:  # 绘制幅度谱
                    # 计算幅度谱并应用对数变换
                    magnitude_spectrum = np.abs(matrix)
                    # 避免log(0)
                    magnitude_spectrum = np.maximum(magnitude_spectrum, 1e-10)
                    log_spectrum = np.log1p(magnitude_spectrum)

                    # 移动零频率到中心
                    shifted_spectrum = np.fft.fftshift(log_spectrum)

                    # 绘制图像而不是线图
                    plt.figure(figsize=(8, 6))
                    plt.imshow(shifted_spectrum, cmap=magnitude_cmap.value)
                    plt.colorbar(label="Log2 Magnitude")
                    plt.title("Magnitude Spectrum (Log Scale)")
                    plt.xlabel("Frequency")
                    plt.ylabel("Frequency")
                    plt.show()
                if XW:  # 绘制相位谱
                    # 计算相位谱
                    phase_spectrum = np.angle(matrix)

                    # 移动零频率到中心
                    shifted_phase = np.fft.fftshift(phase_spectrum)

                    # 绘制图像而不是线图
                    plt.figure(figsize=(8, 6))
                    plt.imshow(shifted_phase, cmap=phase_cmap.value)
                    plt.colorbar(label="Phase (radians)")
                    plt.title("Phase Spectrum")
                    plt.xlabel("Frequency")
                    plt.ylabel("Frequency")
                    plt.show()
            case TransType.DCT:
                spectrum = np.abs(matrix)
                log_spectrum = np.log1p(np.maximum(spectrum, 1e-10))
                
                plt.figure(figsize=(8, 6))
                plt.imshow(log_spectrum, cmap=magnitude_cmap.value)
                plt.colorbar(label="Log2 Coefficient Magnitude")
                plt.title("DCT Coefficient Spectrum")
                plt.xlabel("Frequency")
                plt.ylabel("Frequency")
                plt.show()
                

    def displayTransformBasisImages(self, size: int, transform_type: TransType):
        """
        显示指定变换的基图像

        参数:
            size: 变换尺寸
            transform_type: 变换类型
        """
        basis_images = self.__generateTransformBasisImages(size, transform_type)
        if basis_images is None:
            return None

        # 创建一个网格布局来显示所有基图像
        img_size = int(np.sqrt(size))

        fig, axes = plt.subplots(img_size, img_size, figsize=(10, 10))

        # 为不同变换类型选择合适的颜色映射
        if transform_type == TransType.DFT:
            cmap = "viridis"  # 彩色映射适合复数值
            # DFT值范围可能较大，所以不设置vmin/vmax
            vmin, vmax = None, None
        else:
            # 对于实数变换，使用黑白色映射
            cmap = LinearSegmentedColormap.from_list("bw", ["black", "white"])
            # 获取所有基图像的最小和最大值
            all_values = np.concatenate([img.flatten() for img in basis_images])
            vmin, vmax = np.min(all_values), np.max(all_values)

        # 在每个子图中显示一个基图像
        for i in range(img_size):
            for j in range(img_size):
                idx = i * img_size + j

                # 处理单个子图或多个子图情况
                if img_size == 1:
                    ax = axes
                else:
                    ax = axes[i, j]

                im = ax.imshow(basis_images[idx], cmap=cmap, vmin=vmin, vmax=vmax)
                ax.set_title(f"Basis {idx}")
                ax.axis("off")

        # 添加颜色条
        plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

        plt.tight_layout()
        transform_name = transform_type.name
        plt.suptitle(f"{transform_name} Transform Basis Images for N={size}", y=1.02)
        plt.show()

        return fig
