import numpy as np
from enum import Enum


class NoiseType(Enum):
    """噪声类型枚举类"""

    GAUSSIAN = 1  # 高斯噪声
    POISSON = 2  # 泊松噪声
    SALT_AND_PEPPER = 3  # 盐和胡椒噪声
    SPECKLE = 4  # 斑点噪声
    SPARSE = 5  # 稀疏噪声
    MULTIPLICATIVE = 6  # 乘法噪声


class GaussianNoise(Enum):
    Marsaglia = 1  # Marsaglia方法


class Noise:
    def add_gaussian_noise(
        self,
        image: np.ndarray,
        miu: float = 0,
        sigma: float = 1,
        noise_intensity: float = 1,
        type: GaussianNoise = GaussianNoise.Marsaglia,
        minimum_size=10,
    ) -> np.ndarray:
        """
        添加高斯噪声
        :param image: 输入图像
        :param miu: 均值
        :param sigma: 标准差
        :return: 添加噪声后的图像
        """
        match type:
            case GaussianNoise.Marsaglia:
                # Marsaglis采样过程 => 随机生成 u,v 属于 [-1,1] 的均匀分布
                # 计算u,v 的平方和 s => 如果 s <= 1 && !=0, 则计算 x,y
                #  否则重新采样
                # 计算 factor = √(-2ln(s)/s)
                # 计算 x = u * factor, y = v * factor
                # z1 = x * σ + μ , z2 = y * σ + μ
                # 完成采样
                n, m = image.shape
                size = n * m
                n_size = size // 2 + 1
                # 初始化存储数组
                s_valid = np.zeros(n_size)
                u_valid = np.zeros(n_size)
                v_valid = np.zeros(n_size)

                batch_size = n_size
                valid_size = 0
                while valid_size < n_size:
                    u = np.random.uniform(-1, 1, size=batch_size * 2)
                    v = np.random.uniform(-1, 1, size=batch_size * 2)

                    s = u**2 + v**2
                    valid = (s <= 1) & (s != 0)

                    up_limmit = valid_size + np.sum(valid)

                    if up_limmit > n_size:
                        up_limmit = n_size

                    u_valid[valid_size:up_limmit] = u[valid][: up_limmit - valid_size]
                    v_valid[valid_size:up_limmit] = v[valid][: up_limmit - valid_size]
                    s_valid[valid_size:up_limmit] = s[valid][: up_limmit - valid_size]

                    valid_size = up_limmit
                    batch_size = max(n_size - valid_size, minimum_size)

                # 计算 factor = √(-2ln(s)/s)
                factor = np.sqrt(-2 * np.log(s_valid) / s_valid)
                z1 = u_valid * factor * sigma + miu
                z2 = v_valid * factor * sigma + miu
                # 保留size个采样并reshape为(n,m)
                noise_sample = np.concatenate([z1, z2])[:size].reshape((n, m))
                # 添加噪声
                noisy_image = image + noise_sample * noise_intensity
                # clip
                noisy_image = np.clip(noisy_image, 0, 255)
                return noisy_image.astype(np.uint8)
            case _:
                raise NotImplementedError(
                    "The GaussianNoise type is not implemented, please check the code."
                )

    def add_salt_noise(
        self, img: np.ndarray, p_salt: float = 0.05, p_pepper: float = 0.05
    ) -> np.ndarray:
        """
        添加盐噪声
        :param img: 输入图像
        :param p_salt: 盐噪声概率
        :param p_pepper: 胡椒噪声概率
        :return: 添加噪声后的图像
        """
        # 生成随机噪声矩阵
        noise = np.random.rand(*img.shape)
        noisy_img = img.copy()
        noisy_img[noise < p_salt] = 255
        noisy_img[noise > 1 - p_pepper] = 0
        return noisy_img.astype(np.uint8)
