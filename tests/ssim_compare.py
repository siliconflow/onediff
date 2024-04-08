from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# 加载图像
def validate():
    image1 = np.array(Image.open('/share_nfs/civitai/20240407-163408.jpg').convert('RGB'))
    image2 = np.array(Image.open('/src/test/output_enterprise_sd.png').convert('RGB'))
    # 计算SSIM
    ssim_index = ssim(image1, image2, multichannel=True, win_size=3)
    print("SSIM:", ssim_index)
    if ssim_index < 0.90:
        print("验证失败，将中止工作流")
        sys.exit(1)  # 非0状态码表示失败
    print("验证成功")
    sys.exit(0)  # 0状态码表示成功

if __name__ == "__main__":
    validate()