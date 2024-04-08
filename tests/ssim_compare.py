from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys

def validate():
    image1 = np.array(Image.open('/share_nfs/civitai/20240407-163408.jpg').convert('RGB'))
    image2 = np.array(Image.open('/share_nfs/quant_test/output_enterprise_sd.png').convert('RGB'))
    # Calculate SSIM
    ssim_index = ssim(image1, image2, multichannel=True, win_size=3)
    print("SSIM:", ssim_index)
    if ssim_index < 0.90:
        print("Validation fails, and the workflow is aborted")
        sys.exit(1)  # A status code other than 0 indicates a failure
    print("Verification successful")
    sys.exit(0)  # 

if __name__ == "__main__":
    validate()