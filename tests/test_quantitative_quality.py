from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import unittest

class QuantizeQuality(unittest.TestCase):

    def test_validate(self):
        image1 = np.array(Image.open('/share_nfs/civitai/20240407-163408.jpg').convert('RGB'))
        image2 = np.array(Image.open('/src/onediff/output_enterprise_sd.png').convert('RGB'))
        # Calculate SSIM
        ssim_index = ssim(image1, image2, multichannel=True, win_size=3)
        print("SSIM:", ssim_index)
        self.assertTrue(ssim_index > 0.89,  "SSIM Validation fails, and the workflow is aborted")
        

if __name__ == "__main__":
    unittest.main()
