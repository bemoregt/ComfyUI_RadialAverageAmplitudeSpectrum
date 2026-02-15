import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


def radial_average_amplitude_spectrum(image_np):
    """
    image_np: H x W grayscale float array (0~1)
    Returns: (freqs, radial_avg) 1D arrays
    """
    H, W = image_np.shape
    # 2D FFT, shift DC to center
    fft2 = np.fft.fft2(image_np)
    fft2_shifted = np.fft.fftshift(fft2)
    amplitude = np.abs(fft2_shifted)

    cy, cx = H // 2, W // 2
    # Build radius map
    y_idx, x_idx = np.indices((H, W))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    radial_sum = np.zeros(max_r)
    radial_count = np.zeros(max_r)

    mask = r < max_r
    np.add.at(radial_sum, r[mask], amplitude[mask])
    np.add.at(radial_count, r[mask], 1)

    radial_count = np.where(radial_count == 0, 1, radial_count)
    radial_avg = radial_sum / radial_count

    freqs = np.arange(max_r)
    return freqs, radial_avg


class RadialAvgAmpSpectrumNode:
    """
    ComfyUI custom node: Radial Average Amplitude Spectrum
    Input : IMAGE
    Output: IMAGE (log-log plot of 1D radial average amplitude spectrum)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "plot_width": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "plot_height": ("INT", {"default": 512, "min": 256, "max": 2048, "step": 64}),
                "line_color": ("STRING", {"default": "red"}),
                "log_x": ("BOOLEAN", {"default": True}),
                "log_y": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("spectrum_image",)
    FUNCTION = "compute"
    CATEGORY = "analysis"

    def compute(self, image, plot_width, plot_height, line_color, log_x, log_y):
        # image: [B, H, W, C] float32 tensor, 0~1
        results = []

        for i in range(image.shape[0]):
            img_np = image[i].cpu().numpy()  # H x W x C

            # Convert to grayscale
            if img_np.shape[2] == 1:
                gray = img_np[:, :, 0]
            else:
                # Luminance weights
                gray = (
                    0.2126 * img_np[:, :, 0]
                    + 0.7152 * img_np[:, :, 1]
                    + 0.0722 * img_np[:, :, 2]
                )

            freqs, radial_avg = radial_average_amplitude_spectrum(gray)

            # Avoid log(0)
            start = 1  # skip DC (freq=0)
            freqs_plot = freqs[start:]
            radial_plot = radial_avg[start:]
            radial_plot = np.where(radial_plot <= 0, 1e-12, radial_plot)

            dpi = 100
            fig_w = plot_width / dpi
            fig_h = plot_height / dpi

            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.plot(freqs_plot, radial_plot, color=line_color, linewidth=1.2)

            if log_x:
                ax.set_xscale("log")
            if log_y:
                ax.set_yscale("log")

            ax.set_xlabel("frequency")
            ax.set_ylabel("amplitude")
            ax.set_title("Radial Average Amplitude Spectrum")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            fig.tight_layout()

            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            plt.close(fig)
            buf.seek(0)

            pil_img = Image.open(buf).convert("RGB")
            # Resize to exact requested size
            pil_img = pil_img.resize((plot_width, plot_height), Image.LANCZOS)
            out_np = np.array(pil_img).astype(np.float32) / 255.0  # H x W x 3

            results.append(torch.from_numpy(out_np))

        # Stack back to [B, H, W, C]
        output = torch.stack(results, dim=0)
        return (output,)


NODE_CLASS_MAPPINGS = {
    "RadialAvgAmpSpectrum": RadialAvgAmpSpectrumNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadialAvgAmpSpectrum": "Radial Avg Amplitude Spectrum",
}
