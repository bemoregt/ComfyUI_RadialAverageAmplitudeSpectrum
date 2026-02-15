# ComfyUI Radial Average Amplitude Spectrum

A ComfyUI custom node that computes and visualizes the **Radial Average Amplitude Spectrum** of an input image — a 1D frequency-domain analysis tool useful for characterizing image texture, noise, and spatial frequency content.

![example](https://github.com/bemoregt/ComfyUI_RadialAverageAmplitudeSpectrum/blob/main/ScrShot%209.png)

---

## What It Does

1. Converts the input image to grayscale using luminance weights.
2. Applies a 2D FFT and shifts the DC component to the center.
3. Computes the amplitude spectrum (`|FFT|`).
4. Averages the amplitude values at each integer radial distance from the center, producing a 1D spectrum.
5. Renders the spectrum as a log-log plot and returns it as a standard ComfyUI `IMAGE` tensor.

---

## Installation

### Option A — Copy into custom_nodes

```bash
cp -r ComfyUI_RadialAvgAmpSpectrum /path/to/ComfyUI/custom_nodes/
```

### Option B — Clone directly

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-repo/ComfyUI_RadialAvgAmpSpectrum.git
```

Then restart ComfyUI. The node will appear under the **analysis** category.

---

## Dependencies

All dependencies are part of the standard Python scientific stack that ComfyUI already ships with:

| Package | Purpose |
|---------|---------|
| `numpy` | FFT and array operations |
| `matplotlib` | Plot rendering |
| `Pillow` | Image conversion |
| `torch` | Tensor I/O (ComfyUI standard) |

No additional `pip install` steps are required in a typical ComfyUI environment.

---

## Node Reference

**Node name:** `Radial Avg Amplitude Spectrum`
**Category:** `analysis`

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `image` | `IMAGE` | Input image (any resolution, RGB or grayscale) |
| `plot_width` | `INT` | Width of the output plot image in pixels (default: 512) |
| `plot_height` | `INT` | Height of the output plot image in pixels (default: 512) |
| `line_color` | `STRING` | Matplotlib color string for the spectrum line (default: `red`) |
| `log_x` | `BOOLEAN` | Use logarithmic scale on the frequency axis (default: true) |
| `log_y` | `BOOLEAN` | Use logarithmic scale on the amplitude axis (default: true) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `spectrum_image` | `IMAGE` | Log-log plot of the 1D radial average amplitude spectrum |

---

## Interpreting the Output

- **X axis (frequency):** Spatial frequency in cycles per pixel, expressed as radial distance from the DC center (1 = lowest non-DC frequency, Nyquist ≈ min(H,W)/2).
- **Y axis (amplitude):** Mean amplitude of the FFT at that radial frequency bin.
- A steep negative slope indicates a smooth, low-frequency-dominant image.
- A flat or slowly decaying spectrum indicates high-frequency content (sharp edges, noise, fine texture).

---

## License

MIT
