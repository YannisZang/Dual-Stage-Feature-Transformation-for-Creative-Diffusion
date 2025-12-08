import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

def apply_gabor(image: Image.Image, filter_factor=2.0, device="cuda"):  
    """
    Input: PIL Image (H, W, 3)
    Output: PIL Image (H, W, 3)
    """

    sample = TF.to_tensor(image).unsqueeze(0).to(device)  # [1, 3, H, W]
    assert sample.ndim == 4, "sample must be [B, C, H, W]!"

    device = sample.device
    b, c, h, w = sample.shape

    kernel_size = 9
    sigma = 3.0
    lambd = 6.0
    gamma = 0.7

    # ensure all thetas are torch.Tensors on GPU
    thetas = [torch.tensor(t, device=device).float() for t in
            [0, torch.pi/4, torch.pi/2, 3*torch.pi/4]]
    
    # 
        
    # thetas = [torch.tensor(t, device=device).float() for t in
    #     [0, torch.pi/8, torch.pi/4, 3*torch.pi/8,
    #     torch.pi/2, 5*torch.pi/8, 3*torch.pi/4, 7*torch.pi/8]]

    out = torch.zeros_like(sample)

    y, x = torch.meshgrid(
        torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device).float(),
        torch.arange(-(kernel_size//2), kernel_size//2 + 1, device=device).float(),
        indexing='ij',
    )

    with torch.no_grad():
        for theta in thetas:
            x_t = x * torch.cos(theta) + y * torch.sin(theta)
            y_t = -x * torch.sin(theta) + y * torch.cos(theta)

            gb = torch.exp(-(x_t**2 + (gamma * y_t)**2) / (2 * sigma**2)) * \
                torch.cos(2 * torch.pi * x_t / lambd)

            gb = gb / gb.abs().sum()
            kernel = gb.to(sample.dtype).expand(c, 1, kernel_size, kernel_size).to(device)

            filtered = F.conv2d(sample, kernel,
                                padding=kernel_size//2,
                                groups=c)
            out += filtered

    out = out / len(thetas)
    out = sample + filter_factor * out
    output = out.squeeze(0).cpu().clamp(0, 1)  # [3, H, W]
    output_img = TF.to_pil_image(output)
    return output_img