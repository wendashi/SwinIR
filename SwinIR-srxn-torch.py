import cv2
import numpy as np
import torch
import time
import onnxruntime  
import os
from models.network_swinir import SwinIR as net


def srxn(sr_xn, sr_input):

    # 检查保存结果的目录是否存在
    save_dir = f'/home/stone/Desktop/SR/SwinIR/results/swinir_torch_x{sr_xn}'
    if not os.path.exists(save_dir):
        # 如果目录不存在，则创建目录
        os.makedirs(save_dir)

    path = sr_input
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    if sr_xn == 2:
        output = main_x2(sr_input)
    elif sr_xn == 4:
        output = main_x4(sr_input)
    elif sr_xn == 8:
        output_mid = main_x2(sr_input)

        sr_output_mid = os.path.join(save_dir, f"{imgname}_SwinIRx{sr_xn}_torch.png")
        cv2.imwrite(sr_output_mid, output_mid)

        output = main_x4(sr_output_mid)
    
    saved_image_path = os.path.join(save_dir, f"{imgname}_SwinIRx{sr_xn}_torch.png")
        
    save_success = cv2.imwrite(saved_image_path, output)

    if save_success:
        print(f"Image successfully saved at: {os.path.abspath(saved_image_path)}")
    else:
        print("Failed to save the image.")

    sr_output = saved_image_path

    return sr_output

def main_x2(sr_input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 torch 模型(onnx不需要这里)
    model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'
    pretrained_model = torch.load('/home/stone/Desktop/SR/SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    
    # read image
    path = sr_input
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # image to HWC-BGR, float32
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # 长和宽都 resize 到 1/2（x2专用）
    height, width = img_lq.shape[:2]
    img_lq = cv2.resize(img_lq, (width // 2, height // 2))
    
    # HCW-BGR to CHW-RGB
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))

    # CHW-RGB to NCHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  

    # inference
    with torch.no_grad():
        window_size = 8
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        
        start_time = time.time() # start time

        output = model(img_lq)

        output = output[..., :h_old * 4, :w_old * 4]

        stop_time = time.time() # start time
        print(f'Test time: {stop_time - start_time:.2f}s')  

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    # sr_output = cv2.imwrite(f'{save_dir}/{imgname}_SwinIRx2_torch_1.png', output)

    return output

def main_x4(sr_input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载 torch 模型(onnx不需要这里)
    model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
    param_key_g = 'params_ema'
    pretrained_model = torch.load('/home/stone/Desktop/SR/SwinIR/model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    model.eval()
    model = model.to(device)
    
    # read image
    path = sr_input
    (imgname, imgext) = os.path.splitext(os.path.basename(path))

    # image to HWC-BGR, float32
    img_lq = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.

    # HCW-BGR to CHW-RGB
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  
    
    # CHW-RGB to NCHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  

    # inference
    with torch.no_grad():
        window_size = 8
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        
        start_time = time.time() # start time

        output = model(img_lq)

        output = output[..., :h_old * 4, :w_old * 4]

        stop_time = time.time() # start time
        print(f'Test time: {stop_time - start_time:.2f}s')  

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    # sr_output = cv2.imwrite(f'{save_dir}/{imgname}_SwinIRx4_torch_1.png', output)

    return output


if __name__ == '__main__':
    sr_input = '/home/stone/Desktop/SR/SwinIR/testsets/test-120/49.jpg'
    sr_xn = 4
    sr_output = srxn(sr_xn, sr_input)
