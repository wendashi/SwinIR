import cv2
import numpy as np
import torch
import time
import onnxruntime  
import os
def srxn(sr_xn, sr_input):

    # 检查保存结果的目录是否存在
    save_dir = f'/home/stone/Desktop/SR/SwinIR/results/swinir_onnx_x{sr_xn}'
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

        sr_output_mid = os.path.join(save_dir, f"{imgname}_SwinIRx{sr_xn}_onnx.png")
        cv2.imwrite(sr_output_mid, output_mid)

        output = main_x4(sr_output_mid)
    
    saved_image_path = os.path.join(save_dir, f"{imgname}_SwinIRx{sr_xn}_onnx.png")
        
    save_success = cv2.imwrite(saved_image_path, output)

    if save_success:
        print(f"Image successfully saved at: {os.path.abspath(saved_image_path)}")
    else:
        print("Failed to save the image.")

    sr_output = saved_image_path

    return sr_output

def main_x2(sr_input):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
                # 假设 img_lq 是一个存储在CUDA上的Tensor (NCHW-RGB)
        if img_lq.is_cuda:
            numpy_input = img_lq.cpu().numpy()
        else:
            numpy_input = img_lq.numpy()

        # check is using GPU?
        print(onnxruntime.get_device())

        # runtime
        ort_session = onnxruntime.InferenceSession('/home/stone/Desktop/SR/SwinIR/swinir_real_sr_large_model_dynamic.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # double check is using GPU?
        print(ort_session.get_providers())

        # onnx 的输入是 numpy array 而非 tensor!    
        ort_inputs = {'input': numpy_input}

        ort_output = ort_session.run(['output'], ort_inputs)[0]
        # tensor 转 numpy
        ort_output = torch.from_numpy(ort_output)

        output = ort_output[..., :h_old * 4, :w_old * 4]

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
        
        # output = test(img_lq)
        start_time = time.time() # start time

        # 假设 img_lq 是一个存储在CUDA上的Tensor (NCHW-RGB)
        if img_lq.is_cuda:
            numpy_input = img_lq.cpu().numpy()
        else:
            numpy_input = img_lq.numpy()

        # check is using GPU?
        print(onnxruntime.get_device())

        # runtime
        ort_session = onnxruntime.InferenceSession('/home/stone/Desktop/SR/SwinIR/swinir_real_sr_large_model_dynamic.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

        # double check is using GPU?
        print(ort_session.get_providers())

        # onnx 的输入是 numpy array 而非 tensor!    
        ort_inputs = {'input': numpy_input}

        ort_output = ort_session.run(['output'], ort_inputs)[0]
        # tensor 转 numpy
        ort_output = torch.from_numpy(ort_output)

        output = ort_output[..., :h_old * 4, :w_old * 4]

        stop_time = time.time() # start time
        print(f'Test time: {stop_time - start_time:.2f}s')  

    # save image
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    # cv2.imwrite(f'{save_dir}/{imgname}_SwinIR_onnx.png', output)

    return output



if __name__ == '__main__':
    sr_input = '/home/stone/Desktop/SR/SwinIR/testsets/test-120/49.jpg'
    sr_xn = 8
    sr_output = srxn(sr_xn, sr_input)
