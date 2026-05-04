# """
# 此文件用于测试推理帧率
# """
# # import os
# import json
# # import torch
# # import numpy as np
# # from sklearn.metrics import roc_curve, auc
# # from PIL import Image
# # from torchvision import transforms
# # import matplotlib.pyplot as plt
# # from model import resnet34
# # from model import AlexNet
# # from model import LeNet
# # from sklearn.metrics import confusion_matrix, classification_report,precision_score, recall_score, accuracy_score,f1_score, roc_auc_score, log_loss, cohen_kappa_score
#
# import os
# import time
# import torch
# from PIL import Image
# from torchvision import transforms
# from model import resnet34
#
#
#
#
# def get_label_from_filename(filename):
#     if "gl" in filename:
#         return 0
#     elif "me" in filename:
#         return 1
#     elif "no" in filename:
#         return 2
#     elif "pi" in filename:
#         return 3
#     else:
#         raise ValueError(f"文件名 {filename} 不包含已知的关键字")
#
#
# def main():
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     data_transform = transforms.Compose(
#         [transforms.Resize((224, 224)),
#          transforms.ToTensor(),
#          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#
#     # load image
#     img_path = r"D:\Work\project\brain_class\testdata\all_data"
#     all_files_and_folders = os.listdir(img_path)
#     img_paths = [os.path.join(img_path, item) for item in all_files_and_folders if item.endswith(".jpg")]
#
#     for i in range(len(img_paths)):
#         img = Image.open(img_paths[i]).convert('RGB')
#         img = data_transform(img)
#         img = torch.unsqueeze(img, dim=0)
#
#         json_path = r"D:\Work\project\brain_class\BRAIN_TUMOR\class_indices.json"
#         assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
#         with open(json_path, "r") as f:
#             class_indict = json.load(f)
#         classname = get_label_from_filename(img_paths[i])  # 获取真实标签
#         model = resnet34(num_classes=4).to(device)
#         weights_path = r"D:\Work\project\brain_class\ResNet34.pth"
#         assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
#         model.load_state_dict(torch.load(weights_path))
#         model.eval()
#
#         # 开始计时
#         start_time = time.time()
#
#
#         with torch.no_grad():
#             # predict class
#             output = torch.squeeze(model(img.to(device))).cpu()
#             predict = torch.softmax(output, dim=0)
#             predict_cla = torch.argmax(predict).numpy()  # 预测标签
#
#
#
#
#     total_time = time.time() - start_time
#     avg_time = total_time / len(img_paths)
#
#     print(f"总图片数：{len(img_paths)}")
#     print(f"总推理时间：{total_time:.4f} 秒")
#     print(f"平均单张推理时间：{avg_time * 1000:.2f} ms")
#     print(f"推理帧率（FPS）：{1 / avg_time:.2f}")
#
#
#
#
#
#
#
#
# if __name__ == '__main__':
#     main()




import os
import time
import torch
from PIL import Image
from torchvision import transforms
from model import resnet34
import json

def get_label_from_filename(filename):
    if "gl" in filename:
        return 0
    elif "me" in filename:
        return 1
    elif "no" in filename:
        return 2
    elif "pi" in filename:
        return 3
    else:
        raise ValueError(f"文件名 {filename} 不包含已知的关键字")

def run_once(img_paths, model, device, data_transform):
    """跑一次完整推理，返回总耗时（秒）"""
    start_time = time.time()
    with torch.no_grad():
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            img = data_transform(img).unsqueeze(0).to(device)
            output = model(img)
            _ = torch.argmax(torch.softmax(output, dim=1), dim=1).cpu().numpy()
    return time.time() - start_time

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_path = r"D:\chenwu\segmentation\segcode_zyf\testimage\image"
    img_paths = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.lower().endswith(".tif")]
    assert img_paths, "未找到图片"

    # 只加载一次模型
    model = resnet34(num_classes=4).to(device)
    weights_path = r"D:\chenwu\segmentation\segcode_zyf\unet_segmentation_small_lesion.pth"
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    # 跑 7 次，取平均
    times = []
    for i in range(7):
        t = run_once(img_paths, model, device, data_transform)
        times.append(t)
        print(f"第 {i+1} 次运行耗时：{t:.4f} 秒，FPS：{len(img_paths)/t:.2f}")

    avg_time = sum(times) / len(times)
    print("\n=== 7 次平均结果 ===")
    print(f"平均总耗时：{avg_time:.4f} 秒")
    print(f"平均FPS：{len(img_paths)/avg_time:.2f}")

if __name__ == '__main__':
    main()