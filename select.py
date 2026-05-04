import os
import cv2
import numpy as np


def delete_empty_mask_samples(root_dir):
    """
    删除mask中全图无信号的样本，同时删除对应的images文件

    Args:
        root_dir: 数据集根目录，应包含 train 和 test 子目录，
                  每个子目录下有 images 和 mask 文件夹
    """
    # 定义需要处理的数据集目录
    dataset_dirs = ['train', 'test']

    # 记录删除的文件信息
    deleted_files = []

    for dataset in dataset_dirs:
        # 构建images和mask的路径
        images_dir = os.path.join(root_dir, dataset, 'images')
        masks_dir = os.path.join(root_dir, dataset, 'masks')

        # 检查目录是否存在
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"警告：{dataset} 目录下的 images 或 mask 文件夹不存在，跳过该目录")
            continue

        # 遍历mask文件夹中的所有文件
        for mask_filename in os.listdir(masks_dir):
            # 只处理图像文件
            if not mask_filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                continue

            # 构建mask文件的完整路径
            mask_path = os.path.join(masks_dir, mask_filename)

            try:
                # 读取mask图像
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

                # 检查图像是否读取成功
                if mask is None:
                    print(f"警告：无法读取 {mask_path}，跳过该文件")
                    continue

                # 判断mask是否全为0（无信号）
                if np.all(mask == 0):
                    # 记录要删除的文件
                    deleted_files.append({
                        'dataset': dataset,
                        'mask_file': mask_path,
                        'image_file': os.path.join(images_dir, mask_filename)
                    })

                    # 删除mask文件
                    os.remove(mask_path)
                    print(f"已删除无信号mask文件: {mask_path}")

                    # 删除对应的image文件
                    image_path = os.path.join(images_dir, mask_filename)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        print(f"已删除对应的image文件: {image_path}")
                    else:
                        print(f"警告：对应的image文件不存在: {image_path}")

            except Exception as e:
                print(f"处理文件 {mask_path} 时出错: {str(e)}")

    # 输出删除统计
    print(f"\n=== 删除完成 ===")
    print(f"总共删除了 {len(deleted_files)} 个无信号样本")
    for file_info in deleted_files:
        print(f"- {file_info['dataset']}: {file_info['mask_file']}")


# 使用示例
if __name__ == "__main__":
    # 请修改为你的数据集根目录
    # 目录结构应如下：
    # dataset_root/
    # ├── train/
    # │   ├── images/
    # │   └── mask/
    # └── test/
    #     ├── images/
    #     └── mask/
    dataset_root = r"E:\code\code\DATA"

    # 执行删除操作
    delete_empty_mask_samples(dataset_root)