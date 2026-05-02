import os
import shutil


def check_folders_with_few_files(directory_path, max_files=5):
    """
    检查指定目录下的文件夹，打印文件数量小于等于max_files个的文件夹

    参数:
        directory_path (str): 要检查的目录路径
        max_files (int): 文件数量

    返回:
        list: 包含文件数量≤max_files的文件夹信息，格式为 [(文件夹名, 文件数), ...]
    """
    folders_with_few_files = []

    try:
        # 遍历目录中的所有项
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)

            # 检查是否是文件夹
            if os.path.isdir(item_path):
                # 获取文件夹中的所有文件和子文件夹
                all_items = os.listdir(item_path)

                # 过滤出文件（排除子文件夹）
                files_only = [file for file in all_items if os.path.isfile(os.path.join(item_path, file))]

                num_files = len(files_only)

                if num_files <= max_files:
                    print(f"文件夹 '{item}' 中有 {num_files} 个文件:")
                    for file_name in files_only:
                        print(f"  - {file_name}")
                    print("-" * 40)

                    # 添加到返回列表中
                    folders_with_few_files.append((item, num_files))

    except PermissionError:
        print(f"权限不足，无法访问: {directory_path}")
    except FileNotFoundError:
        print(f"目录不存在: {directory_path}")
    except Exception as e:
        print(f"发生未知错误: {str(e)}")

    return folders_with_few_files


def delete_empty_or_small_folders(directory_path, max_files=5):
    """
    删除指定目录下文件数量少于等于max_files的文件夹

    参数:
        directory_path (str): 要清理的目录路径
        max_files (int): 最大允许的文件数量，默认5

    返回:
        tuple: (成功删除的数量, 失败删除的数量)
    """
    deleted_count = 0
    failed_count = 0

    try:
        # 首先找出需要删除的文件夹
        folders_to_delete = check_folders_with_few_files(directory_path)

        if not folders_to_delete:
            print("没有找到符合条件的文件夹")
            return (deleted_count, failed_count)

        print(f"\n即将删除以下 {len(folders_to_delete)} 个文件夹:")
        for folder_name, file_count in folders_to_delete:
            print(f"- {folder_name} (包含 {file_count} 个文件)")
        print(f"\n即将删除 {len(folders_to_delete)} 个文件夹:")

        # 确认删除
        confirm = input("\n确定要删除这些文件夹吗？(y/N): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("取消删除操作")
            return (deleted_count, failed_count)

        # 执行删除
        for folder_name, _ in folders_to_delete:
            folder_path = os.path.join(directory_path, folder_name)
            try:
                shutil.rmtree(folder_path)  # 递归删除整个文件夹
                print(f"✓ 已删除文件夹: {folder_name}")
                deleted_count += 1
            except Exception as e:
                print(f"✗ 删除失败 ({folder_name}): {str(e)}")
                failed_count += 1

        print(f"\n删除完成: 成功 {deleted_count} 个, 失败 {failed_count} 个")

    except Exception as e:
        print(f"删除过程中发生错误: {str(e)}")

    return (deleted_count, failed_count)


# 使用方法：
if __name__ == "__main__":
    directory_to_check = "/home/hyj/data/log2026/04/logs"  # 当前目录，可以替换成任意路径
    max_files = 5

    # 只检查不删除
    result = check_folders_with_few_files(directory_to_check)
    print(f"\n找到 {len(result)} 个文件数量≤{max_files}的文件夹")

    # 如果需要删除，调用下面的函数
    delete_empty_or_small_folders(directory_to_check)
