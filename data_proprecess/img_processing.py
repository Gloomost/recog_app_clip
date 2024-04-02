from PIL import Image
import pandas as pd
import os


def merge_images_in_subfolders(parent_folder, output_folder, child_folder, rows_per_image=3):
    for subdir in os.scandir(os.path.join(parent_folder, child_folder)):
        if not subdir.is_dir():
            continue

        subdir_name = subdir.name
        images_list = []

        for img_file in os.listdir(os.path.join(parent_folder, child_folder, subdir_name)):
            if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(parent_folder, child_folder, subdir_name, img_file)
                images_list.append((img_path, Image.open(img_path)))

        if images_list:
            num_images = len(images_list)
            cols = min(num_images, rows_per_image)
            rows = (num_images + rows_per_image - 1) // rows_per_image

            # 计算单个子图片的实际宽度和高度
            total_width = sum(im.width for _, im in images_list[:cols])
            row_heights = [im.height for _, im in images_list]
            max_height = sum(sorted(row_heights, reverse=True)[:rows])

            new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))

            col_idx, row_idx = 0, 0
            cur_col_width = 0
            for img_path, im in images_list:
                col_width = im.width
                row_height = im.height

                if col_idx == cols:
                    col_idx = 0
                    row_idx += 1
                    cur_col_width = 0

                pos_x = cur_col_width
                pos_y = int(row_idx * max_height / rows)

                resized_im = im.resize((col_width, row_height))  # 若需要等高则保留这行，否则可删除以便保持原始比例
                new_im.paste(resized_im, (pos_x, pos_y))

                cur_col_width += col_width
                col_idx += 1

            output_filename = f"{subdir_name}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            new_im.save(output_path)


def merge():
    parent_folder = '../data'
    output_folder = '../data/merged_images'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    merge_images_in_subfolders(parent_folder, output_folder, 'Info')


def save_to_classes():
    img_path = '../data/merged_images'
    out_path = '../data/class_data'
    img_label = pd.read_excel('../data/data_Info/new_label_title.xlsx')
    img_label[['label1', 'label2', 'label3']] = img_label[['label1', 'label2', 'label3']].astype(str)
    img_label['label'] = img_label.apply(lambda row:'-'.join(row[['label1','label2','label3']]), axis=1)
    # print(img_label.head(5))
    label_dict = {}
    for idx in img_label.index:
        label_dict[img_label['app_id'][idx]] = img_label['label'][idx]
    # print(label_dict)
    for filepath, _, files in os.walk(img_path):  # 将 dirs 改为 files
        for file in files:  # 将 dir 改为 file
            print(file)
            f1 = os.path.join(filepath, file)
            img = file[:-4]  # 假设图片文件名不含额外的文本信息，仅含 app_id 和扩展名
            img_label = label_dict[img]
            new_path = os.path.join(out_path, img_label)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            with open(f1, 'rb') as fp1:
                b1 = fp1.read()
            ext = os.path.splitext(file)[-1]  # 获取文件扩展名
            f2 = os.path.join(new_path, img + ext)  # 添加扩展名后再保存
            with open(f2, 'wb') as fp2:
                fp2.write(b1)


if __name__ == '__main__':
    merge()
    save_to_classes()