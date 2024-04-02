import clip
from selenium import webdriver
import urllib
import urllib.request
from bs4 import BeautifulSoup
import json, os, requests
import torch
import cv2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import shutil


class Info:
    def __init__(self):
        self.user_agent = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                            'AppleWebKit/537.36 (KHTML, like Gecko) '
                            'Chrome/120.0.0.0 Safari/537.36')
    def Google_info(self, url):
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent',
                           self.user_agent)
            page = urllib.request.urlopen(req, None, 60)
            html = page.read().decode('utf-8')
        except BaseException as e:
            print('bad urlopen,wait next time, error: {}'.format(e))
            return []
        if html != '':
            soup = BeautifulSoup(html, 'lxml')
        else:
            return []
        try:
            img_url = [img['src'] for img in soup.find_all(
                'img', {'alt': 'Screenshot image'})]
            if len(img_url) == 0:
                img_url = [img['src'] for img in
                           soup.find_all('img', {'alt': '屏幕截图图片'})]
            return img_url
        except:
            return []

    def Apple_info(self, url):
        try:
            driver = webdriver.Firefox()
            driver.get(url)
            html = driver.page_source
            driver.close()
        except:
            return []
        if html != '':
            soup = BeautifulSoup(html, 'html.parser')
        else:
            return []
        try:
            font_tags = soup.find_all('script',
                                      {'name': "schema:software-application",
                                       'type': "application/ld+json"})
            json_fron = json.loads(font_tags[0].text)
            return json_fron['screenshot']
        except:
            return []


def read_txt_to_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]


def evaluate(model, img_pah):
    _, preprocess = clip.load("ViT-B/32", device=device)
    # 加载模型
    # 确保模型在评估模式，这对于使用了如dropout和batch normalization层的模型是必要的
    model.eval()
    classes = labels.split('\n')
    image = Image.open(img_pah)
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(classes)]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
    # 选取参数最高的标签
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)  # 对图像描述和图像特征
    # print(similarity)
    values, indices = similarity[0].topk(1)
    return classes[indices.item()], values.item()


def get_app(urls):
    try:
        # print(urls)
        for inx, url in enumerate(urls):
            res = requests.get(url, stream=True)
            name = os.path.join(str(inx) + '.jpg')
            with open(name, 'wb') as f:
                res.raw.decode_content = True
                shutil.copyfileobj(res.raw, f)
        new_im = merge_images_in_subfolders()
        new_im.save('temp_img.jpg')
        label, score = evaluate(model, 'temp_img.jpg')
        for inx, _ in enumerate(urls):
            os.remove(str(inx) + '.jpg')
        os.remove('temp_img.jpg')
        return label, score
    except:
        return '',''


def merge_images_in_subfolders(rows_per_image=3):
    current_directory = os.getcwd()
    images_list = []
    for img_file in os.listdir(current_directory):
        if img_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(current_directory, img_file)
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

        return new_im


def one_url(url):
    info = Info()
    if 'google' in url:
        img_list = info.Google_info(str(url))
    else:
        img_list = info.Apple_info(str(url))
    print(img_list)
    label, score = get_app(img_list)
    return label, score


if __name__ == '__main__':
    class_names = read_txt_to_list('../data/data_info/label.txt')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, preprocess = clip.load("ViT-B/32", device=device)
    model = torch.load('../output/model1000.pkl')
    # 加载模型
    # 确保模型在评估模式，这对于使用了如dropout和batch normalization层的模型是必要的
    model.eval()
    with open('../data/data_info/label.txt', 'r', encoding='utf-8') as f:
        labels = f.read()

    label, score = one_url('https://play.google.com/store/apps/details?id=car.hillside.rush.racing')
    print(label, score)