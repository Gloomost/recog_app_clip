import pandas as pd
from model.clip import load_pretrained_clip_model, convert_models_to_fp32
import torch.optim as optim
import torch.nn as nn
import torch
import clip


def train(epoch, batch_size, learning_rate, df):
    model, preprocess, dataset = load_pretrained_clip_model('ViT-B/32',
                                                                   df, device)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                   shuffle=True,num_workers=4)
    # 假设使用了4个数据加载工作线程，并且需要打乱数据顺序

    # 设置损失函数和优化器
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98),
                           eps=1e-6, weight_decay=0.2)

    for i in range(epoch):
        for batch in train_dataloader:
            list_image, list_txt = batch
            # 在此处实际训练中可能不需要BERT编码，根据需求决定是否使用
            # texts = get_bert_embeddings(list_txt).to(device)
            texts = clip.tokenize(list_txt).to(device)
            images = list_image.to(device)
            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) +
                          loss_txt(logits_per_text, ground_truth)) / 2
            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        print(f'Epoch [{i + 1}] Loss: {total_loss.item():.3f}')

    torch.save(model.state_dict(), 'clip-model/model.pth')

def main():
    epoch = 50
    batch_size = 2
    learning_rate = 5e-5
    ad_path = '../data/merged_images'
    df = pd.read_csv('../data/data_info/img&keywords.csv')
    train(epoch, batch_size, learning_rate, ad_path, df)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    main()