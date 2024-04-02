import clip
import torch.utils.data as data_utils
from PIL import Image


class ImageCaptionDataset(data_utils.Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]))
        caption = self.caption[idx]
        return images, caption


def load_pretrained_clip_model(model_path, df, device):
    model, preprocess = clip.load(model_path, device=device, jit=False)
    if device == "cpu":
        model.float()
    else:
        clip.model.convert_weights(model)
    return model, preprocess, ImageCaptionDataset(df, preprocess)

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad is not None:
            p.grad.data = p.grad.data.float()



# def get_bert_embeddings(texts):
#     # 加载预训练的BERT模型和分词器
#     model = BertModel.from_pretrained('bert-base-uncased')
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     embeddings = []
#     for text in texts:
#         # 对文本进行分词，并转换为PyTorch张量
#         inputs = tokenizer(text, truncation=True, max_length=77,
#                            padding='max_length', return_tensors='pt')
#
#         # 获取BERT的输出
#         outputs = model(**inputs)
#
#         # 使用[CLS]令牌的隐藏状态作为句子嵌入
#         sentence_embedding = outputs.last_hidden_state[0, 0].detach().numpy()
#
#         embeddings.append(sentence_embedding)
#
#     return torch.tensor(embeddings)