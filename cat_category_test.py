import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms , models

from torchvision.utils import make_grid

if torch.cuda.is_available():
    torch.backends.cudnn.detetministic = True

from torchvision.datasets import ImageFolder

import glob
from torchvision import datasets
from tqdm import tqdm
import timm

#streamlitで実行用のライブラリ
import streamlit as st
import PIL.Image

train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder("./dataset/train",transform=train_transform)
test_dataset = datasets.ImageFolder("./dataset/test",transform=test_transform)

train_loader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

def predict(model,dataloader,device):
    test_acc = 0
    model.eval()
    with torch.no_grad():
        for images,labels in tqdm(dataloader):
            images , labels = images.to(device) , labels.to(device)
            outputs=model(images)
            acc = (outputs.max(1)[1] == labels).sum()
            test_acc += acc.item()
    avg_test_acc = test_acc / len(dataloader.dataset)
    return avg_test_acc


num_epochs = 30
num_classes = len(train_dataset.classes)

model = timm.create_model("resnet18",pretrained=False,num_classes=num_classes)
model.load_state_dict(torch.load(f"cat_dog_model.pth" ,map_location=torch.device('cpu')))
#model.to(device)

st.title("犬猫画像分類デモアプリ")

uploaded_file = st.file_uploader("犬か猫の画像をアップロードしてください",type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file).convert("RGB")
    st.image(image,caption="アップロードされた画像",use_column_width=True)

    input_tensor = test_transform(image).unsqueeze(0)

    with st.spinner("予測中..."):
        with torch.no_grad():
            outputs = model(input_tensor)
            preds = outputs.max(1)[1]  # 予測ラベル

    st.markdown(f"### 予測結果: {'猫' if preds.item() == 0 else '犬'}")