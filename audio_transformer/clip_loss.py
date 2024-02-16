import torch
import clip
import platform
import math
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.device, self.device_name = (torch.device('cuda'),
                               torch.cuda.get_device_name(torch.cuda.current_device())) \
            if torch.cuda.is_available() else ('cpu', platform.processor())

    def forward(self, image, prompt):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        sim = self.clip_model(image, text=clip.tokenize([prompt]).to(self.device))[0] / 100
        return 1-sim

    def cal_clip_sim(self, image_list, text_list):
        images = torch.stack([self.preprocess(img) for img in image_list],dim=0).to(self.device)
        print(images.shape)
        text = clip.tokenize(text_list).to(self.device)

        image_features = self.clip_model.encode_image(images)
        text_features = self.clip_model.encode_text(text)

        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T) #* math.exp(0.07)
        loss = 1 - sim
        return loss
