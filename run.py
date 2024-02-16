import sys
import argparse, os, sys, glob
import librosa
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from torch import nn, optim
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from transformers import AutoFeatureExtractor
import torch.nn.functional as F
from audio_transformer.clip_loss import *
import torchvision.transforms as transforms
from torch import optim
import clip
from audio_transformer.audio2caption import Audio2Caption
from torch.optim.lr_scheduler import ReduceLROnPlateau
from audio_transformer.sentence_attention import SentenceAttention
from torchvision.transforms.functional import to_pil_image
import glob
from audio_clip.model import AudioCLIP
from utils.transforms import ToTensor1D
import random

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# The learning rate adjustment function.
def get_lr(t, initial_lr, rampdown=0.50, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.allow_unused=True
    model.cuda()
    model.eval()
    return model

def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def split_prompt(prompt):
    vals = prompt.rsplit(':', 2)
    vals = vals + ['', '1', '-inf'][len(vals):]
    return vals[0], float(vals[1]), float(vals[2])

class ReplaceGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_forward, x_backward):
        ctx.shape = x_backward.shape
        return x_forward

    @staticmethod
    def backward(ctx, grad_in):
        return None, grad_in.sum_to_size(ctx.shape)

replace_grad = ReplaceGrad.apply

class Prompt(nn.Module):
    def __init__(self, embed, weight=1., stop=float('-inf')):
        super().__init__()
        self.register_buffer('embed', embed)
        self.register_buffer('weight', torch.as_tensor(weight))
        self.register_buffer('stop', torch.as_tensor(stop))

    def forward(self, input):
        input_normed = F.normalize(input.unsqueeze(1), dim=2)
        embed_normed = F.normalize(self.embed.unsqueeze(0), dim=2)
        dists = input_normed.sub(embed_normed).norm(dim=2).div(2).arcsin().pow(2).mul(2)
        dists = dists * self.weight.sign()
        return self.weight.abs() * replace_grad(dists, torch.maximum(dists, self.stop)).mean()


def main():
    # ==================== load hyperparameters ====================
    print("Optimization")

    downsampling_factor = 8 
    latent_channels = 4
    image_width = 512
    image_height = 512
    n_iter = 1
    ddim_eta = 0
    ddim_steps = 40
    batch_size = 1
    scale = 7.5  # 7.5
    n_samples = 1
    text_embedding = 768

    outpath = 'image path'  # Output image path
    my_config = 'pre_models/configs/stable-diffusion/v1-inference.yaml'
    ckpt = 'pre_models/stable_diffusion/sd-v1-4.ckpt'
    act_config = 'pre_models/configs/audio-transformer/settings_audioset.yaml'
    audioclip_model_path = 'pre_models/audio_clip/AudioCLIP-Full-Training.pt'
    audio_meta = 'audioset/audioset_test/'

    resize = transforms.Resize(224)
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                    std=[0.26862954, 0.26130258, 0.27577711])

    sample_path = os.path.join(outpath, "output")
    os.makedirs(outpath, exist_ok=True)
    os.makedirs(sample_path, exist_ok=True)

    config = OmegaConf.load(f"{my_config}")
    model = load_model_from_config(config, f"{ckpt}")

    ac_model = Audio2Caption(f'{act_config}')
    captions, attention_list, audio_file_names = ac_model.audio2txt()

    # Load audio for audioclip optimization
    SAMPLE_RATE = 44100
    aclp = AudioCLIP(pretrained=f'{audioclip_model_path}')

    perceptor, _ = clip.load("ViT-B/32", device="cuda")
    perceptor.requires_grad_(False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    aclp = aclp.to(device)
    aclp.eval()

    sa = SentenceAttention()
    error_list = []
    sampler = PLMSSampler(model)

    for idx_caption, caption_ in enumerate(captions):
        random.seed()
        r = random.randrange(1,99999)
        try:
            #Extract audio feature
            with torch.no_grad():
                audio_path = audio_meta + audio_file_names[idx_caption]
                track, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)
                spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)).to(device))
                spec = np.ascontiguousarray(spec.detach().cpu().numpy()).view(np.complex64)
                pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
                if 441000 - track.shape[0] >0:
                    track = np.pad(track, (0, 441000- track.shape[0]), 'constant', constant_values=0)
                audio = list()
                audio.append((track[:441000], pow_spec))
                audio_transforms = ToTensor1D()
                audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
                audio = torch.cat(2*[audio], dim=0).to(device)
                ((audio_features, _, _), _), _ = aclp(audio=audio)
                audio_features = audio_features[0].unsqueeze(0)  
            
            txt, weight, stop = split_prompt(caption_)
            embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
            clip_score_loss = Prompt(embed, weight, stop).to(device)

            sentence_attention = sa.get_attention(caption_)
            nouns_idx = []
            for n_idx, item in enumerate(sentence_attention):
                if item > 0.7:
                    nouns_idx.append(n_idx)
            
            no_noun_flag = False
            if len(nouns_idx) == 0:
                no_noun_flag = True
                nouns_idx.append(1)

            with torch.no_grad():
                tmp_z, tmp_z_attention, index_list = model.get_learned_conditioning([caption_], [attention_list[idx_caption]])

                n_list = []
                n_list.append(['w', 0, 0])  # sos
                w_count = 1
                n_count = 0
                for i, j in enumerate(index_list):
                    n_flag = False
                    for n_idx in nouns_idx:
                        if j == n_idx: 
                            a = ['n',i+1, n_count]
                            n_list.append(a)
                            n_flag = True
                            n_count = n_count + 1
                            break
                    if n_flag == False:
                        a = ['w', i+1, w_count]
                        n_list.append(a)
                        w_count = w_count + 1

                for i in range(len(n_list)-1, 76):
                    a = ['w', i + 1, w_count]
                    n_list.append(a)
                    w_count = w_count + 1

                w_g_list = []
                n_g_list = []
                for i in range(0, 77):
                    if n_list[i][0] == "w":
                        w_g_list.append(torch.full(size=(768,), fill_value=tmp_z_attention[i]).unsqueeze(dim=0))
                    elif n_list[i][0] == "n":
                        n_g_list.append(torch.full(size=(768,), fill_value=tmp_z_attention[i]).unsqueeze(dim=0))

            w_g_list = nn.Parameter(torch.stack(w_g_list, dim=0))
            n_g_list = nn.Parameter(torch.stack(n_g_list, dim=0))
            w_g_list.requires_grad = True
            n_g_list.requires_grad = True

            optimizer = optim.Adam([
                    {'params': w_g_list},
                    {'params': n_g_list, 'lr': 0.01}
                ], lr=0.01)

            base_count = len(os.listdir(sample_path))
            optimization_step = 10
            with model.ema_scope():
                for e in range (optimization_step):
                    torch.manual_seed(r)
                    torch.cuda.manual_seed_all(r)
                    optimizer.zero_grad()
                    uc = None
                    if scale != 1.0:
                        uc, _, _ = model.get_learned_conditioning(batch_size * [""], None)

                    z, _, _ = model.get_learned_conditioning([caption_], [attention_list[idx_caption]])
                    init_z = z.detach().clone()

                    w = w_g_list[0]
                    for i in range(1, 77):
                        if n_list[i][0] == "w":
                            w = torch.cat((w, w_g_list[n_list[i][2]]), dim=0)
                        else:
                            w = torch.cat((w, n_g_list[n_list[i][2]]), dim=0)
                    w = w.unsqueeze(dim=0).to(device)
                    w.retain_grad()

                    original_mean = z.mean()

                    new_z = w * z
                    new_mean = new_z.mean()
                    new_z *= original_mean / new_mean

                    shape = [latent_channels, image_height // downsampling_factor, image_width// downsampling_factor]
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                        conditioning=new_z,
                                                        batch_size=n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    
                    # for save image
                    with torch.no_grad():
                        if e == 2:
                            pass
                        x_samples_ddim_for_img = x_samples_ddim.clone().detach()
                        for x_sample in x_samples_ddim_for_img:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')  # image로 저장하기 위해서
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}{audio_file_names[idx_caption]}{e}.png"))
                            base_count = base_count + 1

                    x_samples_ddim = resize(x_samples_ddim)
                    x_samples_ddim = normalize(x_samples_ddim)

                    ((_, image_features, _), _), _ = aclp(image=x_samples_ddim)

                    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
                    image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

                    scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
                    logits_audio_image = scale_audio_image * audio_features @ image_features.T
                    audio_clip_loss = 1 - 0.01*logits_audio_image.mean()

                    z_distance_loss = ((init_z - new_z) ** 2).sum()
                    z_image = perceptor.encode_image(x_samples_ddim).float()
                    clip_loss = clip_score_loss(z_image)

                    loss = 0.01*z_distance_loss + clip_loss + 0.9*audio_clip_loss
                    loss.backward()
                    optimizer.step()
        except:
            print("error", idx_caption, audio_path)
            error_list.append([idx_caption, audio_path])

        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
            f" \nEnjoy.")
        print(error_list)


if __name__ == "__main__":
    main()
