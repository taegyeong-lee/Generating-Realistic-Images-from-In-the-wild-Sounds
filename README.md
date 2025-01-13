# **Generating Realistic Images from In the wild Sounds, ICCV 2023**

This repository is the official implementation of [**Generating Realistic Images from In the wild Sounds**](https://arxiv.org/abs/2309.02405).

**[Generating Realistic Images from In the wild Sounds](https://arxiv.org/abs/2309.02405)**
<br/>
[Taegyeong Lee](https://sites.google.com/view/taegyeonglee/home), 
Jeonghun Kang, 
Hyeonyu Kim, 
Taehwan Kim, 
<br/>

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://taegyeong-lee.github.io/lee2023generating)
[![arXiv](https://img.shields.io/badge/arXiv-2212.11565-b31b1b.svg)](https://arxiv.org/abs/2309.02405)

## Abstract
Representing wild sounds as images is an important but
challenging task due to the lack of paired datasets between
sound and images and the significant differences in the
characteristics of these two modalities. Previous studies
have focused on generating images from sound in limited
categories or music. In this paper, we propose a novel approach to generate images from in-the-wild sounds. First,
we convert sound into text using audio captioning. Second, we propose audio attention and sentence attention to
represent the rich characteristics of sound and visualize
the sound. Lastly, we propose a direct sound optimization with CLIPscore and AudioCLIP and generate images
with a diffusion-based model. In experiments, it shows that
our model is able to generate high quality images from wild
sounds and outperforms baselines in both quantitative and
qualitative evaluations on wild audio datasets

## News
- [01-10-2024] README.md
- [01-10-2024] We are in the process of refactoring for code deployment.
- [02-06-2024] We released pretrained models (ACT, Stable Diffusion, CLIP ..).
- [02-16-2024] We released code but, it's not yet complete.
- [08-28-2024] We released Multi-ESC50 dataset.

## Approach
![image](https://github.com/etilelab/Generating-Realistic-Images-from-In-the-wild-Sounds/assets/28443896/a9307826-ade6-48c5-a049-0f6d6ca41c78)

## Usage
 We are in the process of refactoring the code, and some parts of it have been improved, which may differ slightly from the paper.

### 1. Pretrained models download
   You can download pre-trained models from [here](https://drive.google.com/file/d/1Gh2bYrU-H47wnHVRhRs9YTGMUaYAVibr/view?usp=drive_link), such as the Audio Captioning Transformer, Audioclip, and so on.
   Also You can download Multi-ESC50, https://drive.google.com/file/d/1_dVHcIZ-13ubMgd09-mm8Zktdh6FX9iu/view?usp=drive_link

### 2. Hyperparameters setting
You need to modify the following hyperparameters in run.py and other config yaml files. Furthermore, various paths within the Python file and data preprocessing (h5) are required.

    outpath = 'iccv_2023'  # Output image path
    my_config = 'pre_models/configs/stable-diffusion/v1-inference.yaml' # Stable diffusion yaml
    ckpt = 'pre_models/stable_diffusion/sd-v1-4.ckpt' # Stable diffusion checkpoint path
    act_config = 'pre_models/configs/audio-transformer/settings_audioset.yaml'  # Audio Captioning Transformer yaml, you need to preprocess audio files
    audioclip_model_path = 'pre_models/audio_clip/AudioCLIP-Full-Training.pt' # AudioCLIP checkpoint
    audio_meta = 'audioset_test/' # Original audio meta file path, *.wav, *.mp3

### 3. Run run.py
If you've completed setting the paths to the files and configuring the hyperparameters, you can perform audio to image conversion using run.py.

![image](https://github.com/etilelab/Generating-Realistic-Images-from-In-the-wild-Sounds/assets/28443896/bc87c582-e58a-4afc-b81f-6c63329556ca)


## Citation
```
@inproceedings{lee2023generating,
      title={Generating Realistic Images from In-the-wild Sounds},
      author={Lee, Taegyeong and Kang, Jeonghun and Kim, Hyeonyu and Kim, Taehwan},
      booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
      pages={7160--7170},
      year={2023}
    }
```
