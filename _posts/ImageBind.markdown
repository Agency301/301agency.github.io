
# [ImageBind: One Embedding Space To Bind Them All](https://arxiv.org/abs/2305.05665)

# Introduction
## ImageBind는 image, text, audio, depth, thermal, IMU data의 6가지 modalities에 대하여 joint embedding을 만들기 위한 시도다.
## 독특한 점은 image alignment만을 사용하여 joint embedding space를 training할 수 있다는 점을 제안한 것.
## 모든 조합의 paired data를 모두 이용할 필요 없이 image-paired data만을 이용하는 것으로 충분하다는 것을 밝혔다.

# <aside>
# 💡 Imagebind는 $(I, M)$의 modality pair를 이용해 training을 한다. (I=image, M=other modality) 이때 audio, depth, thermal, IMU (inertial measurement unit)를 각각 image와 pairing한다.

# </aside>

# ![Untitled](assets\img\ImageBind/Untitled.png)

# Natural Alignment: $(I, M)$간의 alignment → train
## encoders

I, M이 주어졌을 때 이들을 deep network를 이용해 normalized embedding으로 encode한다. Implementation & Training Details에도 나오겠지만, 이때 encoder (f, g)는 모두 Transformer.

$q_i=f(I_i)$, $k_i=g(M_i)$

## loss: InfoNCE loss

정확히는 $L_{I, M}+L_{M,I}$의 symmetric loss 사용함.

![Untitled](assets\img\ImageBind/Untitled%201.png)

τ: temparature. softmax distribution의 smoothness 결정

j: unrelated observation. 즉 mini-batch 내에서 negative pairs 나타냄

# Emergent Alignment: $(M_1, M_2)$ 간의 alignment → evaluate

## Imagebind는 서로 다른 modality pair $(M_1, M_2)$ 의 alignment를 직접 학습하지 않고 $(I, M_1)$, $(I, M_2)$ 를 각각 학습하는데, 그렇게 해도 $(M_1, M_2)$ 을 같은 embedding space에 잘 align할 수 있는지를 평가한다.

## emergent zero-shot은 multimodal model들의 emergent ability를 측정할 수 있게 하는 새로운 benchmark.
# **Emergent** zero-shot classification
## modality pair를 직접 training하지 않고 각각을 image와 aligning하여 training한 후에 modality pair 간의 aligning을 평가하는 것을 ***emergent*** zero-shot classification이라고 한다.
## result of Imagebind
- “fair” baseline은 없지만, 기존의 audio-text aligning model을 사용하거나, depth, thermal 같은 visual-like modality의 경우 그냥 image로 보고 CLIP을 사용하고 이를 baseline삼음

![Untitled](assets\img\ImageBind/Untitled%202.png)

- Text paired가 존재하는 text와 pairing하여 훈련시킨 baseline. SOTA는 각 dataset에서 additional supervision이나 model ensemble을 써서 얻어낸 값으로, 함께 명시함.
# Implementation & Training Details

## <aside>
## 💡 볼드체 정도만 봐도 될듯~

## </aside>

## **Encoders (all Transformer)**
- text encoder: CLIP text encoder
- image, video → same ViT (video as 2 frame image)
- audio → ‘AST: Audio Spectrogram Transformer’를 따라 audio를 encoding함
- thermal, depth → one-channel image로 취급, ViT
- IMU → 역시 Transformer
## Datasets
- (video, audio): Audioset
- (image, depth): SUN RGB-D
- (video, IMU): Ego4D
- (image, text)의 경우, pretrained vision (ViT-H 630M params) and text encoders (302M params) from OpenCLIP을 사용했으므로, image-text supervision from large-scale web data인 셈.
## Training
- **image, text encoder는 pretrained CLIP으로 initialize하고 freeze함.**
- **audio, depth, thermal, IMU encoder만 update했는데, 각각을 따로 훈련시킴.**
- **각 encoder마다 modality-specific linear projection을 하여 같은 size d를 가지는 embedding 만들어냄.**


# <aside>
# 💡 Experiments → 굳이 궁금하다면 볼 것

# </aside>

# Comparison to prior works
## Zero-shot text to audio retrieval and classification

![Untitled](assets\img\ImageBind/Untitled%203.png)

- 더 높거나 비견하는 결과
## Text to audio and video retrieval

![Untitled](assets\img\ImageBind/Untitled%204.png)

- text로부터 audio, video를 함께 retrieval하는 task에서 performance가 잘 나온 것은 pretrained OpenCLIP encoder들을 사용했기 때문으로 보임.
- text로부터 audio만을 retrieval하는 task의 경우는 주목할만함. (emergent)
## Few-shot classification

![Untitled](assets\img\ImageBind/Untitled%205.png)

- 왼쪽 → self-supervised baseline인 AudioMAE보단 좋은 성능을 기록했으며, supervised model도 4shot까지는 넘어섬
- 오른쪽 → MultiMAE를 완전히 넘어섬

# Analysis and Application
## multimodal embedding space arithmetic

![Untitled](assets\img\ImageBind/Untitled%206.png)

## upgrading text-based detectors to audio-based

→ object detection with audio queries

![Untitled](assets\img\ImageBind/Untitled%207.png)

## upgrading text-based diffusion models to audio-based

pretrained DALLE-2 diffusion model에서 prompt embedding을 audio embedding으로 바꾸어 실험함

![Untitled](assets\img\ImageBind/Untitled%208.png)


# CufftY (Yoonah Park)
## BIO
----------
Undergraduate Student majoring Computer Science & Engineering, Interested in Cognitive Architecture, Cellular Automata, and other DL, ML branches of study.

## Organization
----------
Seoul National University, Dept. of Computer Science & Engineering

AttentionX

## Contact
----------
[E-mail](wisdomsword21@snu.ac.kr)

[GitHub](https://github.com/gyuuuna)