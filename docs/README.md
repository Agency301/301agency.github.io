---
layout: page
title: Documentation
description: >
  Here you should be able to find everything you need to know to accomplish the most common tasks when blogging with Hydejack.
hide_description: true
sitemap: false
permalink: /docs/
---

- LLaVA (2023.7.26)
    
    [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
    
    - Introduction
        - **LLaVA: Large Language and Vision Assistant**
        - nlp domain에서 주로 쓰여온 **instruction tuning**을 적용한 multimodal LLM임.
        - model adaption에서의 parameter-efficiency를 향상시키는 것이 future work.
    1. **Multimodal instruction-following data**: vision-language instruction-following data가 없다는 게 key challenge인데, data reformation perspective와 image-text pair를 appropriate instruction-following format으로 바꾸는 pipeline을 개발함
        - GPT-assisted Visual Instruction Data Generation
            1. COCO dataset에서 image caption, bounding box 두 가지 유형의 symbolic representation을 만들어 image context로 사용함. 
            2. ChatGPT 및 GPT-4를 이용해 image context로부터 question-answer pair들을 생성하도록 함.
            3. `Human : Xq Xv<STOP>\n Assistant : Xc<STOP>\n`의 형태로 instruction-following data pair들을 만들어 냄. (이는 conversation, detailed description, complex reasoning의 3 type을 가짐)
            
            ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b694fcda-0c11-4ddf-bdb3-be0f793a9309/Untitled.png)
            
    2. **Large multimodal models**: pretrained CLIP visual encoder와 language decoder LLaMA를 연결하고 end-to-end로 pretraining 및 fine-tuning을 하여 LLM(large multimodal model)을 구축함.
        - Visual Instruction Tuning
            - Architecture
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/30fc9374-559d-486d-ab64-3494c4836117/Untitled.png)
                
                - **Vision Encoder: CLIP visual encoder ViT-L/14**
                    - input image Xv는 pre-trained CLIP visual encoder ViT-L/14에서 encoding함. 이때 마지막에서 두 번째 layer output을 사용함.
                - **Projection Layer: linear layer**
                    - vision encoder output을 linear layer를 거쳐 word embedding space에 mapping함 (Hv)
                - **Language Model: LLaMA 사용.**
                    - 만들어낸 image feature와 language instruction을 함께 LLaMA로 넘겨 fine-tuning하며 visual instruction tuning함.
                - 이는 Q-former, gated cross-attention 같은 더 정교한 방법 대신 lightweight한 간단한 메커니즘을 사용한 것. 더 effective & sophisticated architecture는 future work.
            - Training
                - $X^t_{instruct}$는 $[X^1_q, X_v]$  또는 $[X_v, X^1_q]$  아래 그림상에서 초록색으로 표시된 부분을 generative loss 계산에 이용함
                    
                    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b2dce3dd-f733-474e-98ce-09e6d031ae00/Untitled.png)
                    
                    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9dc60623-933e-49ce-941a-4d3380657ef1/Untitled.png)
                    
                - two-stage instruction-tuning procedure
                    - **Stage 1: Pre-training for Feature Alignment**
                        - visual encoder, LLM weights는 frozen으로 두고 projection matrix만 update
                        - image feature Hv가 pre-trained LLM word embedding과 align이 되도록 해야 됨
                    - **Stage 2: Fine-tuning End-to-End**
                        - visual encoder를 frozen으로 두고 projection matrix와 LLM만 update
                        - Multimodal Chatbot, Science QA를 case scenario로 사용함
    - Experiments
        - Multimodal Chatbot
            - GPT-4, BLIP-2, OpenFlamingo와 비교함. LLaVA가 훨씬 작은 dataset으로 train되었음에도 GPT-4에 비견하는 결과를 냄. BLIP-2와 OpenFlamingo는 user instruction을 잘 따르기보다는 image를 describe하는 데 더 초점을 맞추었음.
            - Quantitative Evaluation
                - GPT-4가 helpfulness, relevance, accuracy, level of details를 평가하고, 그에 대한 이유를 제시하도록 함.
                    
                    ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/91250c71-f907-4a6f-a69e-d6793154193e/Untitled.png)
                    
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5339b340-ade0-4a58-a7e0-54f3e1e0ce59/Untitled.png)
                
        - ScienceQA
            - GPT-3.5(text-davinci-002) with and without CoT, LLaMA-Adapter with and without MM-CoT (current SOTA), GPT-4 using 2-shot과 비교함. LLaVa가 SOTA와 quite close하게 나옴.
            - GPT-4와 LLaVa로 각각 answer만든 후 두 answer를 다시 GPT-4에 external knowledge로 주입하여 돌렸더니 new SOTA 나옴.
            - example
                
                ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/6a3b648e-b6aa-4685-915a-8db6491d37b7/Untitled.png)
                
        - Ablations
            - visual encoder로 CLIP last layer output을 쓰거나 끝에서 두 번째 layer output을 쓸 때를 비교실험함. 끝에서 두 번째가 더 localized property를 보존하기 때문에 더 잘 나온다고 해석함.
            - Chain-of-thought의 contribution은 크지 않았음
            - pre-training
            - model size



Here you should be able to find everything you need to know to accomplish the most common tasks when blogging with Hydejack.
Should you think something is missing, [please let me know](mailto:mail@qwtel.com).
Should you discover a mistake in the docs (or a bug in general) feel free to [open an issue](https://github.com/hydecorp/hydejack/issues) on GitHub.

While this manual tries to be beginner-friendly, as a user of Jekyll it is assumed that you are comfortable running shell commands and editing text files.
{:.note}


## Getting started
* [Install]{:.heading.flip-title} --- How to install and run Hydejack.
* [Upgrade]{:.heading.flip-title} --- You can skip this if you haven't used Hydejack before.
* [Config]{:.heading.flip-title} --- Once Jekyll is running you can start editing your config file.
{:.related-posts.faded}

## Using Hydejack
* [Basics]{:.heading.flip-title} --- How to add different types of content.
* [Writing]{:.heading.flip-title} --- Producing markdown content for Hydejack.
* [Scripts]{:.heading.flip-title} --- How to include 3rd party scripts on your site.
* [Build]{:.heading.flip-title} --- How to build the static files for deployment.
* [Advanced]{:.heading.flip-title} --- Guides for more advanced tasks.
{:.related-posts.faded}

## Other
* [LICENSE]{:.heading.flip-title} --- The license of this project.
* [NOTICE]{:.heading.flip-title} --- Parts of this program are provided under separate licenses.
* [CHANGELOG]{:.heading.flip-title} --- Version history of Hydejack.
{:.related-posts.faded}

[install]: install.md
[upgrade]: upgrade.md
[config]: config.md
[basics]: basics.md
[writing]: writing.md
[scripts]: scripts.md
[build]: build.md
[advanced]: advanced.md
[LICENSE]: ../LICENSE.md
[NOTICE]: ../NOTICE.md
[CHANGELOG]: ../CHANGELOG.md
