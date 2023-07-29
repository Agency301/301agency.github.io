---
layout: post
title:  "data2vec, A General Framework for Self-supervised Learning in Speech Vision and Language"
date:   2023-07-30 04:51:56 +0900
categories: Multi-modal
use_math: true
---

# data2vec, A General Framework for Self-supervised Learning in Speech Vision and Language



## Paper Review

# data2vec, A General Framework for Self-supervised Learning in Speech Vision and Language

## [data2vec: A General Framework for Self-supervised Learning in...](https://arxiv.org/abs/2202.03555)

## Introduction

현재의 self-supervised learning의 한계는 한 modality에서 만든 알고리즘을 다른 modality에 적용하기 어렵다는 것이다. data2vec은 generalized representation learning의 일종으로, model-specific한 token, words 등을 예측하는 대신 전체 input space에서의 masked prediction을 통해 general latent variable을 만들어내는 것을 목표로 한다.

## Prerequisites
- Bootstrap Your Own Latent(BYOL)

    [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

    data2vec의 model architecture는 상당 부분 BYOL의 것을 차용하고 있다.

    기존의 self-supervised learning에서는 positive sample만 사용할 경우 representation collapse가 발생하여 train loss는 감소하나 모두 같은 representation을 가지게 되는 문제점이 있어 negative sample을 이용한 contrastive learning이 필수적이었는데, BYOL은 negative sample 없이도 representation collapse를 발생시키지 않는 representation learning에 관한 내용이다.

    ![Untitled](https://agency301.github.io/assets/img/data2vec,-A-General-Framework-for-Self-supervised-Learning-in-Speech-Vision-and-Language/Untitled.png)

    training data $x$는 변환 $t, t'$를 통해 augmentation 되어 $v, v'$이 되고, 각각 online network와 target network에서 representation을 형성하게 된다. 각각의 층은 동일한 구조의 model을 사용하며, online network에서만 prediction network를 따로 둔다.

    학습 방법은 online network와 target network의 output 사이의 L2 loss를 이용해서 backprop을 하는 것인데, target network에는 stop gradient(sg)를 적용해서 backprop을 하지 않는다. 대신 Exponentially Moving Average라는 방식을 이용해서 가중치를 업데이트 한다.

    ![Untitled](https://agency301.github.io/assets/img/data2vec,-A-General-Framework-for-Self-supervised-Learning-in-Speech-Vision-and-Language/Untitled%201.png)

    2번째 식이 $\xi$(target network parameter)가 업데이트 되는 식이다. $\tau$ 는 0.996에서 시작하는 값으로, cosine annealing을 통해 1에 가까운 값이 되도록 time step에 따라 scheduling된다. 이 의미는 update 과정에서 초반에 $\theta$(online parameter)의 비중을 높게 가져가다가 나중에는 decay시키는 의미이다.


## Model Architecture

![Untitled](https://agency301.github.io/assets/img/data2vec,-A-General-Framework-for-Self-supervised-Learning-in-Speech-Vision-and-Language/Untitled%202.png)

1. 같은 구조의 2가지 모델 - teacher model, student model을 생성한다.
2. train data의 sample을 teacher model의 input으로, masked version을 student의 model의 input으로 사용한다.
3. 각각의 model은 multi-layer model인데, (본 논문에서는 transformer encoder를 전제하였으나, 다른 모델도 사용 가능하다고 밝힘) student model은 teacher model의 top-K layer representation의 average를 이용하여 predicton함.
    1. $y_t=\frac{1}{K}\Sigma^{L}_{l=L-K+1}\hat{a}^l_t$
    2. top layer의 representation만을 학습하는 것보다 top-K 개의 layer에 대해서 학습하는 것이 성능이 좋았는데, average를 이용하면 효율성이 증대되었다고 함.
4. teacher model의 parameter $\Delta$는 student model parameter $\theta$ 에 대해 EMA를 통해 update됨. (Knowledge Distillation)

    $\tau$는 linearly increase한다.

    ![Untitled](https://agency301.github.io/assets/img/data2vec,-A-General-Framework-for-Self-supervised-Learning-in-Speech-Vision-and-Language/Untitled%203.png)

5. Loss

    ![Untitled](https://agency301.github.io/assets/img/data2vec,-A-General-Framework-for-Self-supervised-Learning-in-Speech-Vision-and-Language/Untitled%204.png)

    $\beta$보다 작으면 L2 loss, 크면 L1 loss를 사용한다.


*Backbone 모델:

Visual data: Vit, Text data: RoBERTa, Speech data: fairseq(wav2vec)