---
layout: post
title:  "Encoding-Reccurence-to-Attention-Mechanism2"
date:   2023-07-29 15:24:52 +0900
categories: Modeling
use_math: true
---



# Encoding Recurrence into Transformers

## [](https://openreview.net/pdf?id=7YfHla7IxBJ)

## Introduction

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled.png)

문제 상황: RNN은 구조상, recurrency 가 높은 데이터에 대해서는 학습 데이터가 적어도 효율이 나오는데, transformer는 그렇지 않다. 다만 transformer는 학습 데이터셋 크기가 크면 data의 recurrency에 관계없이 효율이 나온다. 이 점에 착안하여 RNN이 recurrence를 잘 capture하는 특징과 Transformer의 Attention 매커니즘을 결합하여 sequential data에 대한 학습 효율을 증가시키자는 것이 골자이다.

## Model Architecture

용어:

RSA: Self-Attention with Recurrence, 이 논문에서 propose한 것

REM: Recurrence Encoding Matrix, 여기에 rnn의 본질인 recurrence dynamics가 positional encoding의 형태로 담겨 있다.

수식이 많은데 요점은…

RNN은 원래 병렬 처리가 안된다. → 은닉층 가중치가 지수적으로 곱해지기 때문

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%201.png)

$W^{j}_{h}$ 이 부분이 문제임 → 하지만 Jordan Form으로 semi-diagonalize하면 손실 조금만 나면서 행렬 정리 가능

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%202.png)

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%203.png)

위처럼 block RNN 구조로 바꾼 다음에 Self-Attention의 form 으로 이것을 표현하기 위해 아래와 같은 masking 행렬을 정의한다.

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%204.png)

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%205.png)

그리고 이것을 $h^{C_1}_{t}, h^{C_2}_{t}$에 대해서도 똑같이 정의하고, Multi-head Self-Attention에 넣어주면 된다.

결국 RNN의 Recurrence 를 Capture하는 능력이 REM에 담기게 되지만, 계산 상 Attention과는 큰 관련이 없는데, Attention이 non-recurrent한 데이터에 대한 성능도 뛰어나니까 gated unit 하나로 REM과 conventional Self-Attention을 통합한 것이 RSA이다.

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%206.png)

$P$가 REM이다. $\sigma(\mu)$는 learnable gate value이고, attention과 REM 을 사용하는 것의 비중을 정해준다.

(data의 reccurence가 심한 경우 이 값이 크게 학습된다.)

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%207.png)

derivation은 밑 토글 참조.

암튼 RSA는 입력과 출력만 보았을 때 self-attention이랑 같기 때문에, 기존에 SA를 사용하던 Transformer 모델에 RSA를 대체로 넣어주기만 하면 되어서 편하다고 한다.

그리고 Introduction에 있는 그림처럼 RSA를 쓰면 그냥 SA보다 표현력이 증가해서 여러 sequential data learning tasks에서 좋은 결과를 얻었다고 한다…(대단해 O~O)

time series data를 이용한 비교이다. 본 논문에서 가장 recurrent한게 time series라고 밝혔듯이 확실히 개선된 결과를 보여준다. (prefix RSA가 붙은게 RSA를 대체로 쓴 모델이다)

![Untitled](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/Untitled%208.png)

이 외에도 벤치마크 몇 개 더 있는데 암튼 개선된 것 같다!

## Mathmatics

REM 식 derivation하는데 필요한 수학

선대 열심히 할걸 ㅜ

A. Jordan form

![IMG_0044.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0044.jpeg)

![IMG_0045.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0045.jpeg)

![IMG_0046.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0046.jpeg)

B. Applied Theorems

![IMG_0047.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0047.jpeg)

추가) 어떤 행렬이 eigenvalue로 어떤 complex를 가지면 그 conjugate도 eigenvalue이다

RNN 표현

![IMG_0049.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0049.jpeg)

![IMG_0050.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0050.jpeg)

![IMG_0052.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0052.jpeg)

![IMG_0054.jpeg](https://agency301.github.io/assets/img/Encoding-Reccurence-to-Attention-Mechanism2/IMG_0054.jpeg)


# CufftY (Yoonah Park)
## BIO
----------
Undergraduate Student majoring Computer Science & Engineering, Interested in Cognitive Architecture, Cellular Automata, and other DL, ML branches of study.
Very Dangerous Girl, She has so many boyfriends

## Organization
----------
Seoul National University, Dept. of Computer Science & Engineering

AttentionX

## Contact
----------
[E-mail](wisdomsword21@snu.ac.kr)

[GitHub](https://github.com/gyuuuna)