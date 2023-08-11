---
layout: post
title:  "Neural ODE를 이용한 연속적인 계층을 가진 신경망 모델링"
date:   2023-08-11 20:18:10 +0900
categories: Modeling
author: AtlasYang
tags: Modeling
comments: true
katex: true
use_math: true
---

<script src="https://cdnjs.cloudflare.com/ajax/libs/markdown-it/12.2.0/markdown-it.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/markdown-it-katex@2.0.4/dist/markdown-it-katex.js"></script>

1. toc
{:toc}


## Prerequisites

#### Euler Method

Euler method는 미분방정식이 주어질 때 함수의 함숫값을 구하는 한 방법이다.

이 논문에서는 hidden state의 연속적인 mapping을 위해서 상미분방정식을 사용하는데, 이 때 foward 와 backward 과정 모두에서 Euler method를 통해 식을 전개하게 된다.

![IMG_0068.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0068.jpeg)

## Paper Review

### Neural Ordinary Differential Equations (2019.12.14.)

paper link: 

[Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)

### Introduction

대부분의 인공지능 모델이 이산적인 시간에 대한 데이터를 다룬다. 심층 신경망 모델의 Layer 자체가 이산적인 형태이기도 하고, 데이터도 이산적인 형태의 것이기 때문이다.

이 논문에서는 hidden state가 변화하는 과정을 상미분방정식으로 모델링하여 연속적인 깊이를 가진 모델을 구성하였다. 계층 구조가 이산적인 계산 형태에서 탈피하여 연속적인 모델링이 가능해지고, 연속적인 잠재 함수를 모델링하는 것에 대한 표현력이 증가함을 밝혔다.

hidden state를 나타내는 함수의 derivative를 구하는 과정을 Ordinary Differential Equation(ODE)로 표현하였으며, 어떠한 ODE solver에 대해서도 Backpropagation을 수행할 수 있는 Black-box 모델을 통해 출력을 생성하게 된다.

#### Idea

residual network, recurrent network 등을 생각해보면 모델에 들어온 입력에 Transformation을 가하는데, 이는 시간 $t$에 대해 이산적인 sequence를 생성한다. 이를 hidden state $\mathbf{h}_t$와 model parameter $\theta_t$에 관한 식으로 나타내면 다음과 같다.

$\mathbf{h}_{t+1}=\mathbf{h}_t+f(\mathbf{h}_t, \theta_t)$

하지만 만약에 hidden state를 계속 추가해서, 연속적인 시간에 대해 $\mathbf{h}_t$를 구할 수 있다면 어떨까? 이를 위해 다음과 같은 ODE를 구성한다.

$\frac{d\mathbf{h}_t}{dt}=f(\mathbf{h}(t), t, \theta)$

이를 이용하면 어떠한 연속적인 시점 $t$에 대해서도 model output $\mathbf{h}$를 구할 수 있을 것이다.

### Continuous Backpropagation

Neural ODE 모델의 Backpropagation은 Euler method를 이용한 function approximation의 과정이다. 이 과정에서 gradient를 편리하게 계산할 수 있도록 Adjoint sensitivity method를 사용한다.

![IMG_0064.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0064.jpeg)

위 그림과 같이 hidden state가 $\mathbf{z}(t_0)$에서 $\mathbf{z}(t_N)$로 mapping되는 과정을 보자. ODE solver을 통해 최종적으로 도출해야 하는 것은 dynamic parameter $\theta$에 대한 Loss의 gradient인 $\frac{\partial{L}}{\partial{\theta}}$와 $\mathbf{z}(t_0)$보다 이전 계층으로의 gradient 전파를 위한 $\frac{\partial{L}}{\partial{\mathbf{z}(t_0)}}$이다.

Loss는 scalar function이고, $f$는 neural network로 parameterize 되어 있다는 것에 주의하라.

**(1). $L(\mathbf{z}(t_N))=L(\mathbf{z}(t_0)+\int_{t_0}^{t_N}f(\mathbf{z}(t),t,\theta)dt)$**

Derivation of (1)

![IMG_0055.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0055.jpeg)

이제 $\frac{\partial{L}}{\partial{\mathbf{z}(t_0)}}$와 $\frac{\partial{L}}{\partial{\theta}}$를 구해야 하는데, 이를 단번에 유도하기는 복잡하므로, adjoint $\mathbf{a}(t)=\frac{dL}{d\mathbf{z}(t)}$를 정의한다. adjoint $\mathbf{a}$는 어떤 시점 $t$에서의 상태 $\mathbf{z}$에 대한 Loss이다. adjoint는 다음과 같은 ODE를 통해 표현한다.

**(2). $\frac{d\mathbf{a}(t)}{dt}=-\mathbf{a}^T\frac{\partial{f(\mathbf{z}(t), t, \theta)}}{\partial{\mathbf{z}(t)}}$**

Derivation of (2)

![IMG_0056.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0056.jpeg)

![IMG_0057.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0057.jpeg)

위의 adjoint 식 (2)를 Euler method에 적용시키면, $\frac{\partial{L}}{\partial{\mathbf{z}(t_0)}}$를 구할 수 있다.

**(3). $\frac{\partial{L}}{\partial{\mathbf{z}(t_0)}}=\frac{\partial{L}}{\partial{\mathbf{z}(t_N)}}-\int^{t_0}_{t_N}\mathbf{a}(t)^T\frac{\partial{f(\mathbf{z}(t), t, \theta)}}{\partial{\mathbf{z}}(t)}dt$**

Derivation of (3)

![IMG_0065.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0065.jpeg)

마지막으로, 결국 optimize 해야 하는 것은 model parameter $\theta$이므로, $\frac{dL}{d\theta}$를 구한다.

**(4). $\frac{dL}{d\theta}$=$-\int_{t_N}^{t_0}$$\mathbf{a}(t)^T$$\frac{\partial{f(\mathbf{z}(t), t,\theta)}}{\partial{\theta}}dt$**

Derivation of (4)

![IMG_0062.jpeg](https://agency301.github.io/assets/img/NeuralODE/IMG_0062.jpeg)

이로써 완전한 $f$의 update와 Loss backpropagation을 위한 모든 gradient를 얻을 수 있게 되었다.

### Benefits of Neural ODE

#### Continuous time-series model을 만들 수 있다.

일정한 시간 간격을 가진 obeservation이 필요했던 기존의 모델과는 달리, arbitrary-time observation을 사용할 수 있게 되면서 데이터 손실을 최소화 할 수 있고, latent를 더 잘 모델링할 수 있게 된다.

#### 계산의 정확도를 쉽게 조절할 수 있다

오랜 기간 연구되어온 ODE solver인 Euler method를 사용하기 때문에 오차를 필요한 수준으로 감소시키는 작업이나, 계산을 가속화하는 것이 용이하다.

#### Continuous Normalizing Flow 모델 구현

Neural ODE를 이용한 Normalizing Flow 모델은 continuous해지며, 계산이 간편해진다. NF 모델에 관한 내용은 다음을 참고하라.

#### 메모리 효율성

scalar Loss를 사용하는 Neural ODE 모델은 중간 과정의 값을 저장할 필요가 없기 때문에, 모델의 깊이에 관계없이 일정한 메모리를 차지한다.

### Conclusion

Neural ODE는 기존의 이산적인 모델 구조에서 벗어나 연속적인 계층을 가진 Neural Network를 구현한 점에서, 해당 분야의 길을 열었다고 평가받고 있다. 

실제로 Neural ODE의 등장 이후로 Continuous Normalizing Flow 등의 생성 모델 연구가 이루어지고 있으며, Neural ODE로 모든 함수의 근사가 불가능하다는 것이 밝혀지며 Augmented Neural ODE 모델이 등장하기도 하였다.

연속적인 시간에 대한 sampling과 학습이 가능하고, 따라서 continuous function에 대한 latent가 더 높은 정확도로 생성되므로, 시간의 연속적인 정보가 데이터에서 중요한 기상, 경제 지표, 의료 등의 데이터 분야에서 큰 활약을 할 수 있을 것으로 보인다.

## Related Works

#### Augmented Neural ODE

[Augmented Neural ODEs](https://arxiv.org/abs/1904.01681)

#### Normalizing Flow

[Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770)