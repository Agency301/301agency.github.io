---
layout: post
title:  ""
date:   2023-07-30 03:23:00 +0900
categories: 
use_math: true
---


> 1.- Tags: CV

## Related Works

> 2.1. I don’t know
> 3.2. Debugging is too difficult
> 4.3. I want to sleep now

## Paper Review

# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

## [Unpaired Image-to-Image Translation using Cycle-Consistent...](https://arxiv.org/abs/1703.10593)

## Introduction

기존의 pix2pix model은 image mapping에 있어 generalizable하다는 특성이 있으나, 반드시 paired dataset이 필요하다는 단점이 있었다.

![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled.png)

실제로 많은 양의 paired dataset을 구축하기는 어렵기 때문에, cycleGAN은 unpaired dataset으로 image translation을 하는 방법을 제안한다. 

## Model Architecture
- structure

    ![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%201.png)

    cycleGAN의 구조이다. $X→Y$를 mapping하는 $G$, 그리고 $Y→X$를 mapping하는 $F$가 있고, 각각이 생성한 데이터를 평가하는 Discriminator $D_Y, D_X$가 있다. 이들은 기존의 GAN과 마찬가지로 Adversarial Loss를 이용하여 학습하는데, $X, Y$가 paired 되어있지 않으므로, $G, F$가 $D_Y, D_X$를 속이기 위한 데이터를 생성해 input과는 irrelevant한 데이터를 생성하는 문제인 mode collapse가 발생했다고 한다. 그래서 도입된 것이 cycleGAN의 cycle-consistancy loss이다.

    - cycle-consistency loss

    ![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%202.png)

    cycle-consistency loss의 요점은 $X$를 $Y$에 mapping한 데이터를 다시 inverse mapping하였을 때 input data를 reconstruction 할 수 있게 loss를 구성하는 것이다. 즉, 이 과정에서 input의 content를 파괴하지 않는 mapping이 학습될 것이다.

- loss

    cycleGAN의 loss term은 GAN의 Adversarial loss + cycle-consistency loss 이다

    ![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%203.png)

    ![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%204.png)

    full loss term

    ![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%205.png)

    $\lambda$는 $L_{GAN}$과 $L_{cyc}$ 사이의 중요도를 반영하는 값이다.

## Code review

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

대부분이 GAN과 유사하기 때문에 차이점이 있는 부분만 보겠다.

```python
def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()  
        self.backward_G()      
        self.optimizer_G.step()    
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad() 
        self.backward_D_A()   
        self.backward_D_B()      
        self.optimizer_D.step()  
```

Generator와 Discriminator가 각각 2개 있기 때문에 optimizing하는 코드에 반영이 되어있다. `forward()`, `backward_G()` method는 각각 2개의 Generator model에 관해 작동한다.

```python
def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
```

`backward_G()` method에 cycle-consistency loss 가 반영되어 있다. identitiy loss는 generator의 mapping이 identity function에 가까워지도록 하는 loss term인데, target domain을 아는 경우에 사용하면 mapping consistency를 높여주는 역할을 한다.

cf) Identity loss는 ****[Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200)** 에서 처음으로 사용되었다.