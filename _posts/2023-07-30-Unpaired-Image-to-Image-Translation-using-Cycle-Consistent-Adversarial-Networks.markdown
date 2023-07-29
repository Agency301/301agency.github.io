---
layout: post
title:  ""
date:   2023-07-30 03:16:00 +0900
categories: 
use_math: true
---


> 1.2. Debugging is too difficult
> 2.3. I want to sleep now

> 4.- Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

> 6.[Unpaired Image-to-Image Translation using Cycle-Consistent...](https://arxiv.org/abs/1703.10593)

> 8.- Introduction

> 10.기존의 pix2pix model은 image mapping에 있어 generalizable하다는 특성이 있으나, 반드시 paired dataset이 필요하다는 단점이 있었다.

> 12.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled.png)

> 14.실제로 많은 양의 paired dataset을 구축하기는 어렵기 때문에, cycleGAN은 unpaired dataset으로 image translation을 하는 방법을 제안한다. 

> 16.- Model Architecture
> 17.- structure

> 19.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%201.png)

> 21.cycleGAN의 구조이다. $X→Y$를 mapping하는 $G$, 그리고 $Y→X$를 mapping하는 $F$가 있고, 각각이 생성한 데이터를 평가하는 Discriminator $D_Y, D_X$가 있다. 이들은 기존의 GAN과 마찬가지로 Adversarial Loss를 이용하여 학습하는데, $X, Y$가 paired 되어있지 않으므로, $G, F$가 $D_Y, D_X$를 속이기 위한 데이터를 생성해 input과는 irrelevant한 데이터를 생성하는 문제인 mode collapse가 발생했다고 한다. 그래서 도입된 것이 cycleGAN의 cycle-consistancy loss이다.

> 23.- cycle-consistency loss

> 25.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%202.png)

> 27.cycle-consistency loss의 요점은 $X$를 $Y$에 mapping한 데이터를 다시 inverse mapping하였을 때 input data를 reconstruction 할 수 있게 loss를 구성하는 것이다. 즉, 이 과정에서 input의 content를 파괴하지 않는 mapping이 학습될 것이다.

> 29.- loss

> 31.cycleGAN의 loss term은 GAN의 Adversarial loss + cycle-consistency loss 이다

> 33.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%203.png)

> 35.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%204.png)

> 37.full loss term

> 39.![Untitled](https://agency301.github.io/assets/img/Unpaired-Image-to-Image-Translation-using-Cycle-Consistent-Adversarial-Networks/Untitled%205.png)

> 41.$\lambda$는 $L_{GAN}$과 $L_{cyc}$ 사이의 중요도를 반영하는 값이다.

> 43.- Code review

> 45.https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

> 47.대부분이 GAN과 유사하기 때문에 차이점이 있는 부분만 보겠다.

> 49.```python
> 50.def optimize_parameters(self):
> 51.self.forward()
> 52.self.set_requires_grad([self.netD_A, self.netD_B], False)
> 53.self.optimizer_G.zero_grad()  
> 54.self.backward_G()      
> 55.self.optimizer_G.step()    
> 56.self.set_requires_grad([self.netD_A, self.netD_B], True)
> 57.self.optimizer_D.zero_grad() 
> 58.self.backward_D_A()   
> 59.self.backward_D_B()      
> 60.self.optimizer_D.step()  
> 61.```

> 63.Generator와 Discriminator가 각각 2개 있기 때문에 optimizing하는 코드에 반영이 되어있다. `forward()`, `backward_G()` method는 각각 2개의 Generator model에 관해 작동한다.

> 65.```python
> 66.def backward_G(self):
> 67.lambda_idt = self.opt.lambda_identity
> 68.lambda_A = self.opt.lambda_A
> 69.lambda_B = self.opt.lambda_B
> 70.if lambda_idt > 0:
> 71.self.idt_A = self.netG_A(self.real_B)
> 72.self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
> 73.self.idt_B = self.netG_B(self.real_A)
> 74.self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
> 75.else:
> 76.self.loss_idt_A = 0
> 77.self.loss_idt_B = 0

> 79.self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
> 80.self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
> 81.self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
> 82.self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
> 83.self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
> 84.self.loss_G.backward()
> 85.```

> 87.`backward_G()` method에 cycle-consistency loss 가 반영되어 있다. identitiy loss는 generator의 mapping이 identity function에 가까워지도록 하는 loss term인데, target domain을 아는 경우에 사용하면 mapping consistency를 높여주는 역할을 한다.

> 89.cf) Identity loss는 ****[Unsupervised Cross-Domain Image Generation](https://arxiv.org/abs/1611.02200)** 에서 처음으로 사용되었다.