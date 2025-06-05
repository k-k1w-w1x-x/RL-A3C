# BIT 2024 冬季《强化学习》课程期末作业

**查看完整的演示报告：**  
[🌐 完整报告（使用Canva制作）](https://www.canva.cn/design/DAGWc1_iG7c/PMc5c1S34j9NeiM1cSSy0Q/edit?utm_content=DAGWc1_iG7c&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
------------------------
本仓库包含北京理工大学2024年冬季学期《强化学习》课程期末作业的代码与实验实现，内容概述如下：

- **算法框架**：基于 [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c) 仓库进行实现。代码包括了报告中提到的多种对 A3C（Asynchronous Advantage Actor-Critic）算法进行改进的策略，具体内容可见完整报告。
- **主要的贡献**：
  - 通过实验和理论分析探讨了实践中A3C算法效果不好的原因；
  - 为每个 worker 随机分配学习率，使得并行更新具有多样化步长；
  - 设计 **历史权重分配器** ，根据各并行 worker 的历史回报动态调整梯度贡献；
  - 将worker的选取抽象成bandit问题，设计 **置信上界权重分配器** ，根据各并行 worker 的置信上界动态调整梯度贡献；
  - 使用 **KL 散度约束** 控制策略更新步幅，减少策略震荡；
  - 在Atari环境的多个task上测试了三个改进策略及其组合对A3C的改进效果，并与vanilla A3C进行了比较。实验说明我们的改进策略是有效的。






