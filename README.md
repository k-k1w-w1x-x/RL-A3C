# BIT 2024 冬季《强化学习》课程期末作业
------------------------
**查看完整的演示报告：**  
[🌐 完整报告（使用Canva制作）](https://www.canva.cn/design/DAGWc1_iG7c/PMc5c1S34j9NeiM1cSSy0Q/edit?utm_content=DAGWc1_iG7c&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)
------------------------
本仓库包含北京理工大学2024年冬季学期《强化学习》课程期末作业的代码与实验实现，内容概述如下：

- **算法框架**：基于 [pytorch-a3c](https://github.com/ikostrikov/pytorch-a3c) 仓库，考虑了多种重写并扩展了 A3C（Asynchronous Advantage Actor-Critic）算法。
- **主要的贡献**：
  - 通过实验和理论分析探讨了实践中A3C算法效果不好的原因；
  - 为每个 worker 随机分配学习率，使得并行更新具有多样化步长；
  - 基于naive的想法，设计 **历史权重分配器** ，根据各并行 worker 的历史回报动态调整梯度贡献；
  - 将worker的选取抽象成bandit问题，设计 **置信上界权重分配器** ，根据各并行 worker 的置信上界动态调整梯度贡献；
  - 使用 **KL 散度约束** 控制策略更新步幅，减少策略震荡；
  - 尝试在策略/价值网络中加入 **注意力机制（Attention）**，提高高维状态下的表示能力。






