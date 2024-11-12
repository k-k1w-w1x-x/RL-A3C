import torch
import numpy as np

class WeightAllocator:
    def __init__(self, num_processes, c=0.5):
        self.num_processes = num_processes
        self.c = c  # UCB探索参数
        
        # 使用共享内存的张量
        self.weights = torch.ones(num_processes, dtype=torch.float32).share_memory_() / num_processes
        self.performance_buffer = torch.zeros((num_processes, 100), dtype=torch.float32).share_memory_()
        self.history_lengths = torch.zeros(num_processes, dtype=torch.long).share_memory_()
        
        # 添加UCB所需的统计数据
        self.total_counts = torch.zeros(num_processes, dtype=torch.float32).share_memory_()
        self.value_estimates = torch.zeros(num_processes, dtype=torch.float32).share_memory_()
            
    def update_performance(self, rank, reward):
        """更新进程的表现历史和UCB统计"""
        # 更新性能缓冲区
        idx = self.history_lengths[rank] % 100
        self.performance_buffer[rank, idx] = reward
        self.history_lengths[rank] += 1
        
        # 更新UCB统计
        self.total_counts[rank] += 1
        n = self.total_counts[rank]
        value = self.value_estimates[rank]
        self.value_estimates[rank] = ((n - 1) / n) * value + (1 / n) * reward
        
    def update_weights(self):
        """使用UCB和softmax更新权重"""
        total_counts_sum = self.total_counts.sum()
        
        # 计算UCB值
        ucb_values = self.value_estimates + self.c * torch.sqrt(
            torch.log(total_counts_sum + 1) / (self.total_counts + 1e-5)
        )
        
        # 应用softmax转换
        self.weights.copy_(torch.softmax(ucb_values, dim=0))

    def get_weight(self, rank):
        """获取指定进程的权重"""
        return self.weights[rank].item()