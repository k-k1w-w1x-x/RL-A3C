import torch
import numpy as np

class WeightAllocator:
    def __init__(self, num_processes):
        self.num_processes = num_processes
        # 使用共享内存的张量
        self.weights = torch.ones(num_processes, dtype=torch.float32).share_memory_() / num_processes
        # 性能历史也使用共享内存
        self.performance_buffer = torch.zeros((num_processes, 100), dtype=torch.float32).share_memory_()
        self.history_lengths = torch.zeros(num_processes, dtype=torch.long).share_memory_()
            
    def update_performance(self, rank, reward):
        """更新进程的表现历史"""
        idx = self.history_lengths[rank] % 100
        self.performance_buffer[rank, idx] = reward
        self.history_lengths[rank] += 1
            
    def get_weight(self, rank):
        """获取某个进程的权重"""
        return self.weights[rank].item()
        
    def update_weights(self):
        """更新所有进程的权重"""
        avg_performances = []
        for rank in range(self.num_processes):
            length = min(self.history_lengths[rank].item(), 100)
            if length == 0:
                avg_performances.append(0)
            else:
                # 只计算有效历史数据的平均值
                valid_history = self.performance_buffer[rank, :length]
                avg_performances.append(valid_history.mean().item())
        
        # 使用softmax计算新的权重
        performances = torch.tensor(avg_performances)
        self.weights.copy_(torch.softmax(performances / 0.5, dim=0))