import torch
import numpy as np

class UCBWeightAllocator:
    def __init__(self, num_processes, c=0.5):
        self.num_processes = num_processes
        self.c = c  # UCB探索参数
        
        # 使用共享内存的张量
        self.weights = torch.ones(num_processes, dtype=torch.float32).share_memory_() / num_processes
        
        # UCB统计数据
        self.total_counts = torch.zeros(num_processes, dtype=torch.float32).share_memory_()
        self.value_estimates = torch.zeros(num_processes, dtype=torch.float32).share_memory_()
            
    def update_performance(self, rank, reward):
        """更新进程的UCB统计"""
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

class HistoryWeightAllocator:
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

    def get_weight(self, rank):
        return self.weights[rank].item()

# 工厂函数
def create_weight_allocator(allocator_type, num_processes, **kwargs):
    if allocator_type == 'ucb':
        return UCBWeightAllocator(num_processes, **kwargs)
    elif allocator_type == 'history':
        return HistoryWeightAllocator(num_processes)
    else:
        raise ValueError(f"Unknown allocator type: {allocator_type}")