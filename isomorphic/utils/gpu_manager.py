"""
GPU Manager - Multi-GPU Allocation & Distribution

Handles the placement of LLMs across multiple GPUs (e.g. 4x A100) to maximize 
parallelism and manage VRAM efficiently.
"""

import torch
import os
from typing import List, Dict, Optional

class GPUManager:
    """
    Manages distribution of model workloads across multiple GPUs.
    """
    
    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            self.num_gpus = torch.cuda.device_count()
        else:
            self.num_gpus = num_gpus
            
        print(f"GPUManager initialized with {self.num_gpus} GPUs.")
        self.device_stats = {i: {"allocated_models": 0, "active": False} for i in range(self.num_gpus)}

    def get_optimal_device(self, model_name: str) -> str:
        """
        Determines the best GPU for a model by scanning actual free VRAM.
        Crucial for shared servers where some GPUs might be occupied by other jobs.
        """
        if self.num_gpus == 0:
            return "cpu"
            
        best_gpu = 0
        max_free_vram = -1
        
        for i in range(self.num_gpus):
            try:
                # mem_get_info returns (free_bytes, total_bytes)
                free_bytes, _ = torch.cuda.mem_get_info(i)
                if free_bytes > max_free_vram:
                    max_free_vram = free_bytes
                    best_gpu = i
            except Exception:
                pass
        
        self.device_stats[best_gpu]["allocated_models"] += 1
        print(f"  [GPUManager] Routed {model_name} to cuda:{best_gpu} ({max_free_vram / (1024**3):.2f}GB free)")
        return f"cuda:{best_gpu}"

    def get_auto_device_map(self, model_id: str) -> Dict[str, int]:
        """
        Calculates a device map for 'accelerate' to split large models across GPUs.
        Only used if a single model is too large for one GPU.
        """
        # For A100 80GB, most models (up to 35B in 4-bit) fit on ONE GPU.
        # We only return a complex map if explicitly needed.
        return "auto"

    def print_gpu_status(self):
        """Monitor VRAM across all cards."""
        print("\n--- GPU Utilization Status ---")
        for i in range(self.num_gpus):
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            used = torch.cuda.memory_allocated(i) / (1024**3)
            print(f"GPU {i}: {used:.2f}GB / {mem:.2f}GB (Models: {self.device_stats[i]['allocated_models']})")
        print("------------------------------\n")
