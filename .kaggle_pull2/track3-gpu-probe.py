import torch
print("device_name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("arch_list:", torch.cuda.get_arch_list())
print("device_count:", torch.cuda.device_count())
print("mem_total_GB:", round(torch.cuda.get_device_properties(0).total_memory/1e9, 2) if torch.cuda.is_available() else None)
