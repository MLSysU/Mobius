import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.rand(4096, 4096, 64, dtype=torch.float32,device=device)
x.to("cpu")
x.to("cuda")
