import torch

ckpt_path = '../results/taichi128/052-MatLatte-M-2-256-2-F16S3-taichi128-no_load_latent/checkpoints/1000000.pt'

ckpt = torch.load(ckpt_path, map_location='cpu')['model']

for k in list(ckpt.keys()):
    if '.u' in k and not 'linear_v' in k and not 'proj_v' in k:
        print('\n', k)
        # do the abs -> l1 norm for full tensor
        print(ckpt[k].shape)
        u_tensor = ckpt[k].abs()
        u_tensor = (u_tensor / u_tensor.sum(dim=0, keepdim=True)).reshape(8,8,2)
        print(u_tensor[:,:,0].sum())
        print(u_tensor[:,:,1].sum())
        print('---')