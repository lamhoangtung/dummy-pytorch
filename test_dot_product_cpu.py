import torch

dst_norm_trans_dst_pix = torch.load('dst_norm_trans_dst_pix.pt', map_location='cpu')
dst_pix_trans_src_pix = torch.load('dst_pix_trans_src_pix.pt', map_location='cpu')
src_pix_trans_src_norm = torch.load('src_pix_trans_src_norm.pt', map_location='cpu')
dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)

# Running on torch1.8.1-cu111-conda return the correct result, thus we save this a the reference result
# torch.save(dst_norm_trans_src_norm, 'dst_norm_trans_src_norm.pt')

reference_result = torch.load('dst_norm_trans_src_norm.pt', map_location='cpu')
print(dst_norm_trans_src_norm)
print(bool(torch.all(reference_result == dst_norm_trans_src_norm)))