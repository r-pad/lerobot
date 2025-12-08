import torch
import einops
# import torch_cluster
'''import torch_cluster'''
'''def run_fps(context_features, context_pos, fps_subsampling_factor):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape

        # Sample points with FPS
        sampled_inds = dgl_geo.farthest_point_sampler(
            einops.rearrange(
                context_features,
                "npts b c -> b npts c"
            ).to(torch.float64),
            max(npts // fps_subsampling_factor, 1), 0
        ).long()

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expandedtorch_cluster_sampled_inds
        )
        return sampled_context_features, sampled_context_pos'''


def run_fps(context_features, context_pos, fps_subsampling_factor=10):
        # context_features (Np, B, F)
        # context_pos (B, Np, F, 2)
        # outputs of analogous shape, with smaller Np
        npts, bs, ch = context_features.shape
        # import time
        # start = time.time()

        # Sample points with FPS
        # sampled_inds = dgl_geo.farthest_point_sampler(
        #     einops.rearrange(
        #         context_features,
        #         "npts b c -> b npts c"
        #     ).to(torch.float64),
        #     max(npts // self.fps_subsampling_factor, 1), 0
        # ).long()
        # print("npts, bs, ch", npts, bs, ch)


        # temp_context_features = einops.rearrange(
        #     context_features,
        #     "npts b c -> b npts c"
        # ).to(torch.float64)
        # num_points = max(npts // self.fps_subsampling_factor, 1)
        # h = min(9, np.log2(num_points))
        # sampled_indx = [fpsample.bucket_fps_kdline_sampling(temp_context_features[i].detach().cpu().numpy(), num_points, h=h, start_idx=0) for i in range(bs)]
        # sampled_inds = torch.stack(sampled_indx, dim=0).cuda()
        # print("sampled_inds", sampled_inds.shape)

        # import torch_cluster
        temp_context_features = einops.rearrange(
            context_features,
            "npts b c -> b npts c"
        ).to(torch.float64)
        num_points = max(npts // fps_subsampling_factor, 1)
        ratio = num_points / npts
        ratio = min(ratio, 1.0)
        # sampled_indx = [torch_cluster.fps(temp_context_features[i], ratio=ratio, random_start=False) for i in range(bs)]
        # sampled_inds = torch.stack(sampled_indx, dim=0)
        # print("sampled_inds", sampled_inds.shape)
        temp_context_features = temp_context_features.reshape(-1, ch)
        batch = torch.repeat_interleave(torch.arange(bs), npts).reshape(-1, bs).T.reshape(-1).cuda()
        sampled_indx = torch_cluster.fps(temp_context_features, batch, ratio=ratio, random_start=False)
        sampled_inds = sampled_indx.reshape(bs, -1)
        sampled_inds = sampled_inds - sampled_inds[:, 0].unsqueeze(1)

        # from pytorch3d.ops import sample_farthest_points
        # temp_context_features = einops.rearrange(
        #     context_features,
        #     "npts b c -> b npts c"
        # ).to(torch.float64)
        # num_points = max(npts // self.fps_subsampling_factor, 1)
        # # from termcolor import cprint
        # # cprint(f"temp_context_features {temp_context_features}", "blue")
        # sampled_value, sampled_inds = sample_farthest_points(points=temp_context_features.detach().cpu(), K=num_points, random_start_point=False)
        # sampled_inds = sampled_inds.cuda()
        # # print("sampled_inds", sampled_inds.shape) 

        # end = time.time()
        # cprint(f"fps time:  {end-start}", "red")  
        

        # Sample features
        expanded_sampled_inds = sampled_inds.unsqueeze(-1).expand(-1, -1, ch)
        sampled_context_features = torch.gather(
            context_features,
            0,
            einops.rearrange(expanded_sampled_inds, "b npts c -> npts b c")
        )

        # Sample positional embeddings
        _, _, ch, npos = context_pos.shape
        expanded_sampled_inds = (
            sampled_inds.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, ch, npos)
        )
        sampled_context_pos = torch.gather(
            context_pos, 1, expanded_sampled_inds
        )
        return sampled_context_features, sampled_context_pos