"""Pure-camera residual; full temporal attention vs parameter-matched diagonal control.

The residual acts on patchified camera features before the existing per-block
projections. It never receives image/video features. This is an experimental
PlenopticDreamer extension, not CameraCtrl's U-Net encoder implementation.
"""
import math

import torch
from torch import nn


class CameraGeometryResidual(nn.Module):
    def __init__(self, temporal, frames=21, dimension=1536, width=64, heads=4):
        super().__init__()
        self.temporal=temporal;self.frames=frames
        self.down=nn.Linear(dimension,width)
        self.norm=nn.LayerNorm(width)
        self.attention=nn.MultiheadAttention(width,heads,dropout=0,batch_first=True)
        self.out=nn.Linear(width,dimension,bias=False)
        nn.init.zeros_(self.out.weight)
        position=torch.arange(frames,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
        encoding=torch.zeros(frames,width);encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency)
        self.register_buffer('position',encoding,persistent=False)
        self.register_buffer('diagonal_mask',~torch.eye(frames,dtype=torch.bool),persistent=False)

    def forward(self,camera):
        batch,total,height,width,dimension=camera.shape
        if total%self.frames:raise ValueError('Camera frames must split into complete views')
        views=total//self.frames
        tokens=camera.reshape(batch,views,self.frames,height,width,dimension).permute(0,1,3,4,2,5)
        tokens=tokens.reshape(batch*views*height*width,self.frames,dimension)
        hidden=self.norm(self.down(tokens))+.1*self.position.to(tokens.dtype)
        attended,_=self.attention(hidden,hidden,hidden,
                     attn_mask=None if self.temporal else self.diagonal_mask,need_weights=False)
        residual=self.out(attended)
        residual=residual.reshape(batch,views,height,width,self.frames,dimension).permute(0,1,4,2,3,5)
        return camera+residual.reshape_as(camera).to(camera.dtype)


def attach(model,temporal):
    assert not hasattr(model,'cam_encoder_adapter')
    model.add_module('cam_encoder_adapter',CameraGeometryResidual(temporal).to(device=next(model.parameters()).device))
    def before_forward(module,args,kwargs):
        kwargs=dict(kwargs)
        kwargs['camera']=module.cam_encoder_adapter(kwargs['camera'])
        return args,kwargs
    model.register_forward_pre_hook(before_forward,with_kwargs=True)
    return model


def verify():
    torch.manual_seed(17)
    sequential=CameraGeometryResidual(True,frames=5,dimension=12,width=8,heads=2)
    independent=CameraGeometryResidual(False,frames=5,dimension=12,width=8,heads=2)
    independent.load_state_dict(sequential.state_dict())
    camera=torch.randn(1,15,2,2,12)
    assert torch.equal(sequential(camera),camera) and torch.equal(independent(camera),camera)
    sequential(camera).square().mean().backward()
    assert sequential.out.weight.grad.abs().max()>0
    # Turn on a tiny trained residual to check temporal reach and view isolation.
    with torch.no_grad():sequential.out.weight.normal_(std=.01)
    independent.load_state_dict(sequential.state_dict())
    changed=camera.clone();changed[:,2,0,0]+=torch.linspace(-1,1,12)
    full=(sequential(changed)-sequential(camera)).detach()
    diagonal=(independent(changed)-independent(camera)).detach()
    assert torch.count_nonzero(full[:,5:])==0 and torch.count_nonzero(diagonal[:,5:])==0
    assert full[:,0,0,0].abs().max()>1e-7
    assert torch.count_nonzero(diagonal[:,0])==0
    assert torch.count_nonzero(full[:,:,1,1])==0
    assert sum(p.numel() for p in sequential.parameters())==sum(p.numel() for p in independent.parameters())
    return dict(zero_initial_output_preserves_camera=True,output_projection_receives_gradient=True,
                temporal_cross_frame_effect=True,diagonal_control_frame_independent=True,
                no_cross_view_or_cross_pixel_mixing=True,parameter_counts_match=True)


if __name__=='__main__':print(verify())
