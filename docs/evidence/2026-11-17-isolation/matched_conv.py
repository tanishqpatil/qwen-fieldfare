import json,struct
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
root=Path.home()/'models/Qwen3.6-35B-A3B-4bit';name='language_model.model.layers.0.linear_attn.conv1d.weight'
wm=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
with (root/wm[name]).open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));info=h[name];f.seek(8+n+info['data_offsets'][0]);raw=f.read(info['data_offsets'][1]-info['data_offsets'][0])
bits=np.frombuffer(raw,dtype='<u2').astype(np.uint32)<<16;weight=mx.array(bits.view(np.float32).reshape(8192,4,1))
def stats(a,b):
 a=a.astype(np.float64).reshape(-1);b=b.astype(np.float64).reshape(-1);e=a-b
 return dict(max_abs=float(abs(e).max()),rms=float(np.sqrt(np.mean(e*e))),relative_L2=float(np.linalg.norm(e)/max(np.linalg.norm(b),1e-30)),reference_rms=float(np.sqrt(np.mean(b*b))))
results={}
for p in [0,1,2,20,40,60,80]:
 sw=Path('/tmp/qwen-isolation-swift-base')/str(p)
 prev=np.fromfile(sw/'prior_conv.f32',dtype='<f4').reshape(8192,4)[:,1:4].T[None]
 qkv=np.fromfile(sw/'qkv.f32',dtype='<f4').reshape(1,1,8192);inputs=mx.array(np.concatenate([prev,qkv],axis=1))
 output=nn.silu(mx.conv1d(inputs,weight,groups=8192));mx.eval(output)
 results[str(p)]=stats(np.array(output),np.fromfile(sw/'conv.f32',dtype='<f4'))
Path('/tmp/qwen-isolation-conv.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
