# Diagnostic orchestration follows the installed Apple mlx-lm qwen3_5 path;
# no upstream model implementation is modified.
import json,sys,hashlib,math
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
reg=AutoTokenizer.register
AutoTokenizer.register=lambda config,*a,**k: None if isinstance(config,str) else reg(config,*a,**k)
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.models import gated_delta as gd
from mlx_lm.models.cache import ArraysCache
root=Path.cwd();folder=Path('/tmp/qwen-isolation-mlx2');folder.mkdir(exist_ok=True)
f=json.loads(Path('docs/evidence/2026-11-17-parity/reference.json').read_text());tokens=f['prompt_tokens']+f['reference_tokens'][:60]
model,tok=load(str(Path.home()/'models/Qwen3.6-35B-A3B-4bit'));l=model.language_model.model.layers[0];ln=l.linear_attn
positions=[0,1,2,20,40,60,80]; dtypes={};observations={}
def evaluate(x):mx.eval(x);return np.array(x.astype(mx.float32))
def stats(a,b):
 a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);e=a-b
 return dict(max_abs=float(abs(e).max()),rms=float(np.sqrt(np.mean(e*e))),relative_L2=float(np.linalg.norm(e)/max(np.linalg.norm(b),1e-30)),reference_rms=float(np.sqrt(np.mean(b*b))))
def step(ids,cs,ss):
 x=model.language_model.model.embed_tokens(mx.array([ids]));n=l.input_layernorm(x)
 qkv=ln.in_proj_qkv(n);z=ln.in_proj_z(n);a=ln.in_proj_a(n);b=ln.in_proj_b(n)
 if cs is None:cs=mx.zeros((1,3,8192),dtype=n.dtype)
 if ss is None:ss=mx.zeros((1,32,128,128),dtype=mx.float32)
 conv_input=mx.concatenate([cs,qkv],axis=1);cv=nn.silu(ln.conv1d(conv_input))
 q,k,v=[t.reshape(1,len(ids),h,128) for t,h in zip(mx.split(cv,[2048,4096],-1),[16,16,32])]
 q,k=gd.normalize_qk(q,k,inv_scale=128**-.5,eps=1e-6)
 core,ss2=gd.gated_delta_update(q,k,v,a,b,ln.A_log,ln.dt_bias,ss,use_kernel=True)
 gated=ln.norm(core,z.reshape(1,len(ids),32,128));attn=ln.out_proj(gated.reshape(1,len(ids),4096))
 res=x+attn;post=l.post_attention_layernorm(res);router=l.mlp.gate(post);gates=mx.softmax(router,axis=-1,precise=True)
 inds=mx.argpartition(gates,kth=-8,axis=-1)[...,-8:];scores=mx.take_along_axis(gates,inds,axis=-1);scores=scores/scores.sum(axis=-1,keepdims=True)
 experts=l.mlp.switch_mlp(post,inds);routed=(experts*scores[...,None]).sum(axis=-2)
 shared=mx.sigmoid(l.mlp.shared_expert_gate(post))*l.mlp.shared_expert(post)
 out=res+(routed+shared)
 comp=dict(embedding=x,norm=n,qkv=qkv,z=z,a=a,b=b,conv=cv,q=q,k=k,v=v,core=core,gated=gated.reshape(1,len(ids),4096),attn_out=attn,residual=res,post_norm=post,router=router,ids=inds,scores=scores,shared=shared,routed=routed,output=out,prior_conv=cs,prior_state=ss,post_conv=conv_input[:,-3:,:],post_state=ss2,beta=mx.sigmoid(b),decay=gd.compute_g(ln.A_log,a,ln.dt_bias))
 mx.eval(list(comp.values()))
 return comp,comp['post_conv'],ss2
# Actual first-20-token batch followed by tokenwise teacher steps vs fully sequential.
for mode in ['sequential','batched_prompt']:
 cs=ss=None
 if mode=='batched_prompt':
  batch,cs,ss=step(tokens[:20],cs,ss)
  for bp in [0,1,2]:
   od=folder/'prefix_batch'/str(bp);od.mkdir(parents=True,exist_ok=True)
   for name in ['embedding','norm','qkv','conv','gated','output']:
    evaluate(batch[name][:,bp:bp+1]).astype('<f4').tofile(od/(name+'.f32'))
  start=20
 else:start=0
 for p in range(start,81):
  comp,cs,ss=step([tokens[p]],cs,ss)
  if p not in positions:continue
  outdir=folder/mode/str(p);outdir.mkdir(parents=True,exist_ok=True)
  for name,x in comp.items():
   dtypes[name]=str(x.dtype)
   evaluate(x).astype('<f4').tofile(outdir/(name+'.f32'))
  # Native layer execution with exact same embedding and prior states checks transcription.
  cache=ArraysCache(size=2);cache[0]=comp['prior_conv'];cache[1]=comp['prior_state']
  native=l(comp['embedding'],cache=cache)
  observations[f'{mode}_{p}_manual_native']=stats(evaluate(comp['output']),evaluate(native))
  observations[f'{mode}_{p}_state_native']=stats(evaluate(comp['post_state']),evaluate(cache[1]))
# Same-input F32 head/norm/projection tests and matched-state recurrence tests against Swift.
controls={}
for p in positions:
 sw=Path('/tmp/qwen-isolation-swift-base')/str(p)
 if not sw.exists():continue
 def arr(name,shape):return mx.array(np.fromfile(sw/(name+'.f32'),dtype='<f4').reshape(shape))
 norm=arr('norm',(1,1,2048));proj=ln.in_proj_qkv(norm);controls[f'{p}_same_norm_qkv']=stats(evaluate(proj),np.fromfile(sw/'qkv.f32',dtype='<f4'))
 embed=arr('embedding',(1,1,2048));controls[f'{p}_same_embedding_norm']=stats(evaluate(l.input_layernorm(embed)),np.fromfile(sw/'norm.f32',dtype='<f4'))
 cv=arr('conv',(1,1,8192));q,k,v=[t.reshape(1,1,h,128) for t,h in zip(mx.split(cv,[2048,4096],-1),[16,16,32])];q,k=gd.normalize_qk(q,k,inv_scale=128**-.5,eps=1e-6)
 a=arr('a',(1,1,32));b=arr('b',(1,1,32));prior=arr('prior_state',(1,32,128,128));z=arr('z',(1,1,32,128))
 core,post=gd.gated_delta_update(q,k,v,a,b,ln.A_log,ln.dt_bias,prior,use_kernel=True)
 gated=ln.norm(core,z)
 controls[f'{p}_matched_state_update']=stats(evaluate(post),np.fromfile(sw/'post_state.f32',dtype='<f4'))
 controls[f'{p}_matched_state_gated']=stats(evaluate(gated),np.fromfile(sw/'gated.f32',dtype='<f4'))
 # Shared-input MoE controls: compare algebra/quantized operations, not cross-precision trajectories.
 postnorm=arr('post_norm',(1,1,2048));router=l.mlp.gate(postnorm)
 controls[f'{p}_same_postnorm_router']=stats(evaluate(router),np.fromfile(sw/'router.f32',dtype='<f4'))
 ids=mx.array(np.fromfile(sw/'ids.f32',dtype='<u4').reshape(1,1,8))
 weights=mx.array(np.fromfile(sw/'scores.f32',dtype='<f4').reshape(1,1,8))
 contributions=l.mlp.switch_mlp(postnorm,ids)*weights[...,None]
 routed=contributions.sum(axis=-2);shared=mx.sigmoid(l.mlp.shared_expert_gate(postnorm))*l.mlp.shared_expert(postnorm)
 hit=np.fromfile(sw/'hit.f32',dtype='<f4');miss=np.fromfile(sw/'miss.f32',dtype='<f4')
 actualrouted=(hit if hit.size else np.zeros(2048,dtype=np.float32))+(miss if miss.size else np.zeros(2048,dtype=np.float32))
 controls[f'{p}_same_postnorm_routed']=stats(evaluate(routed),actualrouted)
 controls[f'{p}_same_postnorm_shared']=stats(evaluate(shared),np.fromfile(sw/'shared.f32',dtype='<f4'))
 # Same expert contributions, change only sequential reduction order.
 con=evaluate(contributions).reshape(8,2048)
 ascending=np.zeros(2048,dtype=np.float32);descending=ascending.copy()
 for i in range(8):ascending=ascending+con[i];descending=descending+con[7-i]
 controls[f'{p}_expert_reduction_order']=stats(ascending,descending)
 # Alter only prior recurrent state: use actual MLX sequential FP32 prior, keep Swift current inputs.
 mprior=mx.array(np.fromfile(folder/'sequential'/str(p)/'prior_state.f32',dtype='<f4').reshape(1,32,128,128))
 _,other=gd.gated_delta_update(q,k,v,a,b,ln.A_log,ln.dt_bias,mprior,use_kernel=True)
 controls[f'{p}_prior_history_effect']=stats(evaluate(other),evaluate(post))
 # Alter only current projections/conv: replay native BF16 inputs from MLX, keep Swift FP32 prior.
 mc=folder/'sequential'/str(p)
 def ml(name,shape):return mx.array(np.fromfile(mc/(name+'.f32'),dtype='<f4').reshape(shape),dtype=mx.bfloat16)
 cv=ml('conv',(1,1,8192));q,k,v=[t.reshape(1,1,h,128) for t,h in zip(mx.split(cv,[2048,4096],-1),[16,16,32])];q,k=gd.normalize_qk(q,k,inv_scale=128**-.5,eps=1e-6)
 _,other=gd.gated_delta_update(q,k,v,ml('a',(1,1,32)),ml('b',(1,1,32)),ln.A_log,ln.dt_bias,prior,use_kernel=True)
 controls[f'{p}_current_input_effect']=stats(evaluate(other),evaluate(post))
# Verify full-attention cache precision on actual checkpoint token inputs.
cache=model.language_model.make_cache()
actual=model.language_model(mx.array([tokens[:2]]),cache=cache);mx.eval(actual,cache[3].keys,cache[3].values)
attn=model.language_model.model.layers[3].self_attn
probe=attn.rope(mx.zeros((1,16,1,256),dtype=mx.bfloat16),offset=0);mx.eval(probe)
dtypes.update(kv_keys=str(cache[3].keys.dtype),kv_values=str(cache[3].values.dtype),rope_output=str(probe.dtype),attention_logits=str(actual.dtype))
(folder/'summary.json').write_text(json.dumps(dict(dtypes=dtypes,native_checks=observations,controls=controls,peak_bytes=mx.get_peak_memory()),indent=2)+'\n')
print(json.dumps(dict(dtypes=dtypes,native_checks=observations,controls=controls,peak_bytes=mx.get_peak_memory()),indent=2))
