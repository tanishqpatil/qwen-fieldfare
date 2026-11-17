from pathlib import Path
import numpy as np,json
mxroot=Path('/tmp/qwen-isolation-mlx');positions=[0,1,2,20,40,60,80]
def read(path):return np.fromfile(path,dtype='<f4')
def stats(a,b):
 a=a.astype(np.float64).reshape(-1);b=b.astype(np.float64).reshape(-1);e=a-b
 return dict(max_abs=float(abs(e).max()),rms=float(np.sqrt(np.mean(e*e))),relative_L2=float(np.linalg.norm(e)/max(np.linalg.norm(b),1e-30)),reference_rms=float(np.sqrt(np.mean(b*b))))
results={};routes={}
for p in positions:
 mp=mxroot/'sequential'/str(p);sw=Path('/tmp/qwen-isolation-swift-base')/str(p);rounded=Path('/tmp/qwen-isolation-swift-rounded')/str(p);cache=Path('/tmp/qwen-isolation-swift-cache3')/str(p)
 d={}
 for name in ['embedding','norm','qkv','z','a','b','conv','gated','attn_out','residual','post_norm','router','shared','output','prior_state','post_state']:
  a=read(sw/(name+'.f32'));b=read(mp/(name+'.f32'));d[name]=stats(a,b)
  if (rounded/(name+'.f32')).exists():d[name+'_embedding_rounded']=stats(read(rounded/(name+'.f32')),b)
 for name in ['prior_conv','post_conv']:
  a=read(sw/(name+'.f32')).reshape(8192,4)[:,1:4].T;d[name]=stats(a,read(mp/(name+'.f32')))
 hit=read(sw/'hit.f32');miss=read(sw/'miss.f32');routed=(hit if len(hit) else np.zeros(2048))+(miss if len(miss) else np.zeros(2048));d['routed']=stats(routed,read(mp/'routed.f32'))
 if p>=20:
  for name in ['prior_state','post_state','output','qkv']:
   d[name+'_batch_vs_sequential']=stats(read(mxroot/'batched_prompt'/str(p)/(name+'.f32')),read(mp/(name+'.f32')))
 def ids(folder,name='ids'):return np.fromfile(folder/(name+'.f32'),dtype='<u4').tolist()
 si=ids(sw);mi=read(mp/'ids.f32').astype(int).tolist();ci=ids(cache)
 weights=dict(zip(si,read(sw/'scores.f32').tolist()));mw=dict(zip(mi,read(mp/'scores.f32').tolist()))
 routes[str(p)]=dict(swift_ids=si,mlx_ids=mi,same_selected_set=set(si)==set(mi),cache_same_ids=si==ci,cache_layer3_same_ids=ids(sw,'layer3_ids')==ids(cache,'layer3_ids'),weight_max_abs=max(abs(weights[i]-mw[i]) for i in si) if set(si)==set(mi) else None,cache_output=stats(read(cache/'output.f32'),read(sw/'output.f32')))
 results[str(p)]=d
Path('/tmp/qwen-isolation-comparison.json').write_text(json.dumps(dict(components=results,routing=routes),indent=2)+'\n')
for p in positions:
 d=results[str(p)];print(p,'embedding %.4f%% norm %.4f%% qkv %.4f%% conv %.4f%% state %.4f%% output %.4f%% round-output %.4f%%'%(100*d['embedding']['relative_L2'],100*d['norm']['relative_L2'],100*d['qkv']['relative_L2'],100*d['conv']['relative_L2'],100*d['post_state']['relative_L2'],100*d['output']['relative_L2'],100*d['output_embedding_rounded']['relative_L2']),routes[str(p)]['same_selected_set'],routes[str(p)]['cache_same_ids'])
print('routes',routes)
