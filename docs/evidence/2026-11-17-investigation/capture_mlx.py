import sys, json, runpy, heapq
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as utils
real_load=utils.load
capture={}
def summarize(values):
    ids=sorted(range(len(values)),key=lambda i:(-values[i],i))[:5]
    return {"top5":[[i,values[i]] for i in ids], "candidates":{str(i):values[i] for i in [14988,10000]}, "top_two_margin":values[ids[0]]-values[ids[1]]}
class CaptureLayer(nn.Module):
    def __init__(self, layer, name):
        super().__init__(); self.layer=layer; self.name=name; self.tokens=0; self.is_linear=layer.is_linear
    @property
    def input_layernorm(self): return self.layer.input_layernorm
    @property
    def linear_attn(self): return self.layer.linear_attn
    def __call__(self, x, *a, **kw):
        y=self.layer(x,*a,**kw);self.tokens+=x.shape[1]
        if self.tokens==81:
            mx.eval(y)
            capture.setdefault('layers',{})[self.name]=y[0,-1].astype(mx.float32).tolist()
        return y

class CaptureHead(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head=head
        self.tokens=0
    def __call__(self, x):
        output=self.head(x)
        self.tokens+=x.shape[1]
        if self.tokens==81:
            mx.eval(x,output)
            values=output[0,-1].astype(mx.float32).tolist()
            ids=sorted(range(len(values)),key=lambda i:(-values[i],i))[:5]
            # Same-weight head cross-check; no full-vocabulary vectors persisted.
            hidden_inputs={"mlx_bf16":x,"mlx_f32":x.astype(mx.float32)}
            swift_path=Path('/tmp/qwen-investigation-swift-5-pass-1.json')
            if swift_path.exists():
                h=json.loads(swift_path.read_text())['normalized_hidden']
                hidden_inputs.update(swift_f32=mx.array([[h]],dtype=mx.float32),swift_bf16=mx.array([[h]],dtype=mx.bfloat16))
            heads={}
            for name,h in hidden_inputs.items():
                y=self.head(h);mx.eval(y)
                heads[name]={"dtype":str(y.dtype),**summarize(y[0,-1].astype(mx.float32).tolist())}
            capture.update(head_crosscheck=heads)
            capture.update(quantity='raw_logits',dtype=str(output.dtype),absolute_input_position=80,
                top5=[[i,values[i]] for i in ids], candidates={str(i):values[i] for i in [14988,10000]},
                top_two_margin=values[ids[0]]-values[ids[1]],normalized_hidden=x[0,-1].astype(mx.float32).tolist())
        return output

def load(*a,**kw):
    model,tok=real_load(*a,**kw)
    model.language_model.lm_head=CaptureHead(model.language_model.lm_head)
    if len(sys.argv)>1: # selected layers only; does not change their arithmetic
        layers=model.language_model.model.layers
        for i in [0,3,19,39]: layers[i]=CaptureLayer(layers[i],str(i))
    return model,tok
utils.load=load
out=sys.argv[1]
sys.argv=['Scripts/qwen_mlx_greedy_reference.py','--model',str(Path.home()/'models/Qwen3.6-35B-A3B-4bit'),'--prompt','Explain HashMap in Java with an example.','--max-tokens','132','--fixture',out+'-reference.json']
runpy.run_path('Scripts/qwen_mlx_greedy_reference.py',run_name='__main__')
Path(out+'-capture.json').write_text(json.dumps(capture,indent=2)+'\n')
print('PARITY_CAPTURE='+json.dumps({k:v for k,v in capture.items() if k!='normalized_hidden'}))
