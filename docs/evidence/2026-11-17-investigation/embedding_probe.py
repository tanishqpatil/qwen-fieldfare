import json,struct,math
from pathlib import Path
import mlx.core as mx
root=Path.home()/'models/Qwen3.6-35B-A3B-4bit'
index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
def row(name,count,unit):
    with (root/index[name]).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n));t=h[name]
        f.seek(8+n+t['data_offsets'][0]+332*count*unit)
        return f.read(count*unit)
prefix='language_model.model.embed_tokens.'
w=struct.unpack('<256I',row(prefix+'weight',256,4))
s=struct.unpack('<32H',row(prefix+'scales',32,2));b=struct.unpack('<32H',row(prefix+'biases',32,2))
def bf16(n):return struct.unpack('<f',struct.pack('<I',n<<16))[0]
s=list(map(bf16,s));b=list(map(bf16,b))
expected=[((w[i//8]>>((i%8)*4))&15)*s[i//64]+b[i//64] for i in range(2048)]
y=mx.dequantize(mx.array([w],dtype=mx.uint32),scales=mx.array([s],dtype=mx.bfloat16),biases=mx.array([b],dtype=mx.bfloat16),group_size=64,bits=4)
mx.eval(y);actual=y.astype(mx.float32).tolist()[0];rounded=mx.array(expected,dtype=mx.bfloat16).astype(mx.float32).tolist()
e=[x-y for x,y in zip(expected,actual)]
result={'input_token':332,'mlx_dtype':str(y.dtype),'swift_formula_dtype':'float32','element_count':2048,'different_elements':sum(x!=y for x,y in zip(expected,actual)), 'max_abs':max(map(abs,e)),'rms':math.sqrt(sum(x*x for x in e)/2048),'relative_L2':math.sqrt(sum(x*x for x in e)/sum(x*x for x in actual)), 'bf16_rounded_swift_matches_mlx':rounded==actual}
print(json.dumps(result,indent=2))
