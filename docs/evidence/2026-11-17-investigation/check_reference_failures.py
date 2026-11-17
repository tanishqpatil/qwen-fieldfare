import json,copy,subprocess,tempfile
from pathlib import Path
root=Path.cwd(); model=Path.home()/'models/Qwen3.6-35B-A3B-4bit'
source=json.loads((root/'docs/evidence/2026-11-17-parity/reference.json').read_text())
base=root/'docs/evidence/2026-11-17-investigation'
def safety():
    p=subprocess.run(['pgrep','-ifl','QwenFieldfareMac|QwenFieldfareBench|QwenFieldfareServer|mlx_lm|mlx-lm|ollama|LM Studio'],capture_output=True,text=True)
    if p.returncode!=1: raise RuntimeError('Model owner exclusion failed: '+p.stdout)
    p=subprocess.run(['memory_pressure','-Q'],capture_output=True,text=True,check=True)
    print(p.stdout.strip(),flush=True)
    # Validate normal pressure explicitly, not merely the free-memory percentage.
    p=subprocess.run(['sysctl','-n','kern.memorystatus_vm_pressure_level'],capture_output=True,text=True,check=True)
    if p.stdout.strip()!='1': raise RuntimeError('Memory pressure not normal: '+p.stdout)
    if __import__('shutil').disk_usage(model).free<10*1024**3:raise RuntimeError('Disk below 10 GiB')
def run(name,path,needle):
    safety()
    cmd=[str(root/'.build/release/QwenFieldfareBench'),'--model',str(model),'--context','8192','--cache-slots','2183','--teacher-reference',str(path)]
    result=subprocess.run(cmd,capture_output=True,text=True)
    output=result.stdout+result.stderr
    (base/(name+'.log')).write_text(output.replace(str(Path.home()),'$HOME'))
    expected=result.returncode!=0 and needle in output
    print(name,result.returncode,'expected outcome:',expected,flush=True)
    if not expected:raise RuntimeError(name+' unexpected outcome')
    return {'case':name,'exit':result.returncode,'expected_failure_observed':expected}
results=[]
results.append(run('restored-original',root/'docs/evidence/2026-11-17-parity/reference.json','131/132 matched'))
cases=[]
x=copy.deepcopy(source);x['reference_tokens'][0]=(x['reference_tokens'][0]+1)%248320;cases.append(('wrong-target',x,'Teacher-forced parity failed'))
x=copy.deepcopy(source);x['prompt_tokens'][0]=(x['prompt_tokens'][0]+1)%248320;cases.append(('wrong-prompt',x,'prompt-token mismatch'))
x=copy.deepcopy(source);x['checkpoint_sha256']['config.json']='0'*64;cases.append(('wrong-hash',x,'does not match'))
x=copy.deepcopy(source);x['reference_tokens'][0]=-1;cases.append(('invalid-token',x,'Invalid teacher reference'))
x=copy.deepcopy(source);x['reference_versions'].pop('mlx');cases.append(('missing-provenance',x,'Invalid reference provenance'))
with tempfile.TemporaryDirectory(prefix='qwen-investigation-negative-') as temp:
    for name,value,needle in cases:
        path=Path(temp)/(name+'.json');path.write_text(json.dumps(value));results.append(run(name,path,needle))
(base/'validation.json').write_text(json.dumps(results,indent=2)+'\n')
