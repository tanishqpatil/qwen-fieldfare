#!/usr/bin/env python3
"""Opt-in public-fixture client for an already UI-managed server; never loads a model."""
import argparse,http.client,json,os,pathlib,subprocess,tempfile,time
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--key-file',required=True);parser.add_argument('--port',type=int,default=8080)
parser.add_argument('--out',type=pathlib.Path,default=pathlib.Path('docs/evidence/2026-11-17-s203'))
args=parser.parse_args();key=pathlib.Path(args.key_file).read_text().strip();args.out.mkdir(parents=True,exist_ok=True)
def call(path,body=None,authorized=True):
 c=http.client.HTTPConnection('127.0.0.1',args.port,timeout=600)
 headers={'Content-Type':'application/json'}
 if authorized:headers['Authorization']='Bearer '+key
 started=time.monotonic();c.request('POST' if body is not None else 'GET',path,body=json.dumps(body).encode() if body is not None else None,headers=headers)
 r=c.getresponse();data=r.read();c.close();return r.status,data,time.monotonic()-started
metrics={'kind':'real checkpoint already owned by native UI; client executes no filesystem/shell model tools'}
assert call('/health',authorized=False)[0]==401
base={'model':'qwen-fieldfare','messages':[{'role':'user','content':'Reply with exactly: hello'}],'temperature':0,'max_tokens':32}
status,data,elapsed=call('/v1/chat/completions',base);assert status==200
reply=json.loads(data);metrics['json']={'seconds':elapsed,'usage':reply['usage'],'finish':reply['choices'][0]['finish_reason'],'content':reply['choices'][0]['message']['content']}
status,data,elapsed=call('/v1/chat/completions',dict(base,stream=True,stream_options={'include_usage':True}));assert status==200 and data.endswith(b'data: [DONE]\n\n')
events=[json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: {')]
metrics['sse']={'seconds':elapsed,'usage':next(e['usage'] for e in events if e.get('usage')),'content':''.join(c['delta'].get('content','') for e in events for c in e.get('choices',[]))}
tool={'type':'function','function':{'name':'get_project_status','description':'Get the current project status.','parameters':{'type':'object','properties':{'project':{'type':'string'}},'required':['project'],'additionalProperties':False}}}
request=dict(base,tools=[tool],max_tokens=256,messages=[{'role':'user','content':'Call get_project_status for project fieldfare. Then report the returned status.'}])
status,data,elapsed=call('/v1/chat/completions',request);assert status==200
reply=json.loads(data);assistant=reply['choices'][0]['message'];assert reply['choices'][0]['finish_reason']=='tool_calls'
call_value=assistant['tool_calls'][0];assert call_value['function']['name']=='get_project_status'
assert json.loads(call_value['function']['arguments'])=={'project':'fieldfare'}
metrics['tool']={'seconds':elapsed,'usage':reply['usage'],'name':call_value['function']['name']}
request['messages'] += [assistant,{'role':'tool','tool_call_id':call_value['id'],'content':'{"project":"fieldfare","status":"building-local-agent"}'}]
status,data,elapsed=call('/v1/chat/completions',dict(request,stream=True,stream_options={'include_usage':True}));assert status==200 and data.endswith(b'data: [DONE]\n\n')
events=[json.loads(line[6:]) for line in data.decode().splitlines() if line.startswith('data: {')]
text=''.join(c['delta'].get('content','') for e in events for c in e.get('choices',[]));assert 'building-local-agent' in text
metrics['tool_continuation']={'seconds':elapsed,'content':text,'usage':next(e['usage'] for e in events if e.get('usage'))}
# Same provider/npm/options/model fields as the native Copy OpenCode Guidance action.
root=pathlib.Path(tempfile.mkdtemp(prefix='qwen-managed-opencode-')).resolve();subprocess.run(['git','init','-q',str(root)],check=True)
config={'provider':{'fieldfare':{'npm':'@ai-sdk/openai-compatible','options':{'baseURL':f'http://127.0.0.1:{args.port}/v1','apiKey':'{env:QWENFIELDFARE_API_KEY}','timeout':600000},'models':{'qwen-fieldfare':{'limit':{'context':8192,'output':1024},'tool_call':True}}}},'model':'fieldfare/qwen-fieldfare','small_model':'fieldfare/qwen-fieldfare','agent':{'probe':{'mode':'primary','prompt':'Answer briefly. No tools.','permission':{'*':'deny'}}},'autoupdate':False,'share':'disabled'}
(root/'opencode.json').write_text(json.dumps(config))
env=dict(os.environ,PWD=str(root),QWENFIELDFARE_API_KEY=key,OPENCODE_DISABLE_MODELS_FETCH='1',OPENCODE_DISABLE_AUTOUPDATE='1',OPENCODE_DISABLE_CLAUDE_CODE='1',OPENCODE_DISABLE_EXTERNAL_SKILLS='1',OPENCODE_DISABLE_AUTOCOMPACT='1',OPENCODE_DISABLE_LSP_DOWNLOAD='1')
for name in ['CACHE','DATA','STATE','CONFIG']:env[f'XDG_{name}_HOME']=str(root/name.lower())
started=time.monotonic()
p=subprocess.run(['opencode','run','--pure','--format','json','--dir',str(root),'--model','fieldfare/qwen-fieldfare','--agent','probe','--title','UI-managed server qualification','Reply with exactly: hello'],cwd=root,env=env,capture_output=True,text=True,timeout=600)
metrics['opencode']={'version':subprocess.check_output(['opencode','--version'],env=env,text=True).strip(),'exit':p.returncode,'seconds':time.monotonic()-started}
assert p.returncode==0
client_events=[json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
assert any(e.get('type')=='text' and 'hello' in e.get('part',{}).get('text','').lower() for e in client_events)
(args.out/'managed-opencode-events.jsonl').write_text(p.stdout.replace(key,'<redacted>'))
status,data,_=call('/health');assert status==200;metrics['final_health']=json.loads(data)
(args.out/'managed-client.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
