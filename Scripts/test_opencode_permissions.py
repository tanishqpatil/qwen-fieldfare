#!/usr/bin/env python3
"""Model-free installed OpenCode permission rehearsal; scripted replies are not model evidence."""
import json,pathlib,tempfile,subprocess,os,http.server,threading
root=pathlib.Path(tempfile.mkdtemp(prefix='qwen-permission-contract-')).resolve();w=root/'workspace';w.mkdir()
(w/'calculator.py').write_text('def add(a, b):\n    return a - b\n')
(w/'test_calculator.py').write_text('import unittest\nfrom calculator import add\nclass AdditionTests(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(1, 2), 3)\n    def test_four(self): self.assertEqual(add(2, 2), 5)\n')
subprocess.run(['git','init','-q',str(w)],check=True)
repo=pathlib.Path(__file__).resolve().parent.parent
old=repo/'docs/evidence/2026-11-17-s202/provider-example.json'
config=json.loads(old.read_text());config['provider']['fieldfare']['options']['baseURL']='http://127.0.0.1:18091/v1'
config['agent']['fieldfare']['permission']={'*':'deny','read':'allow','grep':'allow','edit':{'*':'deny','calculator.py':'allow','test_calculator.py':'allow'},'bash':{'*':'deny','python3 -m unittest -v':'allow'},'external_directory':'deny'}
(w/'opencode.json').write_text(json.dumps(config))
steps=[('read',{'filePath':str(w/'calculator.py')}),('grep',{'pattern':'add','path':str(w),'include':'*.py'}),('edit',{'filePath':str(w/'calculator.py'),'oldString':'return a - b','newString':'return a + b'}),('bash',{'command':'python3 -m unittest -v'}),('read',{'filePath':str(w/'test_calculator.py')}),('edit',{'filePath':str(w/'test_calculator.py'),'oldString':'add(2, 2), 5','newString':'add(2, 2), 4'}),('bash',{'command':'python3 -m unittest -v'}),('read',{'filePath':str(root/'outside.txt')}),('read',{'filePath':str(w/'opencode.json')}),('edit',{'filePath':str(w/'opencode.json'),'oldString':'Disposable local qualification','newString':'Unauthorized changed description'}),('bash',{'command':'ls'})]
class Handler(http.server.BaseHTTPRequestHandler):
 count=0
 def do_POST(self):
  body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));i=Handler.count;Handler.count+=1
  if i:
   print(i,body['messages'][-1]['content'][:300],flush=True)
  base={'id':'chatcmpl-permission','object':'chat.completion.chunk','created':0,'model':'qwen-fieldfare'}
  if i<len(steps):
   name,args=steps[i];delta={'role':'assistant','tool_calls':[{'index':0,'id':f'call_{i}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]};reason='tool_calls'
  else:delta={'role':'assistant','content':'Permission contract complete.'};reason='stop'
  chunks=[dict(base,choices=[{'index':0,'delta':delta,'finish_reason':None}]),dict(base,choices=[{'index':0,'delta':{},'finish_reason':reason}])]
  data=(''.join('data: '+json.dumps(c)+'\n\n' for c in chunks)+'data: [DONE]\n\n').encode()
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
 def log_message(self,*args):pass
(root/'outside.txt').write_text('outside sentinel')
s=http.server.ThreadingHTTPServer(('127.0.0.1',18091),Handler);threading.Thread(target=s.serve_forever,daemon=True).start()
env=dict(os.environ,PWD=str(w),QWENFIELDFARE_API_KEY='dummy-contract-key',OPENCODE_DISABLE_MODELS_FETCH='1',OPENCODE_DISABLE_AUTOUPDATE='1',OPENCODE_DISABLE_CLAUDE_CODE='1',OPENCODE_DISABLE_EXTERNAL_SKILLS='1',OPENCODE_DISABLE_AUTOCOMPACT='1',OPENCODE_DISABLE_LSP_DOWNLOAD='1')
for name in ['CACHE','DATA','STATE','CONFIG']:env[f'XDG_{name}_HOME']=str(root/name.lower())
p=subprocess.run(['opencode','run','--pure','--format','json','--dir',str(w),'--model','fieldfare/qwen-fieldfare','--agent','fieldfare','--title','Permission contract','Exercise the scripted permission contract.'],cwd=w,env=env,capture_output=True,text=True,timeout=180)
(root/'events.jsonl').write_text(p.stdout);(root/'stderr').write_text(p.stderr)
events=[json.loads(l) for l in p.stdout.splitlines() if l.startswith('{')];tools=[e['part'] for e in events if e.get('type')=='tool_use']
result={'kind':'scripted mock endpoint; client permission check only, not model qualification','exit':p.returncode,'tools':[{'tool':t['tool'],'status':t['state']['status'],'error':t['state'].get('error','')[:200]} for t in tools]};print(json.dumps(result,indent=2));print('ROOT',root)
(repo/'docs/evidence/2026-11-17-s202/permission-contract.json').write_text(json.dumps(result,indent=2)+'\n')
s.shutdown();s.server_close();assert p.returncode==0 and [t['state']['status'] for t in tools]==['completed']*7+['error','completed','error','error']
