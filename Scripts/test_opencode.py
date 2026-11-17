#!/usr/bin/env python3
"""Opt-in real OpenCode qualification in a disposable Git repository.
The relay records aggregate wire timings only; it never changes request/response
bodies or executes tools. OpenCode executes explicitly permitted workspace tools.
"""
import copy,http.client,http.server,json,os,pathlib,secrets,signal,subprocess,tempfile,threading,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
OUT=ROOT/'docs/evidence/2026-11-17-s202'
OUT.mkdir(parents=True,exist_ok=True)
run=pathlib.Path(tempfile.mkdtemp(prefix='qwen-opencode-')).resolve()
workspace=run/'workspace';workspace.mkdir()
key=secrets.token_urlsafe(32);keyfile=run/'api-key';keyfile.write_text(key+'\n');keyfile.chmod(0o600)
model=os.environ['QWENFIELDFARE_SERVER_CHECKPOINT']
(workspace/'calculator.py').write_text('def add(a, b):\n    return a - b\n')
(workspace/'test_calculator.py').write_text('import unittest\nfrom calculator import add\n\nclass AdditionTests(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n    def test_four(self):\n        self.assertEqual(add(2, 2), 5)\n')
subprocess.run(['git','init','-q',str(workspace)],check=True)
permissions={'*':'deny','read':'allow','grep':'allow','edit':{'*':'deny','calculator.py':'allow','test_calculator.py':'allow'},'bash':{'*':'deny','python3 -m unittest -v':'allow'},'external_directory':'deny'}
config={'$schema':'https://opencode.ai/config.json','model':'fieldfare/qwen-fieldfare','small_model':'fieldfare/qwen-fieldfare','provider':{'fieldfare':{'npm':'@ai-sdk/openai-compatible','name':'QwenFieldfare','options':{'baseURL':'http://127.0.0.1:18093/v1','apiKey':'{env:QWENFIELDFARE_API_KEY}','timeout':600000},'models':{'qwen-fieldfare':{'name':'QwenFieldfare local','limit':{'context':8192,'output':1024},'tool_call':True,'temperature':True}}}},'agent':{'fieldfare':{'description':'Disposable local qualification','mode':'primary','prompt':'You are a careful coding assistant. Use enabled tools one at a time. Read calculator.py directly; do not list directories or use ls. Work only in the current disposable Git workspace using its canonical absolute paths. Follow the requested edit and test sequence. Never inspect outside it. Do not ask questions or delegate.','temperature':0,'steps':16,'permission':permissions}},'autoupdate':False,'share':'disabled'}
(workspace/'opencode.json').write_text(json.dumps(config,indent=2)+'\n')
# Commit only a placeholder example, never the runtime key or user settings.
example=json.loads(json.dumps(config));example['provider']['fieldfare']['options']['baseURL']='http://127.0.0.1:8080/v1'
example['agent']['fieldfare']['permission']={'*':'deny','read':'allow','grep':'allow','edit':'ask','bash':'ask','external_directory':'deny'}
(OUT/'provider-example.json').write_text(json.dumps(example,indent=2)+'\n')
client_env=dict(os.environ,PWD=str(workspace),QWENFIELDFARE_API_KEY=key,OPENCODE_DISABLE_MODELS_FETCH='1',OPENCODE_DISABLE_AUTOUPDATE='1',OPENCODE_DISABLE_CLAUDE_CODE='1',OPENCODE_DISABLE_EXTERNAL_SKILLS='1',OPENCODE_DISABLE_AUTOCOMPACT='1',OPENCODE_DISABLE_LSP_DOWNLOAD='1')
for name,folder in [('CACHE','cache'),('DATA','data'),('STATE','state'),('CONFIG','config')]:client_env[f'XDG_{name}_HOME']=str(run/folder)
server_env=dict(os.environ);server_env.pop('QWENFIELDFARE_API_KEY',None)
requests=[];req_lock=threading.Lock()
def save_requests():
 (OUT/'requests.json').write_text(json.dumps(requests,indent=2)+'\n')
class Relay(http.server.BaseHTTPRequestHandler):
 protocol_version='HTTP/1.0'
 def log_message(self,*args):pass
 def do_POST(self):
  body=self.rfile.read(int(self.headers.get('Content-Length','0')))
  parsed=json.loads(body);started=time.monotonic()
  stat={'number':len(requests)+1,'keys':list(parsed),'roles':[m['role'] for m in parsed.get('messages',[])],'message_count':len(parsed.get('messages',[])),'body_bytes':len(body),'tool_names':[t['function']['name'] for t in parsed.get('tools',[])],'started':started}
  with req_lock:requests.append(stat);save_requests()
  connection=http.client.HTTPConnection('127.0.0.1',18092,timeout=900)
  try:
   connection.request('POST',self.path,body=body,headers={'Authorization':self.headers.get('Authorization',''),'Content-Type':'application/json'})
   response=connection.getresponse();stat['status']=response.status;stat['header_seconds']=time.monotonic()-started
   self.send_response(response.status);self.send_header('Content-Type',response.getheader('Content-Type','application/json'));self.send_header('Connection','close');self.end_headers()
   if response.getheader('Content-Type')=='text/event-stream':
    for line in response:
     if line.startswith(b'data: {'):
      event=json.loads(line[6:])
      if event.get('usage'):stat['usage']=event['usage']
      for choice in event.get('choices',[]):
       delta=choice.get('delta',{})
       if delta.get('content') and 'first_content_seconds' not in stat:stat['first_content_seconds']=time.monotonic()-started
       if delta.get('tool_calls') and 'first_tool_seconds' not in stat:stat['first_tool_seconds']=time.monotonic()-started
       if choice.get('finish_reason'):stat['finish_reason']=choice['finish_reason']
     self.wfile.write(line);self.wfile.flush()
   else:
    data=response.read();self.wfile.write(data)
    try:stat['error']=json.loads(data).get('error',{}).get('message')
    except Exception:pass
  except (OSError,http.client.HTTPException) as error:stat['transport_error']=type(error).__name__
  finally:
   connection.close();stat['wall_seconds']=time.monotonic()-started
   with req_lock:save_requests()
relay=http.server.ThreadingHTTPServer(('127.0.0.1',18093),Relay)
threading.Thread(target=relay.serve_forever,daemon=True).start()
log=(OUT/'server.log').open('w')
server=subprocess.Popen([str(ROOT/'.build/release/QwenFieldfareServer'),'--model',model,'--host','127.0.0.1','--port','18092','--context','8192','--memory-gib','5','--queue-capacity','2','--api-key-file',str(keyfile)],env=server_env,stdout=log,stderr=log)
metrics={'workspace':str(workspace),'opencode_version':subprocess.check_output(['opencode','--version'],env=client_env,text=True).strip(),'relay':'Byte-preserving loopback timing relay18093 -> actual server18092; no tool execution or payload modification','request_log':'Aggregate timings/usage only, no conversation or argument logging','client_permission_policy':'OpenCode read/edit patterns are workspace-relative; outside directory deny; edits limited to two fixture files; exact unittest command only' }
client=None
sample_stop=threading.Event()
def sample_memory():
 while not sample_stop.wait(2):
  sampled=subprocess.run(['ps','-o','rss=','-p',str(server.pid)],capture_output=True,text=True)
  if sampled.returncode == 0 and sampled.stdout.strip().isdigit():
   metrics['peak_sampled_rss_bytes']=max(metrics.get('peak_sampled_rss_bytes',0),int(sampled.stdout.strip())*1024)
threading.Thread(target=sample_memory,daemon=True).start()
def health_now():
 c=http.client.HTTPConnection('127.0.0.1',18092,timeout=3)
 c.request('GET','/health',headers={'Authorization':'Bearer '+key})
 r=c.getresponse();data=json.loads(r.read());c.close();return data

def probe_client(name,prompt,output=128,context=8192):
 # Separate session/workspace so qualification conversations never contaminate one another.
 work=run/name;work.mkdir();subprocess.run(['git','init','-q',str(work)],check=True)
 cfg=copy.deepcopy(config)
 cfg['provider']['fieldfare']['options']['baseURL']='http://127.0.0.1:18092/v1'
 cfg['provider']['fieldfare']['models']['qwen-fieldfare']['limit']={'context':context,'output':output}
 cfg['agent']['fieldfare']['permission']={'*':'deny'}
 cfg['agent']['fieldfare']['prompt']='Answer briefly. No tools.'
 (work/'opencode.json').write_text(json.dumps(cfg))
 env=dict(client_env,PWD=str(work))
 return subprocess.Popen(['opencode','run','--pure','--format','json','--dir',str(work),'--model','fieldfare/qwen-fieldfare','--agent','fieldfare','--title',name,prompt],cwd=work,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

def record_probe(name,process):
 stdout,stderr=process.communicate(timeout=180)
 result={'exit':process.returncode,'events':stdout.replace(key,'<redacted>'),'stderr':stderr.replace(key,'<redacted>')}
 (OUT/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
 return result
try:
 for _ in range(900):
  if server.poll() is not None:raise RuntimeError('Server startup failed')
  try:
   c=http.client.HTTPConnection('127.0.0.1',18092,timeout=2);c.request('GET','/health',headers={'Authorization':'Bearer '+key});r=c.getresponse();health=json.loads(r.read());c.close()
   if health.get('status')=='ready':break
  except OSError:time.sleep(.1)
 else:raise RuntimeError('Readiness timeout')
 metrics['health']=health
 # A rejected server/Bench owner probe must not instantiate a second model.
 conflict=subprocess.run([str(ROOT/'.build/release/QwenFieldfareBench'),'--model',model,'--context','8192','--memory-gib','5','--load-only'],env=server_env,capture_output=True,text=True,timeout=30)
 metrics['bench_owner_probe_exit']=conflict.returncode
 metrics['bench_owner_rejected']='owns inference' in conflict.stdout+conflict.stderr
 assert conflict.returncode != 0 and metrics['bench_owner_rejected']
 listing=subprocess.run(['opencode','models','fieldfare','--pure'],cwd=workspace,env=client_env,capture_output=True,text=True,timeout=60)
 metrics['listed_model']='fieldfare/qwen-fieldfare' in listing.stdout
 assert metrics['listed_model']
 # Client-level failures must be understandable without performing GPU inference.
 invalid=record_probe('opencode-output-error',probe_client('output-error','Say hello.',output=5000))
 assert invalid['exit'] != 0 and 'Output limit must be 1–4096' in invalid['events']
 overflow=record_probe('opencode-context-error',probe_client('context-error','Say hello. '+'alpha '*9000,context=32768))
 assert overflow['exit'] != 0 and 'Prompt plus output reservation exceeds context' in overflow['events']
 metrics['client_output_error']=True;metrics['client_context_error']=True
 # Cancel an admitted real OpenCode request before its first prefill completes.
 cancel=probe_client('cancellation','Explain addition in detail.',output=256)
 for _ in range(600):
  if health_now().get('active_generations') == 1:break
  if cancel.poll() is not None:raise RuntimeError('Cancellation client exited before admission')
  time.sleep(.1)
 else:raise RuntimeError('Cancellation never reached inference admission')
 cancelled_at=time.monotonic();cancel.send_signal(signal.SIGINT)
 cancelled=record_probe('opencode-cancellation',cancel)
 for _ in range(1200):
  current=health_now()
  if current.get('active_generations') == 0 and current.get('queued_requests') == 0:break
  time.sleep(.1)
 else:raise RuntimeError('Cancelled request did not drain')
 metrics['client_cancel_exit']=cancelled['exit'];metrics['cancel_drain_seconds']=time.monotonic()-cancelled_at;metrics['health_after_cancel']=current
 assert cancelled['exit'] != 0
 prompt='In this disposable repository, add(a,b) must return a+b. First read calculator.py, then search related Python code for add using grep. Make the small implementation fix. Run exactly `python3 -m unittest -v`. Do not edit tests before observing this run: one test has an intentionally incorrect expectation. Diagnose the failure using its output and test source, repair only the incorrect expectation to match addition, then rerun exactly `python3 -m unittest -v`. Finish with a brief report only after both tests pass. Use tools, not suggested commands; do not delegate or inspect outside the workspace.'
 started=time.monotonic()
 client=subprocess.Popen(['opencode','run','--pure','--format','json','--dir',str(workspace),'--model','fieldfare/qwen-fieldfare','--agent','fieldfare','--title','QwenFieldfare disposable qualification',prompt],cwd=workspace,env=client_env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 stdout,stderr=client.communicate(timeout=3600)
 metrics['wall_seconds']=time.monotonic()-started;metrics['opencode_exit']=client.returncode
 (OUT/'opencode-events.jsonl').write_text(stdout.replace(key,'<redacted>'));(OUT/'opencode-stderr.txt').write_text(stderr.replace(key,'<redacted>'))
 events=[json.loads(s) for s in stdout.splitlines() if s.startswith('{')]
 tool_events=[e for e in events if e.get('type')=='tool_use']
 metrics['tools']=[{'tool':e['part']['tool'],'status':e['part']['state']['status']} for e in tool_events]
 metrics['tool_successes']=sum(t['status']=='completed' for t in metrics['tools']);metrics['tool_errors']=sum(t['status']=='error' for t in metrics['tools'])
 metrics['final_response']=any(e.get('type')=='text' and e.get('part',{}).get('text','').strip() for e in events)
 final_test=subprocess.run(['python3','-m','unittest','-v'],cwd=workspace,capture_output=True,text=True)
 metrics['final_tests_exit']=final_test.returncode
 (OUT/'final-tests.txt').write_text(final_test.stdout+final_test.stderr)
 (OUT/'final-calculator.py.txt').write_text((workspace/'calculator.py').read_text());(OUT/'final-test_calculator.py.txt').write_text((workspace/'test_calculator.py').read_text())
 bash=[e['part']['state'] for e in tool_events if e['part']['tool']=='bash']
 metrics['observed_failing_test']=any('FAILED' in s.get('output','') or 'AssertionError' in s.get('output','') for s in bash)
 metrics['observed_passing_test']=any('\nOK' in s.get('output','') for s in bash)
 required={'read','grep','edit','bash'};metrics['workflow_tools_complete']=required.issubset({t['tool'] for t in metrics['tools'] if t['status']=='completed'})
 metrics['passed']=client.returncode==0 and final_test.returncode==0 and metrics['workflow_tools_complete'] and metrics['observed_failing_test'] and metrics['observed_passing_test'] and metrics['final_response']
 assert metrics['passed'],'OpenCode workflow did not meet the full gate; retain artifacts and diagnose'
finally:
 if client is not None and client.poll() is None:client.terminate();client.wait(timeout=30)
 sample_stop.set()
 if server.poll() is None:server.send_signal(signal.SIGTERM);server.wait(timeout=60)
 metrics['server_exit']=server.returncode
 relay.shutdown();relay.server_close();log.close()
 keyfile.unlink(missing_ok=True)
 (OUT/'opencode-metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
