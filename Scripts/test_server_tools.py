#!/usr/bin/env python3
"""Real tool cycle: the CLIENT supplies a fixed mock result; server executes nothing.
Only this public synthetic test's API exchanges are saved, never bearer secrets.
"""
import http.client,json,os,pathlib,secrets,signal,subprocess,time
root=pathlib.Path(__file__).resolve().parent.parent
out=root/'docs/evidence/2026-11-17-s201'
key=secrets.token_urlsafe(32);env=dict(os.environ,QWENFIELDFARE_API_KEY=key)
model=os.environ['QWENFIELDFARE_SERVER_CHECKPOINT']
log=(out/'server.log').open('w')
proc=subprocess.Popen([str(root/'.build/release/QwenFieldfareServer'),'--model',model,'--port','18090','--context','8192','--memory-gib','5'],env=env,stdout=log,stderr=log)
start=time.monotonic();evidence={'cycles':[]};exchanges=[]
def call(path,body=None):
 c=http.client.HTTPConnection('127.0.0.1',18090,timeout=180)
 headers={'Authorization':'Bearer '+key}
 if body is not None:headers['Content-Type']='application/json'
 c.request('POST' if body is not None else 'GET',path,body=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers=headers)
 r=c.getresponse();return c,r
def completion(body):
 t=time.monotonic();c,r=call('/v1/chat/completions',body);status=r.status;data=r.read();c.close()
 if status!=200:
  exchanges.append({'request':body,'status':status,'error':data.decode(errors='replace')})
  raise RuntimeError(f'Completion HTTP {status}: {data.decode(errors="replace")}')
 if not body.get('stream'):
  result=json.loads(data);message=result['choices'][0]['message'];finish=result['choices'][0]['finish_reason'];usage=result['usage']
  exchanges.append({'request':body,'response':result})
 else:
  chunks=[];calls={};message={'role':'assistant','content':''};finish=None;usage=None;ids=set();done=False
  for record in data.decode().split('\n\n'):
   if not record:continue
   assert record.startswith('data: ')
   if record=='data: [DONE]':done=True;continue
   chunk=json.loads(record[6:]);chunks.append(chunk);assert 'error' not in chunk
   ids.add(chunk['id'])
   if chunk.get('usage'):usage=chunk['usage']
   for choice in chunk['choices']:
    if choice.get('finish_reason'):finish=choice['finish_reason']
    delta=choice['delta'];message['content']+=delta.get('content','')
    for tc in delta.get('tool_calls',[]):
     i=tc['index'];target=calls.setdefault(i,{'id':None,'type':'function','function':{'name':None,'arguments':''}})
     if 'id' in tc:assert target['id'] is None;target['id']=tc['id']
     if 'name' in tc['function']:assert target['function']['name'] is None;target['function']['name']=tc['function']['name']
     target['function']['arguments']+=tc['function'].get('arguments','')
  assert done and len(ids)==1
  if calls:
   assert sorted(calls)==list(range(len(calls)));message['tool_calls']=[calls[i] for i in sorted(calls)]
   if not message['content']:message['content']=None
  exchanges.append({'request':body,'chunks':chunks,'assembled_message':message,'finish_reason':finish,'done':done})
 return message,finish,usage,time.monotonic()-t
try:
 for _ in range(900):
  if proc.poll() is not None:raise RuntimeError('Server failed startup')
  try:
   c,r=call('/health');health=json.loads(r.read());c.close()
   if health.get('status')=='ready':break
  except OSError:time.sleep(.1)
 else:raise RuntimeError('Readiness timeout')
 evidence['startup_seconds']=time.monotonic()-start
 tool={'type':'function','function':{'name':'get_project_status','description':'Retrieve the actual current status of a named project. Use this for project status queries.','parameters':{'type':'object','properties':{'project':{'type':'string','description':'Exact project name.'}},'required':['project'],'additionalProperties':False}}}
 for stream in [False,True]:
  messages=[{'role':'system','content':'Use the provided function to retrieve actual project status. Do not invent a status. After receiving the result, answer briefly using its exact status value.'},{'role':'user','content':'Get the actual current status of project fieldfare using get_project_status, then tell me its status.'}]
  body={'model':'qwen-fieldfare','messages':messages,'tools':[tool],'tool_choice':'auto','temperature':0,'max_tokens':512,'stream':stream}
  if stream:body['stream_options']={'include_usage':True}
  message,finish,usage,seconds=completion(body)
  assert finish=='tool_calls',('Model did not choose a tool',finish)
  calls=message.get('tool_calls',[]);assert len(calls)==1
  call_=calls[0];assert call_['function']['name']=='get_project_status'
  args=json.loads(call_['function']['arguments']);assert args=={'project':'fieldfare'},args
  # No filesystem, shell or external function is executed. Fixed client result.
  result={'project':'fieldfare','status':'building-local-agent'}
  messages=messages+[message,{'role':'tool','tool_call_id':call_['id'],'content':json.dumps(result)}]
  continuation=dict(body,messages=messages)
  final,final_finish,final_usage,continuation_seconds=completion(continuation)
  assert final_finish in ['stop','length'] and not final.get('tool_calls')
  assert 'building-local-agent' in final['content'], 'Model did not use the supplied tool result'
  evidence['cycles'].append({'stream':stream,'call_id':call_['id'],'arguments':args,'finish_reason':finish,'call_usage':usage,'call_seconds':seconds,'continuation_usage':final_usage,'continuation_seconds':continuation_seconds,'final_finish_reason':final_finish,'result_used':True})
 plain={'model':'qwen-fieldfare','messages':[{'role':'user','content':'Say hello.'}],'temperature':0,'max_tokens':32}
 for stream in [False,True]:
  body=dict(plain,stream=stream)
  if stream:body['stream_options']={'include_usage':True}
  message,finish,usage,seconds=completion(body);assert message['content'] and not message.get('tool_calls')
 evidence['plain_json_and_sse']=True
 message,finish,_,_=completion(dict(plain,tools=[tool],tool_choice='none'));assert not message.get('tool_calls');evidence['tool_choice_none']=True
 huge_tool=json.loads(json.dumps(tool));huge_tool['function']['description']='status '*10000
 c,r=call('/v1/chat/completions',dict(plain,tools=[huge_tool]));assert r.status==400;r.read();c.close();evidence['schema_context_overflow_rejected']=True
 c=http.client.HTTPConnection('127.0.0.1',18090,timeout=180)
 cancelled=dict(plain,tools=[tool],stream=True,max_tokens=512,messages=[{'role':'user','content':'Use get_project_status to check project fieldfare.'}])
 c.request('POST','/v1/chat/completions',body=json.dumps(cancelled).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 time.sleep(.1);c.close();time.sleep(.2)
 message,finish,_,_=completion(plain);assert message['content'];evidence['disconnect_recovery']=True
 t=time.monotonic();proc.send_signal(signal.SIGTERM);proc.wait(timeout=60)
 evidence['shutdown_seconds']=time.monotonic()-t;evidence['exit']=proc.returncode;assert proc.returncode==0
finally:
 if proc.poll() is None:proc.send_signal(signal.SIGTERM);proc.wait(timeout=60)
 log.close()
 (out/'http-tool-metrics.json').write_text(json.dumps(evidence,indent=2)+'\n')
 (out/'public-tool-exchanges.json').write_text(json.dumps(exchanges,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(evidence,indent=2))
