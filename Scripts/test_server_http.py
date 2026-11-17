#!/usr/bin/env python3
"""Opt-in real-model server smoke test; never writes prompts, responses or secrets."""
import http.client, json, os, secrets, signal, subprocess, time, pathlib, threading
root=pathlib.Path(__file__).resolve().parent.parent
out=root/'docs/evidence/2026-11-17-s200'
key=secrets.token_urlsafe(32)
log=(out/'server.log').open('w')
env=dict(os.environ,QWENFIELDFARE_API_KEY=key)
model=os.environ['QWENFIELDFARE_SERVER_CHECKPOINT']
command=[str(root/'.build/release/QwenFieldfareServer'),'--model',model,'--port','18089','--context','8192','--memory-gib','5']
start=time.monotonic();proc=subprocess.Popen(command,env=env,stdout=log,stderr=log)
metrics={}
def call(path,body=None,auth=True):
    conn=http.client.HTTPConnection('127.0.0.1',18089,timeout=120)
    headers={'Authorization':'Bearer '+key} if auth else {}
    data=None if body is None else json.dumps(body).encode()
    if body is not None: headers['Content-Type']='application/json'
    conn.request('GET' if body is None else 'POST',path,body=data,headers=headers)
    res=conn.getresponse();return conn,res
request={'model':'qwen-fieldfare','messages':[{'role':'system','content':'Answer briefly.'},{'role':'user','content':'Say hello.'}],'temperature':0,'max_tokens':16}
try:
    for _ in range(600):
        if proc.poll() is not None: raise RuntimeError('Server failed startup')
        try:
            c,r=call('/health');status=r.status;health=json.loads(r.read());c.close()
            if status==200:break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('Readiness timeout')
    metrics['startup_to_health_seconds']=time.monotonic()-start
    assert health['status']=='ready'
    c,r=call('/health',auth=False);assert r.status==401;r.read();c.close()
    c,r=call('/v1/models');assert json.loads(r.read())['data'][0]['id']=='qwen-fieldfare';c.close()
    for name in ['first','repeated']:
        t=time.monotonic();c,r=call('/v1/chat/completions',request);assert r.status==200
        data=json.loads(r.read());c.close();assert data['choices'][0]['message']['content']
        metrics[name]={'seconds':time.monotonic()-t,'usage':data['usage'],'finish_reason':data['choices'][0]['finish_reason']}
    t=time.monotonic();c,r=call('/v1/chat/completions',dict(request,stream=True,stream_options={'include_usage':True}))
    assert r.status==200 and r.getheader('content-type')=='text/event-stream'
    pieces=[];events=0;done=False;first=None
    for line in r:
        if not line.startswith(b'data: '):continue
        if line.strip()==b'data: [DONE]':done=True;break
        event=json.loads(line[6:]);events+=1
        for choice in event['choices']:
            text=choice['delta'].get('content','')
            if text and first is None:first=time.monotonic()-t
            pieces.append(text)
    c.close();assert done and ''.join(pieces)
    metrics['stream']={'seconds':time.monotonic()-t,'first_content_seconds':first,'events':events,'done':done}
    c,r=call('/v1/chat/completions',dict(request,messages=[{'role':'user','content':'x '*9000}]))
    assert r.status==400;r.read();c.close();metrics['context_overflow_rejected']=True
    c,r=call('/v1/chat/completions',dict(request,stream=True,max_tokens=512))
    assert r.status==200
    r.readline();r.close();c.close();time.sleep(.2)
    c,r=call('/v1/chat/completions',request);assert r.status==200;assert json.loads(r.read())['choices'];c.close()
    metrics['disconnect_then_recovery']=True
    # Exercise shutdown during an active request without logging its data.
    def active():
        try:
            c,r=call('/v1/chat/completions',dict(request,max_tokens=512));r.read();c.close()
        except (OSError,http.client.HTTPException):pass
    thread=threading.Thread(target=active);thread.start();time.sleep(.2)
    t=time.monotonic();proc.send_signal(signal.SIGTERM);proc.wait(timeout=60);thread.join(timeout=5)
    metrics['exit']=proc.returncode
    assert proc.returncode==0
    metrics['shutdown_seconds']=time.monotonic()-t;metrics['exit']=proc.returncode
finally:
    if proc.poll() is None:proc.send_signal(signal.SIGTERM);proc.wait(timeout=60)
    log.close()
    (out/'http-integration.json').write_text(json.dumps(metrics,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
