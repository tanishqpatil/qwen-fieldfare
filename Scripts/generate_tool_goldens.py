import json,pathlib,jinja2
from jinja2 import nodes
from jinja2.ext import Extension
class Generation(Extension):
 tags={'generation'}
 def parse(self,parser):
  next(parser.stream);return nodes.Scope(parser.parse_statements(['name:endgeneration'],drop_needle=True))
base=pathlib.Path('Tests/QwenFieldfareServerTests/Fixtures')
properties={'project':{'type':'string'},'details':{'type':'object','properties':{'ok':{'type':'boolean'}},'required':['ok'],'additionalProperties':False},'values':{'type':'array','items':{'type':'number'}},'empty':{'type':'null'}}
tool={'type':'function','function':{'name':'get_project_status','description':'Look up the current project status.','parameters':{'type':'object','properties':properties,'required':['project'],'additionalProperties':False}}}
user={'role':'user','content':'Look up fieldfare.'}
args={'project':'fieldfare "雪"\nnext line','details':{'ok':True},'values':[1,2.5],'empty':None}
call={'id':'call_first','type':'function','function':{'name':'get_project_status','arguments':json.dumps(args,ensure_ascii=False)}}
assistant={'role':'assistant','content':'Checking.','tool_calls':[call]}
result={'role':'tool','tool_call_id':'call_first','content':'{"project":"fieldfare","status":"building-local-agent"}'}
cases=[{'name':'tool-prefix','request':{'model':'qwen','messages':[{'role':'system','content':' Be brief. '},user],'tools':[tool]}}, {'name':'continuation','request':{'model':'qwen','messages':[user,assistant,result],'tools':[tool]}}, {'name':'disabled','request':{'model':'qwen','messages':[user,assistant,result],'tools':[tool],'tool_choice':'none'}}]
env=jinja2.Environment(extensions=[Generation]);env.filters['tojson']=lambda x:json.dumps(x,ensure_ascii=False);env.globals['raise_exception']=lambda x:(_ for _ in ()).throw(ValueError(x))
local=env.from_string((base/'chat_template.jinja').read_text());official=env.from_string(pathlib.Path('docs/evidence/2026-11-17-s201/official-template.jinja').read_text())
for case in cases:
 req=case['request'];messages=json.loads(json.dumps(req['messages']))
 for m in messages:
  for c in m.get('tool_calls',[]):c['function']['arguments']=json.loads(c['function']['arguments'])
 ctx={'messages':messages,'tools':req['tools'] if req.get('tool_choice')!='none' else [],'add_generation_prompt':True,'enable_thinking':False,'preserve_thinking':False}
 expected=local.render(**ctx);assert expected==official.render(**ctx)
 case['expected']=expected
(base/'goldens.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
print('Python Jinja',jinja2.__version__,'goldens generated; local and official text renderings equal for',len(cases),'cases.')
