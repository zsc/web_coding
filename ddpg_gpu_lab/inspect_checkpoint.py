import json,sys,torch
from multiseed import PackedTD3,fw
p=sys.argv[1];c=torch.load(p,map_location='cpu',weights_only=False);rows=c['meta']['history'][-1]['results'];print('checkpoint',c['meta']['updates'])
for j,row in enumerate(rows):
 if row['survived']==row['episodes']:continue
 a=PackedTD3([row['seed']],device='cpu',symmetry=True)
 for w,v in zip(a.actor,c['agent']['actor']):w.copy_(v[j:j+1])
 s=torch.eye(4)[None,:,:]*.01;z=torch.zeros(1,1,4)
 print(json.dumps({'seed':row['seed'],'steps':row['minimum_steps'],'policy_at_coordinate_0.01':a.action(s).flatten().tolist(),'raw_logits':fw(a.actor,s*a.obs_scale,True)[1][-1].flatten().tolist(),'loss':c['runner']['last_loss'][j].tolist(),'max_weight':max(float(v[j].abs().max()) for v in c['agent']['actor'])}))
