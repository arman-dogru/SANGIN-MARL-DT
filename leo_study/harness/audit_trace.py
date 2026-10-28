import pickle,gzip,sys,torch,numpy as np
sys.path.insert(0,'/workspace/leo_study/harness')
from panama_uav.redesign_audit import audit_redesign
from panama_uav.training import load_model
from panama_uav.models import select_action
from run_job import base_config,MODEL
m,_=load_model(MODEL,base_config(),torch.device('cpu'))
for path in sys.argv[1:]:
    t=pickle.load(gzip.open(path))
    a=audit_redesign(t,lambda l,r,o:select_action(m,o,0.,np.random.default_rng(0),torch.device('cpu')))
    print(path,a['status'],{k:v for k,v in a.items() if k.endswith('verified')})
