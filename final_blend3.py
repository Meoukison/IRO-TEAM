import pandas as pd,numpy as np,pickle,glob,sys,warnings; warnings.filterwarnings('ignore')
from scipy.optimize import nnls
from post import polyproj,save
OUT=sys.argv[1]
G=pd.read_pickle('tr_gen.pkl')[['Index','patient_id','age','target','mgen_q']]; Gt=pd.read_pickle('te_gen.pkl')[['Index','patient_id','age','mgen_q']]
y=G.target.values; r=lambda p: np.sqrt(np.mean((p-y)**2))
for k,(o,t) in pickle.load(open('ens3_preds.pkl','rb')).items(): G['e_'+k]=np.asarray(o); Gt['e_'+k]=np.asarray(t)
for f,c in [('pair2','mpair2_q'),('pair3','mpair3_q')]:
    G=G.merge(pd.read_pickle(f'tr_{f}.pkl')[['Index',c]],on='Index'); Gt=Gt.merge(pd.read_pickle(f'te_{f}.pkl')[['Index',c]],on='Index')
tags=sorted(f[2:-7] for f in glob.glob('nn*_te.pkl'))
for tg in tags:
    G=G.merge(pd.read_pickle(f'nn{tg}_tr.pkl'),on='Index'); Gt=Gt.merge(pd.read_pickle(f'nn{tg}_te.pkl'),on='Index')
nnc=[f'nn{tg}' for tg in tags]; print('nets',nnc)
for d in (G,Gt):
    d['nnavg']=d[nnc].mean(axis=1); d['nnavg_q']=polyproj(d,'nnavg',2,0)
print('nnavg %.4f nnavg_q %.4f'%(r(G.nnavg.values),r(G.nnavg_q.values)))
C=['mgen_q','mpair2_q','mpair3_q','nnavg','nnavg_q']+[c for c in G if c.startswith('e_')]
w,_=nnls(G[C].values,y); print(dict(zip(C,w.round(3))))
G['bl']=G[C].values@w; Gt['bl']=Gt[C].values@w
folds=pickle.load(open('folds.pkl','rb')); hb=np.zeros(len(G))
for a,b in folds: ww,_=nnls(G[C].values[a],y[a]); hb[b]=G[C].values[b]@ww
rmse_nnls=r(G.bl.values); rmse_honest=r(hb)
print('NNLS %.4f fold-honest %.4f'%(rmse_nnls,rmse_honest))
save(Gt,'bl',OUT)
pickle.dump(dict(weights=dict(zip(C,w.round(4))),rmse_nnls=float(rmse_nnls),rmse_honest=float(rmse_honest),components=C,submission=OUT),open('blend_meta.pkl','wb'))
