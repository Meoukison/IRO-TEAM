import pandas as pd,numpy as np,pickle,torch,torch.nn as nn,sys,time
torch.set_num_threads(2); SEED=int(sys.argv[2]); H=int(sys.argv[3]); TAG=sys.argv[4]; torch.manual_seed(SEED); np.random.seed(SEED)
Tr=pd.read_pickle('tr_gen.pkl'); Te=pd.read_pickle('te_gen.pkl'); F=pickle.load(open('FG.pkl','rb')); folds=pickle.load(open('folds.pkl','rb'))
F=[c for c in F if c in Te.columns]
y=Tr.target.values.astype(np.float32); base_tr=Tr.p6.values.astype(np.float32); base_te=Te.p6.values.astype(np.float32)
X=pd.concat([Tr[F],Te[F]]).astype(float)
mu=X.mean(); sd=X.std().replace(0,1)
def prep(D):
    Z=((D[F].astype(float)-mu)/sd).clip(-6,6); M=Z.isna().astype(np.float32).values; Z=Z.fillna(0).values.astype(np.float32)
    return np.concatenate([Z,M[:,M.std(0)>0] if False else M],1)
Ztr=prep(Tr); Zte=prep(Te)
def pack(D,Z,base,yv=None):
    idx=list(D.groupby('patient_id',sort=False).indices.values()); L=12; n=len(idx); d=Z.shape[1]
    Xb=np.zeros((n,L,d),np.float32); B=np.zeros((n,L),np.float32); Y=np.zeros((n,L),np.float32); Mk=np.ones((n,L),bool); rows=np.full((n,L),-1)
    for i,ix in enumerate(idx):
        k=len(ix); Xb[i,:k]=Z[ix]; B[i,:k]=base[ix]; Mk[i,:k]=False; rows[i,:k]=ix
        if yv is not None: Y[i,:k]=yv[ix]
    return [torch.tensor(a) for a in (Xb,B,Y,Mk)],rows,idx
class Net(nn.Module):
    def __init__(s,d,h=None):
        h=h or H; super().__init__(); s.inp=nn.Sequential(nn.Linear(d,h),nn.GELU(),nn.Linear(h,h))
        s.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(h,4,2*h,dropout=0.15,batch_first=True),4)
        s.out=nn.Sequential(nn.Linear(h,h),nn.GELU(),nn.Linear(h,1))
    def forward(s,x,m): return s.out(s.enc(s.inp(x),src_key_padding_mask=m)).squeeze(-1)
def train(P,Pv=None,epochs=60,evals=(),K=8):
    X_,B_,Y_,M_=P; net=Net(X_.shape[2]); opt=torch.optim.AdamW(net.parameters(),2e-3,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,2e-3,total_steps=epochs*((len(X_)+127)//128))
    for ep in range(epochs):
        net.train(); perm=torch.randperm(len(X_))
        for i in range(0,len(X_),128):
            b=perm[i:i+128]; p=net(X_[b],M_[b])+B_[b]; l=((p-Y_[b])**2)[~M_[b]].mean()
            opt.zero_grad(); l.backward(); opt.step(); sched.step()
        if ep>=epochs-K:
            with torch.no_grad():
                net.eval()
                for E in evals: E['acc']=E.get('acc',0)+(net(E['P'][0],E['P'][3])+E['P'][1]).numpy()/K
        if Pv is not None and ep%10==9:
            with torch.no_grad():
                net.eval(); pv=net(Pv[0],Pv[3])+Pv[1]; print(' ep',ep,'val %.4f'%((pv-Pv[2])**2)[~Pv[3]].mean().sqrt().item(),flush=True)
    return net
def predict(net,P,rows,n):
    out=np.zeros(n,np.float32)
    with torch.no_grad():
        net.eval(); p=(net(P[0],P[3])+P[1]).numpy()
    m=rows>=0; out[rows[m]]=p[m]; return out
oof=np.zeros(len(Tr),np.float32); pids=Tr.patient_id.values
EP=int(sys.argv[1]) if len(sys.argv)>1 else 60
for f,(a,b) in enumerate(folds):
    t0=time.time()
    Pa,_,_=pack(Tr.iloc[a],Ztr[a],base_tr[a],y[a]); Pb,rb,_=pack(Tr.iloc[b],Ztr[b],base_tr[b],y[b])
    E={'P':Pb}; net=train(Pa,None,EP,[E]); pr=np.zeros(len(b),np.float32); m=rb>=0; pr[rb[m]]=E['acc'][m]; oof[b]=pr
    print('fold',f,'rmse %.4f'%np.sqrt(np.mean((pr-y[b])**2)),'%.0fs'%(time.time()-t0),flush=True)
print('NN OOF %.4f'%np.sqrt(np.mean((oof-y)**2)))
Pall,_,_=pack(Tr,Ztr,base_tr,y); Pt,rt,_=pack(Te,Zte,base_te)
E={'P':Pt}; train(Pall,None,EP,[E]); te=np.zeros(len(Te),np.float32); m=rt>=0; te[rt[m]]=E['acc'][m]
pd.DataFrame({'Index':Tr.Index.values,'nn'+TAG:oof}).to_pickle(f'nn{TAG}_tr.pkl'); pd.DataFrame({'Index':Te.Index.values,'nn'+TAG:te}).to_pickle(f'nn{TAG}_te.pkl')
