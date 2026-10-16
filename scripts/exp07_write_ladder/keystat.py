"""KEYSTAT / PoE 1-step screen v1 (PROTOCOL_stage01.md, registered 2026-10-17 00:5x KST): the per-chunk gradient transforms.
Layout: TTT-E2E linear weights are (in, out) and y = x @ W, so per chunk and matrix G = sum_t x_t delta_t^T (in x out), with the key
x_t = the matrix input and delta_t = dL/dy_t. Transforms act on the key (left) side. Pure jnp; CPU-tested in test_keystat.py.
  a     G - xbar (sum delta)^T                          xbar = mean key over positions with delta_t != 0  (= G - N xbar deltabar^T)
  a_s   G - xbar_stream (sum delta)^T                   xbar_stream = causal mean key over earlier chunks of the world
  b_r   (I - U_r U_r^T) G                               U_r = top-r eigenvectors of M = sum_t x_t x_t^T (uncentred, this chunk)
  c     (C + alpha I)^-1 G                              C = W0 key second moment of generic text (per layer and matrix input)
PoE: inner loss -log softmax(z + beta log phat)_y with phat_t = (count_<t + eps) / (N_<t + eps |V|) over stream tokens before t."""
import numpy as np
import jax,jax.numpy as jnp
def t_a(G,sd,xbar):return G-jnp.outer(xbar,sd)
def t_b(G,M,r):
    w,U=jnp.linalg.eigh(M);Ur=U[:,-r:];return G-Ur@(Ur.T@G)
def t_c(G,C,alpha):return jnp.linalg.solve(C+alpha*jnp.eye(C.shape[0],dtype=C.dtype),G)
def key_stats(x,d):
    """x [T, din], d [T, dout] -> (G, sum delta, xbar over delta != 0 rows, N, M)."""
    nz=jnp.any(d!=0,axis=1);N=jnp.sum(nz);xbar=jnp.sum(jnp.where(nz[:,None],x,0.0),0)/jnp.maximum(N,1)
    return x.T@d,jnp.sum(d,0),xbar,N,x.T@x
def poe_bias(stream_ids,lo,CH,V,beta,eps,base_counts=None):
    """log phat for every position of the chunk [lo, lo+CH): counts of stream tokens strictly before each predicted target,
    i.e. before position t+1 when predicting token t+1 from position t (earlier chunks + this chunk's prefix up to t).
    Returns beta * log phat [CH, V] (fp32) and the counts used (for the causality assert)."""
    if base_counts is None:base_counts=np.bincount(np.asarray(stream_ids[:lo],np.int64),minlength=V).astype(np.float32)
    ch=np.asarray(stream_ids[lo:lo+CH],np.int64)
    oh=jax.nn.one_hot(jnp.asarray(ch),V,dtype=jnp.float32);cum=jnp.cumsum(oh,0)     # cum[t] = counts of chunk tokens 0..t (inputs up to t)
    cnt=jnp.asarray(base_counts)[None,:]+cum;n=lo+jnp.arange(1,CH+1,dtype=jnp.float32)[:,None]
    return beta*jnp.log((cnt+eps)/(n+eps*V)),cnt
# ---- amendment v1.1 (2026-10-17 01:35 KST) ----
def t_wridge(G,X,lam,a,C0eig=None):
    """(lam C0 + Chat + alpha I)^-1 G, Chat = X^T X / N, alpha = a * mean eig(lam C0 + Chat), Woodbury on U = X / sqrt(N).
    C0eig = (Q, Lambda) with C0 = Q diag(Lambda) Q^T (needed when lam > 0)."""
    N,d=X.shape;U=X/jnp.sqrt(jnp.asarray(N,X.dtype));trC=jnp.sum(U*U)
    if lam==0:
        al=a*trC/d;K=al*jnp.eye(N,dtype=X.dtype)+U@U.T
        return (G-U.T@jnp.linalg.solve(K,U@G))/al
    Q,Lm=C0eig;al=a*(lam*jnp.sum(Lm)+trC)/d;inv=1.0/(lam*Lm+al)
    Ainv=lambda Y:Q@(inv[:,None]*(Q.T@Y))                  # (lam C0 + alpha I)^-1 Y
    AG=Ainv(G);AU=Ainv(U.T)                                # d x dout, d x N
    K=jnp.eye(N,dtype=X.dtype)+U@AU
    return AG-AU@jnp.linalg.solve(K,U@AG)
def muon(G,steps=5,coef=(3.4445,-4.7750,2.0315)):
    """Newton-Schulz polar factor (Muon quintic iteration), X0 = G / ||G||_F; works on the smaller side."""
    a,b,c=coef;tr=G.shape[0]>G.shape[1];X=G.T if tr else G;X=X/(jnp.linalg.norm(X)+1e-7)
    for _ in range(steps):A=X@X.T;B=b*A+c*A@A;X=a*X+B@X
    return X.T if tr else X
def topk_zero(D,frac=0.08):
    """zero the top `frac` of |D| entries (one tensor)."""
    k=int(round(frac*D.size))
    if k==0:return D
    thr=jnp.sort(jnp.abs(D).ravel())[-k];return jnp.where(jnp.abs(D)>=thr,0.0,D)
