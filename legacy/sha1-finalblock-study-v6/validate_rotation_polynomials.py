#!/usr/bin/env python3
import random
CHARSET=b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
MASK=0xffffffff

def rol(x,n): return ((x<<n)&MASK)|(x>>(32-n)) if n else x

def expand(w):
    w=list(w)+[0]*64
    for t in range(16,80): w[t]=rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1)
    return w

def poly(src):
    p=[0]*80;p[src]=1
    for t in range(16,80): p[t]=rol(p[t-3]^p[t-8]^p[t-14]^p[t-16],1)
    return p

def evalpoly(pm,x):
    z=0
    for k in range(32):
        if pm>>k&1:z^=rol(x,k)
    return z

def main():
    rng=random.Random(0x60_11_4)
    cases=0
    for src in range(16):
        p=poly(src)
        for _ in range(400):
            x=rng.getrandbits(32)
            w=[0]*16;w[src]=x
            e=expand(w)
            for t in range(16,80):
                got=evalpoly(p[t],x)
                if got!=e[t]: raise SystemExit(f'FAIL src={src} t={t} x={x:08x} got={got:08x} want={e[t]:08x}')
            cases+=1
    # Actual byte-shaped deltas for every position-in-word and charset value.
    for bytepos in range(4):
        sh=8*(3-bytepos)
        for c in CHARSET:
            x=c<<sh
            for src in range(16):
                p=poly(src);w=[0]*16;w[src]=x;e=expand(w)
                for t in range(16,80):
                    assert evalpoly(p[t],x)==e[t]
                cases+=1
    print(f'PASS: rotation-polynomial SHA-1 delta identity ({cases} source/input cases; all W16..W79)')
if __name__=='__main__': main()
