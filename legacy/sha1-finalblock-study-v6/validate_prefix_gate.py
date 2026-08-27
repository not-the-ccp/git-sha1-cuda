#!/usr/bin/env python3
"""Differential-test the generated kernel's late SHA-1 prefix gate for 4..160 bits."""
import random
MASK=(1<<32)-1

def rol(x,n): return ((x<<n)&MASK)|(x>>(32-n))
def expand(w):
    w=list(w)+[0]*64
    for t in range(16,80): w[t]=rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1)
    return w

def before79(h,w16):
    w=expand(w16); a,b,c,d,e=h
    for t in range(79):
        if t<20:f=(b&c)|((~b)&d);k=0x5a827999
        elif t<40:f=b^c^d;k=0x6ed9eba1
        elif t<60:f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc
        else:f=b^c^d;k=0xca62c1d6
        z=(rol(a,5)+f+e+k+w[t])&MASK;e=d;d=c;c=rol(b,30);b=a;a=z
    # final round output a80 and pre-round-79 state needed for H1..H4
    z=(rol(a,5)+(b^c^d)+e+0xca62c1d6+w[79])&MASK
    words=[(h[0]+z)&MASK,(h[1]+a)&MASK,(h[2]+rol(b,30))&MASK,(h[3]+c)&MASK,(h[4]+d)&MASK]
    return z,(a,b,c,d,e),words

def gate(a80,s,h,prefix_value,bits):
    words=[(prefix_value>>(128-32*i))&MASK for i in range(5)]
    if bits<32:
        mask=(MASK<<(32-bits))&MASK; span=1<<(32-bits); base=((words[0]&mask)-h[0])&MASK
        if ((a80-base)&MASK)>=span:return False
    else:
        if a80 != ((words[0]-h[0])&MASK):return False
    vals=[None,(s[0]+h[1])&MASK,(rol(s[1],30)+h[2])&MASK,(s[2]+h[3])&MASK,(s[3]+h[4])&MASK]
    for wi in range(1,5):
        used=bits-32*wi
        if used<=0:break
        take=min(32,used); mask=MASK if take==32 else (MASK<<(32-take))&MASK
        if (vals[wi]&mask)!=(words[wi]&mask):return False
    return True

def main():
    r=random.Random(0x5A1F1A7E); cases=4000; checks=0
    for _ in range(cases):
        h=[r.getrandbits(32) for _ in range(5)];w=[r.getrandbits(32) for _ in range(16)]
        a80,s,dig=before79(h,w); full=sum(x<<(128-32*i) for i,x in enumerate(dig))
        for bits in range(4,161,4):
            # matching prefix
            pv=full & (((1<<bits)-1)<<(160-bits)); assert gate(a80,s,h,pv,bits); checks+=1
            # a guaranteed-different prefix by flipping its final selected bit
            bad=pv ^ (1<<(160-bits)); assert not gate(a80,s,h,bad,bits); checks+=1
    print(f'PASS: {checks} late-prefix gate checks ({cases} digests x 40 prefix widths x match/mismatch)')
if __name__=='__main__':main()
