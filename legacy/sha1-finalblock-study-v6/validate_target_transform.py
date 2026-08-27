#!/usr/bin/env python3
import random
MASK32=0xffffffff

def old(x,c,target,bits):
 m=(MASK32 << (32-bits)) & MASK32 if bits else 0
 return (((x+c)&MASK32)&m)==(target&m)
def new(x,c,target,bits):
 if bits==0:return True
 if bits==32:return x==((target-c)&MASK32)
 span=1<<(32-bits)
 lo=target & ((MASK32 << (32-bits))&MASK32)
 base=(lo-c)&MASK32
 return ((x-base)&MASK32)<span
r=random.Random(0xfeed32)
for bits in list(range(1,33)):
 for _ in range(100000):
  x=r.getrandbits(32);c=r.getrandbits(32);t=r.getrandbits(32)
  if old(x,c,t,bits)!=new(x,c,t,bits):raise SystemExit(f'FAIL bits={bits}')
print('PASS: pre-feedforward target transform (3,200,000 randomized comparisons; prefix bits 1..32)')
