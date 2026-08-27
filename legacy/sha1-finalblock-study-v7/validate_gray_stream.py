#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,random,struct,sys
from pathlib import Path
from validate_literal_stream import MASK,rol,expand,poly,apply,compress

def gray(x:int)->int:return x^(x>>1)
def ctz(x:int)->int:
 if not x:raise ValueError('ctz(0)')
 return (x&-x).bit_length()-1

def main():
 d=Path(sys.argv[1]);m=json.loads((d/'job.json').read_text());b16=[int(x,16) for x in m['base16']];hin=[int(x,16) for x in m['hin']];op=poly(12)
 raw=(d/'outer_gray_delta.bin').read_bytes();tab=list(struct.unpack('<%dI'%(len(raw)//4),raw))
 if len(tab)!=32*64:raise SystemExit('bad outer basis size')
 # Basis file must equal direct linear transform for every source bit/round.
 for bit in range(32):
  x=1<<bit
  for k,t in enumerate(range(16,80)):
   want=apply(x,op[t]);got=tab[bit*64+k]
   if got!=want:raise SystemExit(f'basis mismatch bit={bit} t={t}')
 rng=random.Random(0x67a9)
 checks=0
 # Simulate persistent Gray ranges, including nonzero starts.  Maintain the
 # entire distributed schedule state by exactly the XOR update used on GPU.
 for _ in range(800):
  start=rng.randrange(0,(1<<32)-300);length=rng.randrange(1,257)
  outer=gray(start)&MASK
  state=[0]*64
  for bit in range(32):
   if outer>>bit&1:
    row=tab[bit*64:(bit+1)*64]
    state=[a^b for a,b in zip(state,row)]
  for ix in range(start,start+length):
   direct=[apply(outer,op[t]) for t in range(16,80)]
   if state!=direct:raise SystemExit(f'gray state mismatch start={start:x} ix={ix:x}')
   checks+=64
   if ix+1<start+length:
    bit=ctz((ix+1)&MASK)
    row=tab[bit*64:(bit+1)*64]
    state=[a^b for a,b in zip(state,row)]
    outer^=1<<bit
    if outer!=gray(ix+1):raise SystemExit('gray outer update mismatch')
 # Exact object/digest equivalence for Gray-enumerated outers.
 obj=bytearray((d/'object_template.bin').read_bytes());off=m['nonce_object_offset']
 for _ in range(1500):
  ix=rng.randrange(1<<32);outer=gray(ix)&MASK;inner=rng.randrange(256)
  w=b16.copy();w[12]=outer;w[13]|=inner<<24
  got=compress(hin,w)
  z=obj.copy();z[off:off+5]=outer.to_bytes(4,'big')+bytes([inner])
  want=hashlib.sha1(z).digest();gb=b''.join(x.to_bytes(4,'big') for x in got)
  if gb!=want:raise SystemExit('gray exact-object digest mismatch')
 # Inner mappings for the two generated widths.
 for V in (4,8):
  seen=[]
  chunk=32*V
  for c in range(0,256,chunk):
   for lane in range(32):
    seen.extend(c+lane*V+i for i in range(V))
  if sorted(seen)!=list(range(256)) or len(seen)!=len(set(seen)):
   raise SystemExit(f'inner mapping bad V={V}')
 print(f'PASS: Gray basis 32x64, {checks:,} persistent schedule-word checks, 1,500 exact Git SHA-1 cases, V4/V8 coverage')
if __name__=='__main__':main()
