#!/usr/bin/env python3
import argparse,hashlib,json,random,struct
from pathlib import Path
from prepare_tailn_job import CHARSET,compress,expand,words,pad

def main():
 ap=argparse.ArgumentParser();ap.add_argument('jobdir');ap.add_argument('--cases',type=int,default=1000);a=ap.parse_args();d=Path(a.jobdir)
 m=json.loads((d/'job.json').read_text());obj0=bytearray((d/'object_template.bin').read_bytes())
 hin=[int(x,16) for x in m['prefix_hin']];base=[int(x,16) for x in m['base16']]
 raw=(d/'suffix80.bin').read_bytes();suf=list(struct.unpack('<%dI'%(len(raw)//4),raw))
 rng=random.Random(0x7a11);off=m['nonce_absolute_offset'];L=m['nonce_len'];iw=m['inner_word'];ish=m['inner_shift'];fw=m['first_word'];within=m['nonce_within_block']
 for case in range(a.cases):
  idx=rng.randrange(64**L);x=idx;chars=[]
  for _ in range(L):chars.append(CHARSET[x&63]);x>>=6
  obj=bytearray(obj0);obj[off:off+L]=bytes(chars)
  # specialized first mutable block + fixed suffix
  w=base[:]
  for k in range(L):
   p=within+k;wi=p//4;sh=8*(3-(p&3));w[wi]|=chars[k]<<sh
  h=compress(hin,w)
  for b in range(m['suffix_blocks']):
   # compress accepts 16, but suffix is preexpanded. Inline rounds equivalent via helper unavailable;
   # recover first16 from expanded schedule.
   h=compress(h,suf[b*80:b*80+16])
  got=''.join(f'{z:08x}' for z in h)
  want=hashlib.sha1(obj).hexdigest()
  if got!=want:raise SystemExit(f'FAIL case={case} got={got} want={want}')
 print(f"PASS: tailN exact-object differential ({a.cases} randomized candidates, {m['candidate_dependent_blocks']} candidate-dependent blocks)")
if __name__=='__main__':main()
