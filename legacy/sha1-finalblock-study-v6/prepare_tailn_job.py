#!/usr/bin/env python3
import argparse, hashlib, json, struct
from pathlib import Path
from signed_commit_planner import insert_armor_comment, git_obj

MASK=0xffffffff
CHARSET=b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
def rol(x,n): return ((x<<n)&MASK)|(x>>(32-n))
def expand(w):
 w=list(w)+[0]*64
 for t in range(16,80):w[t]=rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1)
 return w

def rounds_prefix(h,w,start,end):
 a,b,c,d,e=h
 for t in range(start,end):
  if t<20:f=(b&c)|((~b)&d);k=0x5a827999
  elif t<40:f=b^c^d;k=0x6ed9eba1
  elif t<60:f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc
  else:f=b^c^d;k=0xca62c1d6
  z=(rol(a,5)+f+e+k+w[t])&MASK;e=d;d=c;c=rol(b,30);b=a;a=z
 return [a,b,c,d,e]

def compress(h,w16):
 w=expand(w16);a,b,c,d,e=h
 for t in range(80):
  if t<20:f=(b&c)|((~b)&d);k=0x5a827999
  elif t<40:f=b^c^d;k=0x6ed9eba1
  elif t<60:f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc
  else:f=b^c^d;k=0xca62c1d6
  z=(rol(a,5)+f+e+k+w[t])&MASK;e=d;d=c;c=rol(b,30);b=a;a=z
 return [(h[0]+a)&MASK,(h[1]+b)&MASK,(h[2]+c)&MASK,(h[3]+d)&MASK,(h[4]+e)&MASK]
def words(b):return [int.from_bytes(b[i:i+4],'big') for i in range(0,64,4)]
def pad(data):
 z=bytearray(data);z.append(0x80)
 while len(z)%64!=56:z.append(0)
 z += (len(data)*8).to_bytes(8,'big');return bytes(z)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('commit_payload');ap.add_argument('-o','--out',required=True);ap.add_argument('--nonce-len',type=int,default=6);a=ap.parse_args()
 payload=Path(a.commit_payload).read_bytes(); ph=b'P'*a.nonce_len
 payload2,poff=insert_armor_comment(payload,ph);obj=git_obj(payload2); hdr=len(obj)-len(payload2); noff=hdr+poff
 mb=noff//64; within=noff%64
 if within+a.nonce_len>64: raise SystemExit('nonce crosses compression block; tailN v1 requires one mutable block')
 padded=pad(obj); blocks=[padded[i:i+64] for i in range(0,len(padded),64)]
 iv=[0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0];h=iv
 for b in blocks[:mb]:h=compress(h,words(b))
 mutable=bytearray(blocks[mb]);mutable[within:within+a.nonce_len]=b'\0'*a.nonce_len
 base16=words(mutable);base80=expand(base16);suffix80=[]
 for b in blocks[mb+1:]:suffix80.extend(expand(words(b)))
 inner_pos=within+a.nonce_len-1;inner_word=inner_pos//4;inner_shift=8*(3-(inner_pos&3));first_word=within//4
 pre=rounds_prefix(h,base80,0,first_word)
 # delta table: raw + W16..79, identical format to finalblock study
 delta=[]
 raw=[]; expanded=[[] for _ in range(64)]
 for c in CHARSET:
  d=[0]*16;d[inner_word]=c<<inner_shift;e=expand(d);raw.append(d[inner_word]);expanded[len(raw)-1]=e
 delta.extend(raw)
 for t in range(16,80):delta.extend(expanded[j][t] for j in range(64))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'suffix80.bin').write_bytes(b''.join(struct.pack('<I',x) for x in suffix80))
 (out/'delta.bin').write_bytes(b''.join(struct.pack('<I',x) for x in delta))
 (out/'object_template.bin').write_bytes(obj)
 (out/'payload_template.bin').write_bytes(payload2)
 meta=dict(nonce_len=a.nonce_len,nonce_absolute_offset=noff,nonce_within_block=within,mutable_block=mb,
           first_word=first_word,inner_word=inner_word,inner_shift=inner_shift,padded_blocks=len(blocks),
           suffix_blocks=len(blocks)-mb-1,candidate_dependent_blocks=len(blocks)-mb,
           prefix_hin=[f'{x:08x}' for x in h],pre_state=[f'{x:08x}' for x in pre],base16=[f'{x:08x}' for x in base16],
           suffix80_words=len(suffix80),suffix80_bytes=len(suffix80)*4,delta_bytes=len(delta)*4,
           placeholder_sha1=hashlib.sha1(obj).hexdigest())
 (out/'job.json').write_text(json.dumps(meta,indent=2)+'\n')
 def arr(name,x):return f'static constexpr uint32_t {name}[{len(x)}]={{'+','.join(f'0x{v:08x}u' for v in x)+'};\n'
 hdrtxt='// generated tailN job\n#pragma once\n#include <stdint.h>\n'
 hdrtxt+=f'#define TN_NONCE_LEN {a.nonce_len}\n#define TN_NONCE_OFF {within}\n#define TN_FIRST_WORD {first_word}\n#define TN_INNER_WORD {inner_word}\n#define TN_INNER_SHIFT {inner_shift}\n#define TN_SUFFIX_BLOCKS {len(blocks)-mb-1}\n'
 hdrtxt+=arr('TN_HIN',h)+arr('TN_PRE',pre)+arr('TN_BASE16',base16)
 (out/'tailn_job.cuh').write_text(hdrtxt)
 print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
