#!/usr/bin/env python3
import argparse, hashlib, json, struct
from pathlib import Path
MASK=0xffffffff

def rol(x,n): return ((x<<n)&MASK)|(x>>(32-n))
def expand(w):
    w=list(w)+[0]*64
    for t in range(16,80):w[t]=rol(w[t-3]^w[t-8]^w[t-14]^w[t-16],1)
    return w
def compress(h,w16):
    w=expand(w16);a,b,c,d,e=h
    for t in range(80):
      if t<20:f=(b&c)|((~b)&d);k=0x5a827999
      elif t<40:f=b^c^d;k=0x6ed9eba1
      elif t<60:f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc
      else:f=b^c^d;k=0xca62c1d6
      z=(rol(a,5)+f+e+k+w[t])&MASK;e=d;d=c;c=rol(b,30);b=a;a=z
    return [(h[0]+a)&MASK,(h[1]+b)&MASK,(h[2]+c)&MASK,(h[3]+d)&MASK,(h[4]+e)&MASK]
def rounds(h,w,start,end):
    a,b,c,d,e=h
    for t in range(start,end):
      if t<20:f=(b&c)|((~b)&d);k=0x5a827999
      elif t<40:f=b^c^d;k=0x6ed9eba1
      elif t<60:f=(b&c)|(b&d)|(c&d);k=0x8f1bbcdc
      else:f=b^c^d;k=0xca62c1d6
      z=(rol(a,5)+f+e+k+w[t])&MASK;e=d;d=c;c=rol(b,30);b=a;a=z
    return [a,b,c,d,e]
def words(b): return [int.from_bytes(b[i:i+4],'big') for i in range(0,64,4)]
def pad(data):
    z=bytearray(data);z.append(0x80)
    while len(z)%64!=56:z.append(0)
    z+=(len(data)*8).to_bytes(8,'big');return bytes(z)
def git_obj(payload): return b'commit '+str(len(payload)).encode()+b'\0'+payload

def poly_masks(src=13):
    p=[0]*80;p[src]=1
    for t in range(16,80):p[t]=rol(p[t-3]^p[t-8]^p[t-14]^p[t-16],1)
    return p

def find_layout(payload,label):
    if not payload.endswith(b'\n'): payload+=b'\n'
    # Keep the high-throughput binary field isolated on its own final line.
    for filler in range(0,256):
      ph=b'\x01'*5
      suffix=label+b'_'*filler+ph+b'\n'
      p=payload+suffix;obj=git_obj(p); hdr=len(obj)-len(p)
      poff=len(payload)+len(label)+filler; off=hdr+poff
      if off%64==48:
        padded=pad(obj);mb=off//64
        if mb==len(padded)//64-1 and len(obj)%64==54:
          return p,obj,poff,off,filler
    raise RuntimeError('could not align 5-byte nonce at final-block offset 48')

def arr(name,x): return f'static constexpr uint32_t {name}[{len(x)}]={{'+','.join(f'0x{v:08x}u' for v in x)+'};\n'

def main():
    ap=argparse.ArgumentParser(description='Prepare exact Git final-block job for raw 4+1-byte CUDA kernel')
    ap.add_argument('commit_payload');ap.add_argument('-o','--out',required=True);ap.add_argument('--label',default='Vanity-Binary: ')
    a=ap.parse_args();src=Path(a.commit_payload).read_bytes();label=a.label.encode('ascii')
    payload,obj,poff,noff,filler=find_layout(src,label);padded=pad(obj);blocks=[padded[i:i+64] for i in range(0,len(padded),64)];mb=noff//64
    iv=[0x67452301,0xefcdab89,0x98badcfe,0x10325476,0xc3d2e1f0];h=iv
    for b in blocks[:mb]:h=compress(h,words(b))
    mutable=bytearray(blocks[mb]);mutable[48:53]=b'\0'*5;base16=words(mutable);base80=expand(base16);pre=rounds(h,base80,0,12)
    pm=poly_masks(13);complex_rounds=[t for t in range(16,80) if pm[t].bit_count()>1]
    table=[]
    for t in complex_rounds:
      for j in range(256):
        d=[0]*16;d[13]=j<<24;table.append(expand(d)[t])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    (out/'payload_template.bin').write_bytes(payload);(out/'object_template.bin').write_bytes(obj)
    (out/'complex_delta.bin').write_bytes(b''.join(struct.pack('<I',x) for x in table))
    meta=dict(mode='binary-tail-4+1',label=a.label,filler_bytes=filler,payload_bytes=len(payload),object_bytes=len(obj),
      nonce_payload_offset=poff,nonce_object_offset=noff,nonce_within_block=48,mutable_block=mb,padded_blocks=len(blocks),
      fixed_prefix_blocks=mb,data_bytes_in_final_block=len(obj)%64,first_word=12,inner_word=13,inner_shift=24,
      search_bits=40,outer_bits=32,inner_bits=8,complex_rounds=complex_rounds,complex_rows=len(complex_rounds),
      complex_table_bytes=len(table)*4,hin=[f'{x:08x}' for x in h],pre=[f'{x:08x}' for x in pre],base16=[f'{x:08x}' for x in base16],
      placeholder_sha1=hashlib.sha1(obj).hexdigest(),valid_hit_probability=(255/256)**5,expected_reject_overhead=1/((255/256)**5))
    (out/'job.json').write_text(json.dumps(meta,indent=2)+'\n')
    hdr='// generated exact binary-tail job\n#pragma once\n#include <stdint.h>\n#define JOB_BINARY_TAIL 1\n'
    hdr+=arr('JOB_HIN',h)+arr('JOB_PRE',pre)+arr('JOB_BASE16',base16)+arr('JOB_BASE80',base80)
    hdr+=f'static constexpr uint64_t JOB_NONCE_OBJECT_OFFSET={noff}ull;\nstatic constexpr uint64_t JOB_NONCE_PAYLOAD_OFFSET={poff}ull;\n'
    (out/'job_constants.cuh').write_text(hdr)
    print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
