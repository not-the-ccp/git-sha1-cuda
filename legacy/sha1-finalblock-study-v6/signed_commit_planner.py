#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def git_obj(payload): return b'commit '+str(len(payload)).encode()+b'\0'+payload

def insert_armor_comment(payload, nonce=b'PPPPPP'):
    marker=b'gpgsig -----BEGIN PGP SIGNATURE-----\n'
    i=payload.find(marker)
    if i<0: raise ValueError('no PGP gpgsig header found')
    pos=i+len(marker)
    # Continuation line in a commit header: the physical line begins with SP.
    line=b' Comment: '+nonce+b'\n'
    out=payload[:pos]+line+payload[pos:]
    # nonce starts after leading SP + "Comment: "
    nonce_payload_off=pos+len(b' Comment: ')
    return out,nonce_payload_off

def analyze(payload,nonce=b'PPPPPP'):
    p,poff=insert_armor_comment(payload,nonce)
    obj=git_obj(p); hdrlen=len(obj)-len(p); off=hdrlen+poff
    total_blocks=(len(obj)+9+63)//64
    block=off//64
    return {
      'payload_bytes':len(p),'git_header_bytes':hdrlen,'object_data_bytes':len(obj),
      'nonce_payload_offset':poff,'nonce_absolute_offset':off,'nonce_block':block,
      'nonce_within_block':off%64,'padded_blocks':total_blocks,
      'candidate_dependent_blocks':total_blocks-block,
      'fixed_prefix_blocks':block,
      'suffix_bytes_after_nonce':len(obj)-(off+len(nonce)),
      'sha1_placeholder':hashlib.sha1(obj).hexdigest(),
    }

def main():
 ap=argparse.ArgumentParser();ap.add_argument('commit_payload');ap.add_argument('--nonce',default='PPPPPP');args=ap.parse_args()
 p=Path(args.commit_payload).read_bytes();print(json.dumps(analyze(p,args.nonce.encode()),indent=2))
if __name__=='__main__':main()
