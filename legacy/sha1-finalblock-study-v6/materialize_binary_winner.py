#!/usr/bin/env python3
"""Materialize and verify a 40-bit winner from a binary-tail Git job."""
import argparse, hashlib, json, subprocess, tempfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('job',help='binary-tail job directory')
    ap.add_argument('winner',help='40-bit winner id: decimal or 0x...; outer<<8 | inner')
    ap.add_argument('--prefix-hex',required=True,help='expected SHA-1 hex prefix')
    ap.add_argument('-o','--output-dir')
    a=ap.parse_args(); d=Path(a.job); m=json.loads((d/'job.json').read_text())
    if m.get('mode')!='binary-tail-4+1': raise SystemExit('not a binary-tail-4+1 job')
    wid=int(a.winner,0)
    if wid<0 or wid >= (1<<40): raise SystemExit('winner must fit 40 bits')
    outer=(wid>>8)&0xffffffff; inner=wid&0xff
    cand=outer.to_bytes(4,'big')+bytes([inner])
    if 0 in cand: raise SystemExit(f'winner contains NUL byte and is intentionally invalid: {cand.hex()}')
    prefix=a.prefix_hex.lower().removeprefix('0x')
    if not prefix or len(prefix)>40 or any(c not in '0123456789abcdef' for c in prefix): raise SystemExit('bad --prefix-hex')
    payload=bytearray((d/'payload_template.bin').read_bytes()); po=m['nonce_payload_offset']; payload[po:po+5]=cand
    obj=b'commit '+str(len(payload)).encode()+b'\0'+payload
    # Ensure the template's object location agrees with direct serialization.
    oo=m['nonce_object_offset']
    if obj[oo:oo+5] != cand: raise SystemExit('job offset mismatch while materializing')
    oid=hashlib.sha1(obj).hexdigest()
    if not oid.startswith(prefix): raise SystemExit(f'winner does not match prefix: sha1={oid} prefix={prefix}')
    out=Path(a.output_dir) if a.output_dir else d/'winner'; out.mkdir(parents=True,exist_ok=True)
    (out/'commit_payload.bin').write_bytes(payload); (out/'commit_object.bin').write_bytes(obj)
    # Normal Git parser check, not --literally: proves the selected non-NUL
    # binary message is accepted as a commit object by ordinary Git plumbing.
    with tempfile.TemporaryDirectory(prefix='winner-git-') as td:
        subprocess.run(['git','init','-q',td],check=True)
        q=subprocess.run(['git','hash-object','-t','commit','-w','--stdin'],input=payload,cwd=td,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if q.returncode: raise SystemExit(f'git rejected materialized commit: {q.stderr.decode(errors="replace")}')
        git_oid=q.stdout.decode().strip()
        if git_oid!=oid: raise SystemExit(f'git/hashlib oid mismatch: {git_oid} != {oid}')
        cat=subprocess.run(['git','cat-file','commit',git_oid],cwd=td,stdout=subprocess.PIPE,check=True).stdout
        if cat!=payload: raise SystemExit('git cat-file roundtrip mismatch')
    info={'winner_id':wid,'winner_hex':f'{wid:010x}','outer':f'{outer:08x}','inner':f'{inner:02x}','nonce_hex':cand.hex(),'sha1':oid,'prefix':prefix,'payload_bytes':len(payload)}
    (out/'winner.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__': main()
