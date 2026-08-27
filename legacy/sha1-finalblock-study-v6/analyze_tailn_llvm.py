#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path

FILES = {
    'vector': Path('/mnt/data/tailn_study.ll'),
    'scalar': Path('/mnt/data/tailn_scalar_study.ll'),
    'split': Path('/mnt/data/tailn_split_study.ll'),
    'fused': Path('/mnt/data/tailn_fused_study.ll'),
}

# Known candidates processed by one thread per suffix pass. For the original
# grouped kernel this is V; for scalar/split it is the suffix vector width.
PATTERNS = [
    (re.compile(r'_Z7k_tailnILi(?P<g>\d+)ELi(?P<v>\d+)ELi(?P<cache>\d+)ELb(?P<ff>[01])ELi(?P<pad>\d+)EE'),
     lambda m: (f"tailn-g{m['g']}-v{m['v']}-c{m['cache']}-ff{m['ff']}", int(m['v']))),
    (re.compile(r'_Z14k_tailn_scalarILi(?P<cache>\d+)EE'),
     lambda m: (f"scalar-c{m['cache']}", 1)),
    (re.compile(r'_Z18k_tailn_suffix_vecILi(?P<v>\d+)ELi(?P<cache>\d+)EE'),
     lambda m: (f"split-sv{m['v']}-c{m['cache']}", int(m['v']))),
    (re.compile(r'_Z13k_tailn_fusedILi(?P<v>\d+)ELi(?P<cache>\d+)ELb(?P<outline>[01])EE'),
     lambda m: (f"fused-sv{m['v']}-c{m['cache']}-call{m['outline']}", int(m['v']))),
    (re.compile(r'_Z19k_tailn_head_scalarmmPj'),
     lambda m: ("split-head", 1)),
]

def functions(text: str):
    lines=text.splitlines()
    i=0
    while i<len(lines):
        if lines[i].startswith('define '):
            start=i; name_m=re.search(r'@([^ (]+)', lines[i]); name=name_m.group(1) if name_m else '?'
            depth=lines[i].count('{')-lines[i].count('}')
            i+=1
            while i<len(lines) and depth:
                depth += lines[i].count('{')-lines[i].count('}')
                i+=1
            yield name, '\n'.join(lines[start:i])
        else:i+=1

def classify(name):
    for rx,fn in PATTERNS:
        m=rx.search(name)
        if m:return fn(m)
    return None

def metrics(body: str):
    # Static body counts. Suffix-block loop is intentionally not unrolled, so
    # SHA round/body instruction counts approximate one dynamic suffix block,
    # while loop-control counts occur once per block too.
    out={}
    pats={
        'ir_lines': r'^\s*[%a-zA-Z].*$',
        'add32': r'\badd(?: nuw| nsw| nuw nsw| nsw nuw)? i32\b',
        'xor32': r'\bxor i32\b',
        'and32': r'\band i32\b',
        'or32': r'\bor i32\b',
        'icmp': r'\bicmp\b',
        'select': r'\bselect\b',
        'load_i32': r'\bload i32\b',
        'store_i32': r'\bstore i32\b',
        'load_v4': r'\bload <4 x i32>\b',
        'store_v4': r'\bstore <4 x i32>\b',
        'fshl32': r'llvm\.fshl\.i32',
        'lop3_asm': r'lop3\.b32',
        'ldg128_ca': r'ld\.global\.ca\.v4\.u32',
        'ldg128_nc': r'ld\.global\.nc\.v4\.u32',
        'ldg128_cg': r'ld\.global\.cg\.v4\.u32',
        'atomic_cas': r'atomic\.cas|atomicCAS|atom\.cas',
        'alloca': r'\balloca\b',
        'phi': r'\bphi\b',
        'branch': r'\bbr i1\b',
    }
    flags=re.M
    for k,p in pats.items():out[k]=len(re.findall(p,body,flags))
    out['ldg128']=out['ldg128_ca']+out['ldg128_nc']+out['ldg128_cg']
    out['bytes']=len(body)
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--json');a=ap.parse_args()
    rows=[]
    for kind,path in FILES.items():
        if not path.exists():continue
        for mangled,body in functions(path.read_text(errors='replace')):
            c=classify(mangled)
            if not c:continue
            name,v=c;m=metrics(body);m.update(kind=kind,name=name,candidates_per_thread=v)
            # Per-candidate static normalization is useful for the inlined SHA
            # body; schedule loads are deliberately shared and therefore divide
            # by V. It is *not* an exact GPU instruction count.
            for key in ('add32','xor32','fshl32','lop3_asm','ldg128','load_i32','store_i32'):
                m[key+'_per_candidate']=m[key]/v
            rows.append(m)
    # stable, compact output
    keys=['name','candidates_per_thread','ir_lines','ldg128','ldg128_per_candidate','lop3_asm','lop3_asm_per_candidate','fshl32','fshl32_per_candidate','add32','add32_per_candidate','load_i32','load_i32_per_candidate','store_i32','store_i32_per_candidate','alloca','phi','branch']
    print('\t'.join(keys))
    for r in rows:
        print('\t'.join(str(round(r[k],3)) if isinstance(r[k],float) else str(r[k]) for k in keys))
    if a.json:Path(a.json).write_text(json.dumps(rows,indent=2))

if __name__=='__main__':main()
