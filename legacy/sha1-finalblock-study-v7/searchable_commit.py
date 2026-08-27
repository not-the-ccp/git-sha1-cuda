#!/usr/bin/env python3
"""Analyze or prepare a Git commit payload for efficient SHA-1 vanity searching.

This tool does *not* search hashes. It plans the serialized commit layout so a
GPU backend can do as little candidate-dependent SHA-1 work as possible.

Input is the raw commit payload (the bytes returned by `git cat-file commit X`),
not the loose-object header. Git hashes: b"commit " + decimal_length + NUL + payload.
"""
from __future__ import annotations
import argparse, dataclasses, hashlib, json, math, sys, subprocess
from pathlib import Path
from typing import Optional

B64URL = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"


def git_object(payload: bytes) -> bytes:
    return b"commit " + str(len(payload)).encode("ascii") + b"\0" + payload


def padded_blocks(nbytes: int) -> int:
    return (nbytes + 9 + 63) // 64


def success_probability(space: int, target_bits: int) -> float:
    # 1-(1-2^-b)^N; exp form stays well-behaved for useful sizes.
    if space <= 0:
        return 0.0
    lam = math.ldexp(float(space), -target_bits) if target_bits < 1024 else 0.0
    if lam > 50:
        return 1.0
    return -math.expm1(-lam)


def bits_for_probability(target_bits: int, p: float) -> float:
    # Solve 1-exp(-2^(s-b)) >= p.
    return target_bits + math.log2(-math.log1p(-p))


def detect_signature(payload: bytes) -> str:
    if b"\ngpgsig -----BEGIN PGP SIGNATURE-----\n" in b"\n" + payload:
        return "pgp"
    if b"\ngpgsig -----BEGIN SSH SIGNATURE-----\n" in b"\n" + payload:
        return "ssh"
    if b"\ngpgsig -----BEGIN SIGNED MESSAGE-----\n" in b"\n" + payload:
        return "x509"
    if b"\ngpgsig " in b"\n" + payload:
        return "other"
    return "none"


def candidate_metrics(payload: bytes, nonce_payload_offset: int, nonce_len: int, alphabet_size: int) -> dict:
    obj = git_object(payload)
    hdr = len(obj) - len(payload)
    abs_off = hdr + nonce_payload_offset
    pb = padded_blocks(len(obj))
    first_block = abs_off // 64
    last_nonce_block = (abs_off + nonce_len - 1) // 64
    return {
        "git_object_header_bytes": hdr,
        "payload_bytes": len(payload),
        "object_bytes_before_padding": len(obj),
        "padded_blocks": pb,
        "nonce_payload_offset": nonce_payload_offset,
        "nonce_object_offset": abs_off,
        "nonce_length": nonce_len,
        "nonce_block": first_block,
        "nonce_last_block": last_nonce_block,
        "nonce_within_block": abs_off % 64,
        "first_affected_word": (abs_off % 64) // 4,
        "candidate_dependent_blocks": pb - first_block,
        "nonce_crosses_block": first_block != last_nonce_block,
        "search_space": alphabet_size ** nonce_len,
        "search_bits": math.log2(alphabet_size) * nonce_len,
    }


def append_aligned_tail(payload: bytes, *, label: bytes, nonce_len: int, target_mod: int, filler_byte: bytes = b"_") -> tuple[bytes,int,int]:
    if len(filler_byte) != 1:
        raise ValueError("filler byte must be length 1")
    base = payload if payload.endswith(b"\n") else payload + b"\n"
    for filler in range(0, 4096):
        nonce = b"P" * nonce_len
        line = label + filler_byte * filler + nonce + b"\n"
        candidate = base + line
        poff = len(base) + len(label) + filler
        obj = git_object(candidate)
        abs_off = len(obj) - len(candidate) + poff
        if abs_off % 64 == target_mod:
            return candidate, poff, filler
    raise RuntimeError("failed to find alignment within 4096 filler bytes")


def best_printable_tail(payload: bytes, *, label: bytes, nonce_len: int) -> tuple[bytes,int,int]:
    # Search all alignments induced by filler length. Prefer a one-block candidate
    # tail, then the latest first affected SHA word/byte. This keeps 7/8-char
    # nonces in the final padded block instead of blindly forcing offset 48.
    base = payload if payload.endswith(b"\n") else payload + b"\n"
    best=None
    for filler in range(0, 512):
        candidate=base+label+b"_"*filler+b"P"*nonce_len+b"\n"
        poff=len(base)+len(label)+filler
        m=candidate_metrics(candidate,poff,nonce_len,64)
        # Require nonce itself to stay in one compression block. Prefer one
        # candidate-dependent block and legal SHA padding in that same block.
        if m["nonce_crosses_block"]:
            continue
        key=(m["candidate_dependent_blocks"], -m["first_affected_word"], -m["nonce_within_block"], filler)
        if best is None or key<best[0]: best=(key,candidate,poff,filler)
    if best is None: raise RuntimeError("no usable tail alignment")
    return best[1],best[2],best[3]

def insert_pgp_armor_comment(payload: bytes, nonce_len: int, prefix: bytes=b"Vanity-") -> tuple[bytes,int]:
    marker=b"gpgsig -----BEGIN PGP SIGNATURE-----\n"
    pos=payload.find(marker)
    if pos < 0:
        raise ValueError("PGP gpgsig header not found")
    pos += len(marker)
    # Every continuation line of a multi-line Git header starts with one SP.
    physical = b" Comment: " + prefix + b"P"*nonce_len + b"\n"
    out=payload[:pos]+physical+payload[pos:]
    poff=pos+len(b" Comment: ")+len(prefix)
    return out,poff



def best_pgp_armor_comment(payload: bytes, nonce_len: int) -> tuple[bytes,int,int]:
    # Vary harmless Comment text before the nonce. First minimize total
    # candidate-dependent blocks, then move the nonce as late as possible in
    # its mutable block. The latter only affects one head block, but is free.
    best=None
    for filler in range(0,64):
        prefix=b"Vanity-"+b"_"*filler
        p,off=insert_pgp_armor_comment(payload,nonce_len,prefix)
        m=candidate_metrics(p,off,nonce_len,64)
        if m["nonce_crosses_block"]: continue
        key=(m["candidate_dependent_blocks"], -m["first_affected_word"], -m["nonce_within_block"], filler)
        if best is None or key<best[0]:best=(key,p,off,filler)
    if best is None: raise RuntimeError("no PGP armor nonce placement found")
    return best[1],best[2],best[3]

def make_plan(name: str, payload: bytes, poff: int, nlen: int, alphabet: int, bits: int,
              cleanliness: str, signature_effect: str, note: str, legal_fraction: float=1.0) -> dict:
    m=candidate_metrics(payload,poff,nlen,alphabet)
    effective_space=m["search_space"]*legal_fraction
    m.update({
        "name":name,
        "cleanliness":cleanliness,
        "signature_effect":signature_effect,
        "effective_search_space":effective_space,
        "effective_search_bits":math.log2(effective_space),
        "success_probability":success_probability(int(effective_space),bits),
        "note":note,
        "backend": ("tailN-pgp" if name.startswith("pgp-") else "tail1-binary" if name.startswith("binary-") else "tail1-printable"),
        "compression_work_multiplier_vs_tail1": m["candidate_dependent_blocks"],
    })
    return m


def plans(payload: bytes, bits: int) -> list[dict]:
    sig=detect_signature(payload)
    out=[]
    # Clean printable default. Offset 48 gives W12/W13 and a late first dependency.
    for n in (5,6,7,8):
        p,off,fill=best_printable_tail(payload,label=b"Vanity: ",nonce_len=n)
        out.append(make_plan(
            f"printable-tail-{n}",p,off,n,64,bits,"clean-text",
            "invalidates-existing-signature" if sig!="none" else "unsigned",
            f"base64url nonce; {fill} alignment filler bytes; final-message placement"
        ))
    # Maximum-throughput 4+1 raw field. All 256 values are hashed; winner is
    # accepted only when no byte is NUL. This preserves direct W12 counter math.
    p,off,fill=append_aligned_tail(payload,label=b"Vanity-Binary: ",nonce_len=5,target_mod=48)
    out.append(make_plan(
        "binary-tail-4+1",p,off,5,256,bits,"binary-message",
        "invalidates-existing-signature" if sig!="none" else "unsigned",
        f"raw W12 counter + raw W13 byte; {fill} filler bytes; reject NUL-containing winners",
        legal_fraction=(255/256)**5,
    ))
    if sig=="pgp":
        for n in (6,8,10):
            p,off,fill=best_pgp_armor_comment(payload,n)
            out.append(make_plan(
                f"pgp-armor-comment-{n}",p,off,n,64,bits,"clean-signature-metadata",
                "preserves-signed-payload; signature packet can remain unchanged",
                f"PGP ASCII-armor Comment continuation inside gpgsig; {fill} alignment filler bytes; benchmark candidate-dependent suffix before choosing"
            ))
    return out


def score(p: dict, sig: str, want_preserve: bool, prefer_throughput: bool=False) -> tuple:
    # Feasibility first, then SHA work, then later word, then cleanliness/search margin.
    if want_preserve:
        sig_bad = 0 if p["signature_effect"].startswith("preserves") or sig=="none" else 1
    else:
        sig_bad = 0
    insufficient = 0 if p["success_probability"] >= 0.99 else 1
    clean_rank={"binary-message":0,"clean-text":1,"clean-signature-metadata":1}.get(p["cleanliness"],2) if prefer_throughput else {"clean-text":0,"clean-signature-metadata":0,"binary-message":1}.get(p["cleanliness"],2)
    req=bits_for_probability(_SCORE_BITS,0.99)
    excess=max(0.0,p["effective_search_bits"]-req)
    return (sig_bad, insufficient, p["candidate_dependent_blocks"], -p["first_affected_word"], clean_rank, excess, p["nonce_length"])


def recommendation(payload: bytes,bits:int,preserve:bool,prefer_throughput:bool=False) -> tuple[dict,list[dict]]:
    global _SCORE_BITS; _SCORE_BITS=bits
    sig=detect_signature(payload); ps=plans(payload,bits); ranked=sorted(ps,key=lambda x:score(x,sig,preserve,prefer_throughput));return ranked[0],ranked


def human(payload: bytes,bits:int,preserve:bool,prefer_throughput:bool=False) -> str:
    sig=detect_signature(payload); best,ranked=recommendation(payload,bits,preserve,prefer_throughput)
    req99=bits_for_probability(bits,.99);req999999=bits_for_probability(bits,.999999)
    lines=[]
    lines.append(f"signature: {sig}")
    lines.append(f"target prefix: {bits} bits")
    lines.append(f"search bits needed for 99% one-pass success: {req99:.2f}")
    lines.append(f"search bits needed for 99.9999% one-pass success: {req999999:.2f}")
    lines.append("")
    lines.append(f"RECOMMENDED: {best['name']}")
    lines.append(f"  candidate-dependent SHA-1 blocks: {best['candidate_dependent_blocks']}")
    lines.append(f"  nonce block/offset/first word: {best['nonce_block']} / {best['nonce_within_block']} / W{best['first_affected_word']}")
    lines.append(f"  effective search bits: {best['effective_search_bits']:.2f}")
    lines.append(f"  one-pass success probability: {best['success_probability']:.9f}")
    lines.append(f"  signature effect: {best['signature_effect']}")
    lines.append(f"  note: {best['note']}")
    lines.append("")
    lines.append("ranked candidates:")
    for p in ranked:
        lines.append(f"  {p['name']:<24} blocks={p['candidate_dependent_blocks']:<4} off={p['nonce_within_block']:<2} W{p['first_affected_word']:<2} bits={p['effective_search_bits']:.2f} p={p['success_probability']:.6f} {p['signature_effect']}")
    return "\n".join(lines)


def main() -> int:
    ap=argparse.ArgumentParser(description="Plan an efficiently searchable Git SHA-1 commit layout")
    ap.add_argument("commit_payload",nargs="?",help="raw commit payload file; omit when using --git-ref")
    ap.add_argument("--git-ref",metavar="REV",help="read raw commit payload with `git cat-file commit REV`")
    ap.add_argument("--repo",default=".",help="repository for --git-ref (default: current directory)")
    g=ap.add_mutually_exclusive_group()
    g.add_argument("--prefix-bits",type=int,choices=range(1,161),metavar="1..160",help="target prefix width; default 32")
    g.add_argument("--prefix",metavar="HEX",help="actual hexadecimal object-id prefix; planning uses its bit length")
    ap.add_argument("--preserve-signature",action="store_true",help="rank strategies that keep an existing signature valid first")
    ap.add_argument("--prefer-throughput",action="store_true",help="prefer binary layout over human-readable layout when SHA work is otherwise tied")
    ap.add_argument("--json",action="store_true")
    ap.add_argument("--emit",metavar="DIR",help="write the recommended payload template + plan.json")
    args=ap.parse_args()
    if bool(args.commit_payload)==bool(args.git_ref):ap.error("provide exactly one of commit_payload or --git-ref")
    if args.prefix:
        x=args.prefix.lower()
        if not 1<=len(x)<=40 or any(c not in '0123456789abcdef' for c in x):ap.error("--prefix must be 1..40 hexadecimal digits")
        bits=len(x)*4
    else:bits=args.prefix_bits or 32
    if args.git_ref:
        p=subprocess.run(['git','-C',args.repo,'cat-file','commit',args.git_ref],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if p.returncode:raise SystemExit(p.stderr.decode(errors='replace').strip() or 'git cat-file failed')
        payload=p.stdout
    else:payload=Path(args.commit_payload).read_bytes()
    sig=detect_signature(payload)
    if args.preserve_signature and sig in ('ssh','x509','other'):
        raise SystemExit(f"no signature-preserving nonce strategy is implemented for {sig} commit signatures; ordinary message mutation would invalidate the signature")
    best,ranked=recommendation(payload,bits,args.preserve_signature,args.prefer_throughput)
    result={"signature":sig,"target_bits":bits,"target_hex":args.prefix,"recommended":best,"plans":ranked}
    if sig!='none' and not args.preserve_signature:
        result['warning']='commit is signed; tail-message strategies invalidate the existing signature. Use --preserve-signature for supported PGP armor planning.'
    if args.json:print(json.dumps(result,indent=2))
    else:
        if result.get('warning'):print('WARNING:',result['warning'],'\n')
        print(human(payload,bits,args.preserve_signature,args.prefer_throughput))
    if args.emit:
        d=Path(args.emit);d.mkdir(parents=True,exist_ok=True)
        name=best['name']
        if name.startswith('printable-tail-'):
            n=int(name.rsplit('-',1)[1]);p,off,_=best_printable_tail(payload,label=b"Vanity: ",nonce_len=n)
        elif name=='binary-tail-4+1':
            p,off,_=append_aligned_tail(payload,label=b"Vanity-Binary: ",nonce_len=5,target_mod=48)
        elif name.startswith('pgp-armor-comment-'):
            n=int(name.rsplit('-',1)[1]);p,off,_=best_pgp_armor_comment(payload,n)
        else: raise AssertionError(name)
        (d/'commit-template.bin').write_bytes(p)
        (d/'plan.json').write_text(json.dumps(best,indent=2)+'\n')
        (d/'README.txt').write_text(f"Template strategy: {name}\nNonce starts at payload byte {off}. Replace exactly {best['nonce_length']} placeholder P bytes.\n")
    return 0
if __name__=='__main__': raise SystemExit(main())
