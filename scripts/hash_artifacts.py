#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

EXCLUDE_PARTS={'cache','env','runs','.pytest_cache','__pycache__','third_party'}

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); ap.add_argument('--output',type=Path)
 args=ap.parse_args(); root=args.root.resolve(); rows=[]
 for p in sorted(root.rglob('*')):
  if not p.is_file(): continue
  rel=p.relative_to(root)
  if any(x in EXCLUDE_PARTS for x in rel.parts): continue
  if args.output and p.resolve()==args.output.resolve(): continue
  rows.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)})
 payload={'schema_version':1,'root_label':'competition_workspace','files':rows,'total_bytes':sum(x['bytes'] for x in rows)}
 text=json.dumps(payload,ensure_ascii=False,indent=2)
 if args.output: args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(text,encoding='utf-8')
 else: print(text)

if __name__=='__main__': main()
