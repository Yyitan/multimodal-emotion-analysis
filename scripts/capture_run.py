#!/usr/bin/env python3
import argparse, datetime, hashlib, json, os, platform, subprocess, sys
from pathlib import Path

def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--name',required=True); ap.add_argument('--config',type=Path); ap.add_argument('--output-dir',type=Path,required=True); ap.add_argument('command',nargs=argparse.REMAINDER)
 a=ap.parse_args(); out=a.output_dir/a.name; out.mkdir(parents=True,exist_ok=True)
 env={k:v for k,v in os.environ.items() if k in {'CUDA_VISIBLE_DEVICES','PYTHONHASHSEED','OMP_NUM_THREADS','MKL_NUM_THREADS'}}
 record={'name':a.name,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,'platform':platform.platform(),'command':a.command,'safe_environment':env}
 if a.config: record['config']={'path':a.config.name,'sha256':sha(a.config)}
 (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
 if not a.command: return
 proc=subprocess.run(a.command,cwd=Path(__file__).resolve().parents[1])
 record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); record['returncode']=proc.returncode
 (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
 raise SystemExit(proc.returncode)
if __name__=='__main__': main()
