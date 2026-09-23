#!/usr/bin/env python3
import argparse, hashlib, importlib.util, json, os, shutil, sys, tempfile
from pathlib import Path

REQUIRED_IMPORTS = ['torch','numpy','pandas','sklearn','scipy','yaml','openpyxl','matplotlib','librosa','soundfile','cv2','mediapipe']
BERT_REQUIRED = ['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','vocab.txt']


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--min-free-gib',type=float,default=20.0)
    args=ap.parse_args(); root=args.project_root.resolve(); data=args.data_root.resolve()
    checks={}; errors=[]
    checks['python']=sys.version.split()[0]
    missing=[m for m in REQUIRED_IMPORTS if importlib.util.find_spec(m) is None]
    checks['missing_imports']=missing
    if missing: errors.append(f'missing imports: {missing}')
    if not data.is_dir(): errors.append(f'data root missing: {data}')
    checks['data_root_exists']=data.is_dir()
    free=shutil.disk_usage(root).free/(1024**3); checks['free_gib']=round(free,3)
    if free<args.min_free_gib: errors.append(f'free disk {free:.2f} GiB < {args.min_free_gib:.2f} GiB')
    bert=root/'third_party'/'bert-base-uncased'; missing_bert=[x for x in BERT_REQUIRED if not (bert/x).is_file()]
    checks['missing_bert_files']=missing_bert
    if missing_bert: errors.append(f'missing BERT files: {missing_bert}')
    manifest=bert/'manifest.json'
    if manifest.is_file():
        info=json.loads(manifest.read_text())['files']; bad=[]
        for name,meta in info.items():
            p=bert/name
            if not p.is_file() or sha256(p)!=meta['sha256']: bad.append(name)
        checks['bert_hash_mismatches']=bad
        if bad: errors.append(f'BERT hash mismatch: {bad}')
    else: errors.append('BERT manifest missing')
    for d in [root/'runs',root/'artifacts',root/'cache'/'tmp']:
        d.mkdir(parents=True,exist_ok=True)
        try:
            with tempfile.NamedTemporaryFile(dir=d,delete=True) as f: f.write(b'ok'); f.flush()
        except OSError as e: errors.append(f'not writable: {d}: {e}')
    try:
        import torch
        checks['torch']=torch.__version__; checks['cuda_available']=torch.cuda.is_available(); checks['cuda_device_count']=torch.cuda.device_count()
        if not torch.cuda.is_available(): errors.append('CUDA unavailable')
        else:
            idx=torch.cuda.device_count()-1
            x=torch.tensor([1.0,2.0],device=f'cuda:{idx}',requires_grad=True); y=(x*x).sum(); y.backward()
            checks['gpu']=torch.cuda.get_device_name(idx); checks['cuda_smoke']=bool(x.grad.tolist()==[2.0,4.0])
    except Exception as e: errors.append(f'CUDA smoke failed: {e!r}')
    checks['errors']=errors; checks['ok']=not errors
    print(json.dumps(checks,ensure_ascii=False,indent=2))
    return 1 if errors else 0

if __name__=='__main__': raise SystemExit(main())
