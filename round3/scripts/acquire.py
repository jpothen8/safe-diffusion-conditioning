"""Acquire public primary resources with resolved revisions and content hashes."""
import hashlib,json,sys,urllib.request,urllib.parse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def get(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'simulation-research-agent'}),timeout=90) as r:return r.read()
def save(url,path):
    data=get(url);target=ROOT/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    return {'url':url,'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
records=[]
for url,path in [('https://arxiv.org/html/2606.21014v1','records/bayesfp.html'),
                 ('https://arxiv.org/pdf/2606.21014v1','records/bayesfp.pdf'),
                 ('https://api.github.com/search/repositories?q=BayesFP','records/bayesfp_repository_search.json')]:
    try:records.append(save(url,path))
    except Exception as e:records.append({'url':url,'error':str(e)})
for repo in ['farama-minari/Walker2d-v5-SAC-expert','farama-minari/Walker2d-v5-SAC-medium',
             'farama-minari/HalfCheetah-v5-SAC-expert','farama-minari/HalfCheetah-v5-SAC-medium',
             'sb3/td3-MountainCarContinuous-v0']:
    try:
        info=json.loads(get('https://huggingface.co/api/models/'+repo));rev=info['sha'];name=repo.split('/')[-1]
        (ROOT/f'records/{name}_metadata.json').write_text(json.dumps(info,indent=2))
        files=[x['rfilename'] for x in info['siblings']]
        zips=sorted(f for f in files if f.endswith('.zip') and 'metrics' not in f.lower())
        # Checkpoints may have duplicate capitalization; record one explicit file.
        # Evaluation archives are not policies and must never overwrite them.
        for f in (['README.md'] if 'README.md' in files else [])+zips[:1]:
            path=f'assets/{name}.zip' if f.endswith('.zip') else f'records/{name}_README.md'
            record=save(f'https://huggingface.co/{repo}/resolve/{rev}/{urllib.parse.quote(f)}',path)
            record.update(repo=repo,revision=rev,license=info.get('cardData',{}).get('license','Not declared in inspected metadata'))
            records.append(record)
    except Exception as e:records.append({'repo':repo,'error':str(e)})
(ROOT/'records/assets.json').write_text(json.dumps(records,indent=2))
print(json.dumps(records,indent=2))
