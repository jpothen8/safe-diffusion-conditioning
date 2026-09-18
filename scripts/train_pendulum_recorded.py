"""Reproduce the recorded training arithmetic without overwriting a frozen model."""
import argparse,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import core
import pendulum_train

parser=argparse.ArgumentParser()
parser.add_argument('--output-root',type=Path,default=ROOT)
args=parser.parse_args();destination=args.output_root.resolve()
checkpoint=destination/'assets/pendulum_diffusion.pt'
if checkpoint.exists():raise SystemExit('Refusing to overwrite an existing model; choose a fresh --output-root.')
(destination/'assets').mkdir(parents=True,exist_ok=True)
(destination/'results/pendulum').mkdir(parents=True,exist_ok=True)
source=ROOT/'assets/td3-Pendulum-v1.zip';target=destination/'assets/td3-Pendulum-v1.zip'
if source!=target:shutil.copyfile(source,target)
# This explicit historical adapter exactly regenerates the saved demonstrations.
# Corrected float64 Gym-matching dynamics are used for reported evaluation.
spec=importlib.util.spec_from_file_location('recorded_pendulum',ROOT/'records/pendulum_initial_dynamics.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
pendulum_train.pd=module;core.ROOT=destination
pendulum_train.train()
print(json.dumps({'checkpoint':str(checkpoint),'sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
