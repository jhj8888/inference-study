"""Reproduce one reviewed Pyre exercise issue without changing its source file."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
original = runpy.run_path(str(args.source))['my_softmax']
x = torch.tensor([[1000., 1001., 1002.]], dtype=torch.float64)
try:
    original(x)
except TypeError as error:
    result = {'original_result': 'TypeError', 'message': str(error)}
else:
    raise AssertionError('The old issue was not reproduced; the source may have changed.')
maximum = x.max(dim=-1, keepdim=True).values
y = (x - maximum).exp()
y = y / y.sum(dim=-1, keepdim=True)
torch.testing.assert_close(y, torch.softmax(x, dim=-1))
result.update({'torch': torch.__version__, 'input': x.tolist(), 'corrected_output': y.tolist(),
               'source_modified': False, 'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest()})
output = Path(__file__).resolve().parent / 'data/prior-softmax-check.json'
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('PASS: original TypeError reproduced; corrected values match torch.softmax.')
