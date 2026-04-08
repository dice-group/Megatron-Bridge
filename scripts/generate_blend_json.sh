#!/bin/bash

# Generate blend.json for a fineweb data directory.
# Usage: bash scripts/generate_blend_json.sh [DATA_DIR]

DATA_DIR="${1:-/scratch/hpc-prf-merlin/luke/Megatron-Bridge/data/fineweb}"

python3 -c "
import json, os

output_dir = '${DATA_DIR}'
blend = {
    'train': ['1.0', os.path.join(output_dir, 'fineweb_train_text_document')],
    'valid': ['1.0', os.path.join(output_dir, 'fineweb_valid_text_document')],
    'test':  ['1.0', os.path.join(output_dir, 'fineweb_test_text_document')],
}
path = os.path.join(output_dir, 'blend.json')
with open(path, 'w') as f:
    json.dump(blend, f, indent=2)
print(f'Written: {path}')
print(json.dumps(blend, indent=2))
"
