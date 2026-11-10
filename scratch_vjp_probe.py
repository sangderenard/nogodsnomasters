from pathlib import Path
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import c_backend_repository_ssa_reference

module, _, _ = lower_ast_source_to_ssa(
    '''
from dataclasses import dataclass
from typing import Mapping
import torch
@dataclass(frozen=True)
class Reading:
    voltage_v: Mapping[str, torch.Tensor]
def sample(reading: Reading, key: str):
    selected = reading.voltage_v[key]
    return selected.real.sum()
''',
    'sample', name='keyed_tensor_region',
    extraction_contract=Path('extraction_contracts/program_extraction.yaml').resolve(),
    tensor_ssa_reference=c_backend_repository_ssa_reference(),
    progress=lambda _message: None,
)
for function in module.functions.values():
    if not (function.metadata or {}).get('source_region_integral'):
        continue
    print('REGION', function.name, function.metadata)
    table = module.tensor_tables.get(function.name)
    for formal in function.args:
        print('FORMAL', formal.id, formal.dtype, formal.shape, formal.accounting)
        print('DESCRIPTOR', table.by_id(formal.id) if table else None)
