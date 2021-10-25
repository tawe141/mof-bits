import pytest
from mofbits.process import *

@pytest.fixture
def hkust():
    return "[Cu][Cu].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0"


def test_process_topology(hkust):
    assert process_topology(hkust) == {'tbo'}


def test_process_metal_and_organic(hkust):
    metal_organic = hkust.split(' ')[0]
    assert process_metal_and_organic(metal_organic) == (
        {"[Cu][Cu]"},
        {canonicalize_smiles("[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-]")},
        {'Cu'},
        {'[M][M]'}
    )


def test_process_mofid(hkust):
    assert process_mofid(hkust) == {
        'metal nodes': {'[Cu][Cu]'},
        'organic linkers': {canonicalize_smiles("[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-]")},
        'topology': {'tbo'},
        'metal list': {'Cu'},
        'metal node types': {'[M][M]'}
    }