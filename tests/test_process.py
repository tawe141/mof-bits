import pytest
from mofbits.process import *
from rdkit.DataStructs.cDataStructs import ExplicitBitVect

@pytest.fixture
def hkust():
    return "[Cu][Cu].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0"


@pytest.fixture
def list_of_mofids():
    return [
        "[Cu][Cu].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # Cu-BTC
        "[Cr][Cr].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # Cr-BTC
        "[Cr][Cu].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # Cu/Cr-BTC
        "[Mn][Mn].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # Mn-BTC
        "[Zn][Zn].[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # Zn-BTC
        "Cl[Mn][Mn]Cl.[O-]C(=O)c1cc(cc(c1)C(=O)[O-])C(=O)[O-] MOFid-v1.tbo.cat0",  # MnCl-BTC
        "[O-]C(=O)c1ccc(cc1)C(=O)[O-].[Zn][O]([Zn])([Zn])[Zn] MOFid-v1.pcu.cat0"  # MOF-5
    ]


@pytest.fixture
def featurizer(list_of_mofids):
    return MOFBits(list_of_mofids)


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


def test_mofbits_set_featurize(featurizer, list_of_mofids):
    processed_mofids = [process_mofid(i) for i in list_of_mofids]
    # ordering of unique entries is random, so can only test length
    assert len(featurizer._set_featurize(processed_mofids[0]['metal list'], 'metal list').GetOnBits()) == 1
    assert len(featurizer._set_featurize(processed_mofids[2]['metal list'], 'metal list').GetOnBits()) == 2
    assert len(featurizer._set_featurize(processed_mofids[0]['organic linkers'], 'organic linkers')) == 2048


def test_ohe(featurizer):
    assert len(featurizer._ohe('Zn', 'metal list').GetOnBits()) == 1
    with pytest.warns(UserWarning):
        assert len(featurizer._ohe('Mg', 'metal list').GetOnBits()) == 0


def test_get_bvs_from_mofid(featurizer, hkust):
    result = featurizer.get_bvs_from_mofid(hkust)
    assert len(result[0]) == 2048
    assert len(result[1]) == 2048
    assert len(result[2]) == 2
    assert len(result[3]) == 4
    assert len(result[4]) == 2048


def test_get_full_bv(featurizer, hkust):
    result = featurizer.get_full_bv(hkust)
    assert len(result) == 3*2048+2+4
