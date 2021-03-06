import re
from typing import Tuple, List, Union, Set, Callable
from rdkit import Chem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
import tqdm
import warnings
import os
from pymatgen.io.cif import CifParser
from pathlib import Path
import re
import sqlite3


# probably not the right way to do this, but...
# TODO: probably best to use importlib.resources here, but this works
_script_dir = os.path.dirname(__file__)
metals = open(_script_dir+'/metals.txt').read().splitlines()
halides = ['F', 'Cl', 'Br', 'I']


def canonicalize_smiles(smi: str) -> Union[None, str]:
    # TODO: I think there's an RDKit builtin function for doing this already; maybe use that instead
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None
    else:
        return Chem.MolToSmiles(mol)


def process_topology(s: str) -> Set[str]:
    query = re.compile('MOFid-v1.(.*).cat')
    result = query.search(s)
    if result is not None:
        return set(result.group(1).split(','))
    else:
        return None


def process_metal_and_organic(s: str) -> Tuple[Set[str], Set[str], Set[str], Set[str]]:
    metal_nodes = set()
    metal_node_types = set()
    metal_list = set()
    organic_linkers = set()
    items = s.split('.')
    bracket_query = re.compile('\[(.*?)\]')
    for i in items:
        has_metal = False

        results = bracket_query.findall(i)

        # search for metals
        for n in results:
            if n in metals:
                has_metal = True
                metal_list.add(n)
                metal_nodes.add(i)

        # add metal node type
        if has_metal:
            for m in metals:
                i = i.replace('[%s]' % m, '[M]')
            for h in halides:
                i = i.replace(h, '$')
            metal_node_types.add(i)
        else:
            organic_linkers.add(canonicalize_smiles(i))

    return metal_nodes, organic_linkers, metal_list, metal_node_types


def process_mofid(s: str) -> dict:
    """
    Returns five things in a tuple from a MOFid:
    1. set of metal nodes with explicit metals
    2. set of organic linkers
    3. set of topologies (usually only 1, but some have multiple?)
    4. set of metals found
    5. set of metal node motifs
    """
    s1, s2 = s.split(' ')
    metal_nodes, organic_linkers, metal_list, metal_node_types = process_metal_and_organic(s1)
    topology = process_topology(s2)
    #     return metal_nodes, organic_linkers, topology, metal_list, metal_node_types
    return {
        'metal nodes': metal_nodes,
        'organic linkers': organic_linkers,
        'topology': topology,
        'metal list': metal_list,
        'metal node types': metal_node_types
    }


def fp_smiles(s: str, **kwargs):
    mol = Chem.MolFromSmiles(s, sanitize=False)
    return Chem.RDKFingerprint(mol, **kwargs)


def metal_node_fp(s: str):
    if bool(re.match(r"^\[[A-Z][a-z]*\]$", s)):  # if matches to a single atom like [Fe]
        s += "[Ne]"  # tack on another atom so there's an edge in the molecular graph to fingerprint
    return fp_smiles(s, maxPath=3)


def metal_node_type_fp(s: str):
    # return fp_smiles(s.replace('[M]', '[Zn]').replace('$', 'Cl'), maxPath=3)
    return metal_node_fp(s.replace('[M]', '[Zn]').replace('$', 'Cl'))


def collect_set(list_of_sets) -> List[str]:
    return list(set().union(*[l for l in list_of_sets if l is not None]))


# def one_hot_fp(s: str, unique: List[str]) -> ExplicitBitVect:
#     fp = ExplicitBitVect(len(unique))
#     fp.SetBit(unique.index(s))
#     return fp


def union_bv(list_of_bv: List[ExplicitBitVect], bv=None) -> ExplicitBitVect:
    if len(list_of_bv) == 0:
        return bv
    if bv is None:
        bv = list_of_bv.pop()
        return union_bv(list_of_bv, bv)
    else:
        new_bv = list_of_bv.pop()
        return union_bv(list_of_bv, bv | new_bv)


def collective_fp(list_to_fp: Set[str], fingerprint_fn: Callable, **kwargs) -> ExplicitBitVect:
    return union_bv(
        [fingerprint_fn(i, **kwargs) for i in list_to_fp]
    )


def concatenate_bvs(list_of_bvs: List[ExplicitBitVect], current_bv=None):
    if len(list_of_bvs) == 0 and current_bv is None:
        raise RuntimeError('Tried to concatenate bitvects but no bitvects were given')
    elif len(list_of_bvs) == 0:
        return current_bv
    elif current_bv is None:
        current_bv = list_of_bvs.pop(0)
        return concatenate_bvs(list_of_bvs, current_bv)
    else:
        new_bv = ExplicitBitVect(len(current_bv) + len(list_of_bvs[0]))
        new_bv.SetBitsFromList(current_bv.GetOnBits())
        next_bv = list_of_bvs.pop(0)
        new_bv.SetBitsFromList([i+len(current_bv) for i in next_bv.GetOnBits()])
        return concatenate_bvs(list_of_bvs, new_bv)


class MOFBits:
    def __init__(self, list_of_mofids: List[str]):
        self.topology_bv_generator = TopologyBVGenerator()
        # these keys will go through a custom fingerprint operation; others will be one-hot encoded
        self._custom_fingerprint_funcs = {
            'metal node types': metal_node_type_fp,
            'metal nodes': metal_node_fp,
            'topology': self.topology_bv_generator.get_bv,
            'organic linkers': fp_smiles
        }
        processed_mofids = [process_mofid(i) for i in tqdm.tqdm(list_of_mofids, desc='Processing MOFids', unit='ids')]
        self._keys = collect_set([i.keys() for i in processed_mofids])  # TODO: is this ordered?
        self._uniques = {k: collect_set([i[k] for i in processed_mofids]) for k in self._keys if k not in self._custom_fingerprint_funcs.keys()}
        test_bv = self._get_bvs(processed_mofids[0])
        self.lengths = [len(i) for i in test_bv]

        # self._cache = dict(zip(list_of_mofids, [self._to_bitvec()))

    def _ohe(self, s: str, feature_name: str) -> ExplicitBitVect:
        fp = ExplicitBitVect(len(self._uniques[feature_name]))
        try:
            fp.SetBit(self._uniques[feature_name].index(s))
        except ValueError:
            warnings.warn('The string %s has not been seen in the provided dataset under the \"%s\" category' % (s, feature_name))
        return fp

    def _set_featurize(self, x: Set[str], feature_name: str) -> ExplicitBitVect:
        if feature_name in self._custom_fingerprint_funcs:
            return collective_fp(x, self._custom_fingerprint_funcs[feature_name])
        else:
            return collective_fp(x, self._ohe, feature_name=feature_name)

    def _get_bvs(self, x: dict) -> List[ExplicitBitVect]:
        # bvs = [self._set_featurize(v, k) for k, v in x.items()]
        bvs = []
        for i, (k, v) in enumerate(x.items()):
            if v == set() or v is None or v == {'*'}:
                bvs.append(ExplicitBitVect(self.lengths[i]))
            else:
                bvs.append(self._set_featurize(v, k))
        return bvs

    def get_bvs_from_mofid(self, mofid: str):
        return self._get_bvs(process_mofid(mofid))

    def get_full_bv(self, mofid: str):
        return concatenate_bvs(self.get_bvs_from_mofid(mofid))


class TopologyBVGenerator:
    def __init__(self):
        self.conn = sqlite3.connect(os.path.dirname(__file__) + '/template.db')
        self._cache = dict()

    @staticmethod
    def _cif_to_mol(cif: CifParser, topology_name: str):
        site_labels = cif._cif.data[topology_name].data['_atom_site_label']
        elements = [re.search(r'^[a-zA-Z]+', i).group(0) for i in site_labels]

        mol = Chem.RWMol()
        for element in elements:
            mol.AddAtom(Chem.Atom(element))
        edge_from = [
            site_labels.index(i)
            for i in
            cif._cif.data[topology_name].data['_geom_bond_atom_site_label_1']
        ]
        edge_to = [
            site_labels.index(i)
            for i in
            cif._cif.data[topology_name].data['_geom_bond_atom_site_label_2']
        ]

        bond_record = []
        for a, b in zip(edge_from, edge_to):
            if [a, b] not in bond_record and [b, a] not in bond_record:
                mol.AddBond(a, b)
                bond_record.append([a, b])

        return mol

    def _process_cif(self, top: str):
        """
        Takes a cif path and generates a fingerprint. If it cannot find the cif path, returns an empty bit vector
        """
        cif_str = self.conn.cursor().execute('select cif from topology where top=?', (top,)).fetchone()
        if cif_str is None:
            return ExplicitBitVect(2048)
        cif = CifParser.from_string(cif_str[0])
        mol = self._cif_to_mol(cif, top)
        fp = Chem.RDKFingerprint(mol)
        return fp

    def get_bv(self, topology: str):
        topology = ''.join([i for i in topology if i != '-'])

        if topology in ('UNKNOWN', 'TIMEOUT', 'ERROR'):
            return ExplicitBitVect(2048)
        elif topology in self._cache:
            return self._cache[topology]
        else:
            fp = self._process_cif(topology)
            self._cache[topology] = fp
            return fp
