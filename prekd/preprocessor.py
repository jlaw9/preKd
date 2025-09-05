
from typing import Any, Dict, Optional
from inspect import getmembers
from nfp.preprocessing.tokenizer import Tokenizer

import numpy as np
import pandas as pd
import networkx as nx
import tensorflow as tf
from nfp.preprocessing.mol_preprocessor import SmilesPreprocessor
import rdkit
#from rdkit.Chem.Descriptors import MolWt


class Preprocessor(SmilesPreprocessor):
    """Preprocessor with pandas df rows as inputs.

    :param smiles_col: Column name for SMILES strings.
    :param feat_cols: List of columns to use as global features.

    """

    def __init__(self, atom_features, bond_features, smiles_col="smiles", feat_cols=None):
        super(Preprocessor, self).__init__(atom_features, bond_features)
        self.smiles_col = smiles_col
        self.feat_cols = feat_cols
        if feat_cols is None:
            # use a default set of feature columns
            self.feat_cols = []

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        nx_graph = super().create_nx_graph(row[self.smiles_col], **kwargs)
        # add the global features
        for col in self.feat_cols:
            # val = row[col].fillna(0)
            val = row[col]
            nx_graph.graph[col] = val
        return nx_graph

    def get_graph_features(self, graph_data: dict):
        feature_vec = np.asarray([graph_data[col] for col in self.feat_cols])
        return {"global": feature_vec}

    @property
    def output_signature(self):
        signature = super().output_signature
        # for col in self.feat_cols:
        signature["global"] = tf.TensorSpec(shape=tf.TensorShape([None]), dtype="float32")
        return signature

    @property
    def padding_values(self):
        padding_values = super().padding_values
        # for col in self.feat_cols:
        padding_values["global"] = tf.constant(0, dtype=tf.float16)
        return padding_values


class SolventFeaturesPreprocessor(SmilesPreprocessor):
    """Preprocessor that includes the solvent molecules in the same graph
    The solvent ratios are included as node and edge weights.
    Can also include extra solvent features as node and edge features.

    """

    def __init__(self, atom_features, bond_features, compound_col="rdkit_canonical_smiles",
                 solvent_cols=None,
                 compound_feature_cols=None,
                 df_solvent_features=None,
                 solvent_df_smiles_col="smiles", 
                 solvent_feature_cols=None,
                 ):
        super().__init__(atom_features, bond_features)
        self.compound_col = compound_col
        self.solvent_cols = solvent_cols
        self.compound_feature_cols = compound_feature_cols or []  # Columns for molecular descriptors

        self.df_solvent_features = df_solvent_features  # DataFrame with solvent descriptors
        self.solvent_df_smiles_col = solvent_df_smiles_col
        self.solvent_feature_cols = solvent_feature_cols or []  # Columns for solvent descriptors
        # include the ratio as a feature 
        self.num_solv_feat_cols = len(self.solvent_feature_cols) + 1 if solvent_feature_cols is not None else 0
        if self.df_solvent_features is not None:
            assert self.solvent_df_smiles_col in self.df_solvent_features.columns, \
            "'smiles' column not found in solvent feature dataframe"
            for col in self.solvent_feature_cols:
                assert col in self.df_solvent_features.columns, f"Solvent feature column not found: {col}"
            for solvent_col in self.solvent_cols:
                assert solvent_col in self.df_solvent_features.index, f"Solvent {solvent_col} not found in solvent feature dataframe"

    def create_weighted_nx_graph(self, 
                                 smiles: str,
                                 weight: float = 1.0,
                                 idx_offset: int = 0,
                                 **kwargs) -> nx.DiGraph:
        """Create a weighted directed graph from an RDKit molecule.
        The atom index can be offset to allow for multiple molecules in the same graph.
        """
        mol = rdkit.Chem.MolFromSmiles(smiles)
        if self.explicit_hs:
            mol = rdkit.Chem.AddHs(mol)
        g = nx.Graph(mol=mol)
        g.add_nodes_from(((atom.GetIdx() + idx_offset, 
                           {"atom": atom, "weight": weight}) 
                           for atom in mol.GetAtoms()))
        g.add_edges_from(
            (
                (bond.GetBeginAtomIdx() + idx_offset, 
                 bond.GetEndAtomIdx() + idx_offset, 
                 {"bond": bond, "weight": weight})
                for bond in mol.GetBonds()
            )
        )
        return nx.DiGraph(g)

    def create_nx_graph(self, row: pd.Series, **kwargs) -> nx.DiGraph:
        combined_graph = self.create_weighted_nx_graph(row[self.compound_col], weight=1, **kwargs)

        # add the global features
        for feature_col in self.compound_feature_cols:
            assert feature_col in row, f"Feature column {feature_col} not found in row: {row}"
            combined_graph.graph[feature_col] = row[feature_col].fillna(0)

        # Include solvent features
        for solvent_col in self.solvent_cols:
            assert solvent_col in row, f"Solvent column {solvent_col} not found in row: {row}"
            ratio = row[solvent_col]
            if ratio == 0:
                continue
            solvent_smiles = self.get_solvent_smiles(solvent_col)
            # the node indexes are offset so they can all be combined into one (disconnected) graph
            solvent_graph = self.create_weighted_nx_graph(solvent_smiles, 
                                                          weight=ratio, 
                                                          idx_offset=combined_graph.number_of_nodes(),
                                                          **kwargs)
            if self.num_solv_feat_cols > 0:
                solvent_feature_vec = self.get_solvent_features(solvent_col, ratio=ratio)
                # Add solvent features to the graph, likely by including them as node (and bond?) features
                # TODO need to line up the feature vectors for the compound and solvent graphs
                for n, atom_data in solvent_graph.nodes(data=True):
                    atom_data["feature_vec"] = solvent_feature_vec
                for _, _, bond_data in solvent_graph.edges(data=True):
                    bond_data["feature_vec"] = solvent_feature_vec

            # add the solvent graphs to the compound graph
            combined_graph = nx.compose(combined_graph, solvent_graph)

        # assert the solvent ratios add up to 1
        assert np.isclose(sum(row[self.solvent_cols]), 1), \
            f"Solvent ratios do not add up to 1: {row[self.solvent_cols]}"

        return combined_graph

    def get_solvent_features(self, solvent_name, ratio=None):
        """ Get the feature vector for a solvent by name. """
        feature_vec = self.df_solvent_features.loc[solvent_name][self.solvent_feature_cols]
        # Fill NaN values with 0
        # raises a bunch of warnings so just fillna beforehand
        feature_vec = feature_vec.astype(float).values
        if ratio is not None:
            # add the ratio to the feature vector
            feature_vec = np.append(feature_vec, ratio)
        return feature_vec

    def get_solvent_feature(self, solvent_name, feature_name):
        """Get a specific feature value for a solvent."""
        value = self.df_solvent_features.loc[solvent_name][feature_name]
        if np.isnan(value):
            value = 0
        return value

    def get_solvent_smiles(self, solvent_name):
        """Get SMILES representation for a solvent by name.
        Will check the solvent feature dataframe, and fall back on a mapping if not found.
        """
        if self.df_solvent_features is not None and solvent_name in self.df_solvent_features.index:
            return self.df_solvent_features.loc[solvent_name][self.solvent_df_smiles_col]
        else:
            solvent_smiles_map = {
                "water": "O",
                "ethyl acetate": "CC(=O)OCC",
                "ethanol": "CCO",
                "methanol": "CO",
                "hexane": "CCCCCC",
                "chloroform": "ClC(Cl)Cl",
                "petroleum ether": "CCCCC",
                "acetonitrile": "CC#N",
                "heptane": "CCCCCCC",
                "acetone": "CC(=O)C",
                "carbon tetrachloride": "ClC(Cl)(Cl)Cl",
                "dichloromethane": "ClCCl",
                "butanol": "CCCCO",
                "methyl tertiary butyl ether": "CC(C)(C)OC",
                "isopropanol": "CC(C)O",
                # Add more solvents as needed
            }
            assert solvent_name in solvent_smiles_map, f"Unknown solvent: {solvent_name}"
            return solvent_smiles_map[solvent_name]

    @property
    def output_signature(self):
        signature = super().output_signature
        signature["atom_weight"] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
        signature["bond_weight"] = tf.TensorSpec(shape=(None,), dtype=tf.float32)
        signature["atom_feature_vec"] = tf.TensorSpec(shape=(None,self.num_solv_feat_cols), dtype=tf.float32)
        signature["bond_feature_vec"] = tf.TensorSpec(shape=(None,self.num_solv_feat_cols), dtype=tf.float32)
        return signature

    @property
    def padding_values(self):
        padding_values = super().padding_values
        padding_values["atom_weight"] = tf.constant(0, dtype=tf.float16)
        padding_values["bond_weight"] = tf.constant(0, dtype=tf.float16)
        padding_values["atom_feature_vec"] = tf.constant(0, dtype=tf.float16)
        padding_values["bond_feature_vec"] = tf.constant(0, dtype=tf.float16)
        return padding_values
    
    def get_edge_weights(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_feature_matrix = np.zeros(max_num_edges, dtype=np.float32)
        for n, (_, _, bond_dict) in enumerate(edge_data):
            bond_feature_matrix[n] = bond_dict["weight"] 
        return {"bond_weight": bond_feature_matrix}

    def get_node_weights(
        self, node_data: list, max_num_nodes
    ) -> Dict[str, np.ndarray]:
        atom_feature_matrix = np.zeros(max_num_nodes, dtype=np.float32)
        for n, atom_dict in node_data:
            atom_feature_matrix[n] = atom_dict["weight"]
        return {"atom_weight": atom_feature_matrix}
    
    def get_edge_extra_features(
        self, edge_data: list, max_num_edges
    ) -> Dict[str, np.ndarray]:
        bond_feature_matrix = np.zeros((max_num_edges, self.num_solv_feat_cols), dtype=np.float32)
        for n, (_, _, bond_dict) in enumerate(edge_data):
            if "feature_vec" in bond_dict:
                bond_feature_matrix[n] = bond_dict["feature_vec"] 
            else:
                bond_feature_matrix[n] = np.zeros(self.num_solv_feat_cols, dtype=np.float32)
                # The last feature is the ratio. Give a ratio of 1 for the compound
                bond_feature_matrix[n][-1] = 1.0
        return {"bond_feature_vec": bond_feature_matrix}

    def get_node_extra_features(
        self, node_data: list, max_num_nodes
    ) -> Dict[str, np.ndarray]: 
        atom_feature_matrix = np.zeros((max_num_nodes, self.num_solv_feat_cols), dtype=np.float32)
        for n, atom_dict in node_data:
            if "feature_vec" in atom_dict:
                atom_feature_matrix[n] = atom_dict["feature_vec"]
            else:
                atom_feature_matrix[n] = np.zeros(self.num_solv_feat_cols, dtype=np.float32)
        return {"atom_feature_vec": atom_feature_matrix}

    def __call__(
        self,
        structure: Any,
        *args,
        train: bool = False,
        max_num_nodes: Optional[int] = None,
        max_num_edges: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Convert an input graph structure into a featurized set of node, edge,
         and graph-level features.

        Parameters
        ----------
        structure
            An input graph structure (i.e., molecule, crystal, etc.)
        train
            A training flag passed to `Tokenizer` member attributes
        max_num_nodes
            A size attribute passed to `get_node_features`, defaults to the
            number of nodes in the current graph
        max_num_edges
            A size attribute passed to `get_edge_features`, defaults to the
            number of edges in the current graph
        kwargs
            Additional features or parameters passed to `construct_nx_graph`

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary of key, array pairs as a single sample.
        """
        nx_graph = self.create_nx_graph(structure, *args, **kwargs)

        max_num_edges = len(nx_graph.edges) if max_num_edges is None else max_num_edges
        assert (
            len(nx_graph.edges) <= max_num_edges
        ), "max_num_edges too small for given input"

        max_num_nodes = len(nx_graph.nodes) if max_num_nodes is None else max_num_nodes
        assert (
            len(nx_graph.nodes) <= max_num_nodes
        ), "max_num_nodes too small for given input"

        # Make sure that Tokenizer classes are correctly initialized
        for _, tokenizer in getmembers(self, lambda x: type(x) == Tokenizer):
            tokenizer.train = train

        node_features = self.get_node_features(nx_graph.nodes(data=True), max_num_nodes)
        edge_features = self.get_edge_features(nx_graph.edges(data=True), max_num_edges)
        node_weights = self.get_node_weights(nx_graph.nodes(data=True), max_num_nodes)
        edge_weights = self.get_edge_weights(nx_graph.edges(data=True), max_num_edges)
        connectivity = self.get_connectivity(nx_graph, max_num_edges)
        feature_matrices = {**node_features, **edge_features, 
                             **node_weights, **edge_weights, 
                             **connectivity}
        if len(self.compound_feature_cols) > 0:
            # add the global features
            graph_features = self.get_graph_features(nx_graph.graph)
            feature_matrices = {**feature_matrices, **graph_features}
        if self.num_solv_feat_cols > 0:
            # add the solvent features
            # add the extra features to both nodes and edges so the # features stays the same for both
            node_extra_features = self.get_node_extra_features(nx_graph.nodes(data=True), max_num_nodes)
            edge_extra_features = self.get_edge_extra_features(nx_graph.edges(data=True), max_num_edges)
            # check to make sure the extra features are the same length 
            # since they are both used in the global attention mechanism
            num_extra_node_features = node_extra_features["atom_feature_vec"].shape[1] 
            num_extra_edge_features = edge_extra_features["bond_feature_vec"].shape[1]
            assert num_extra_node_features == num_extra_edge_features, \
                f"Node and edge extra features have different lengths: {num_extra_node_features} vs {num_extra_edge_features}"
            feature_matrices = {**feature_matrices, **node_extra_features, **edge_extra_features}

        return feature_matrices

