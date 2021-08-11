# -*- coding: utf-8 -*-
#
#  Copyright 2021 Pavel Sidorov <pavel.o.sidorov@gmail.com>
#  This file is part of ChemInfoTools repository.
#
#  ChemInfoTools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.

from CGRtools import smiles, CGRContainer, MoleculeContainer
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Augmentor(BaseEstimator, TransformerMixin):
    """
    Augmentor class is a scikit-learn compatible transformer that calculates the fragment features 
    from molecules and Condensed Graphs of Reaction (CGR). The features are augmented substructures - 
    atom-centered fragments that take into account atom and its environment. Implementation-wise,
    this takes all atoms in the molecule/CGR, and builds topological neighborhood spheres around them.
    All atoms and bonds that are in a sphere of certain radius (1 bond, 2 bonds, etc) are taken into 
    the substructure. All such substructures are detected and stored as distinct features. The 
    substructures will keep any rings found within them. The value of the feature is the number of
    occurrence of such substructure in the given molecule.

    The parameters of the augmentor are the lower and the upper limits of the radius. By default,
    both are set to 0, which means only the count of atoms.
    Additionally, only_dynamic flag indicates of only fragments that contain a dynamic bond or atom 
    will be considered (only works in case of CGRs).
    """

    def __init__(self, lower:int=0, upper:int=0, only_dynamic:bool=False):
        self.feature_names = []
        self.lower = lower 
        self.upper = upper
        self.only_dynamic = only_dynamic
    
    def fit(self, X, y=None):
        """Fits the augmentor - finds all possible substructures in the given array of molecules/CGRs.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to train the augmentor. Collects all possible substructures.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        for i, mol in enumerate(X):
            for length in range(self.lower, self.upper):
                for atom in mol.atoms():
                    # deep is the radius of the neighborhood sphere in bonds
                    sub = str(mol.augmented_substructure([atom[0]], deep=length))
                    if sub not in self.feature_names:
                        # if dynamic_only is on, skip all non-dynamic fragments
                        if self.only_dynamic and ">" not in sub:
                            continue
                        self.feature_names.append(sub)
        return self
        
    def transform(self, X, y=None):
        """Transforms the given array of molecules/CGRs to a data frame with features and their values.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to transform to feature table using trained feature list.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        table = pd.DataFrame(columns=self.feature_names)
        for i, mol in enumerate(X):
            table.loc[len(table)] = 0
            for sub in self.feature_names:
                # if CGRs are used, the transformation of the substructure to the CGRcontainer is needed
                if type(mol) == CGRContainer:
                    mapping = list(CGRContainer().compose(smiles(sub)).get_mapping(mol, optimize=False))
                else:
                    mapping = list(smiles(sub).get_mapping(mol, optimize=False))
                # mapping is the list of all possible substructure mappings into the given molecule/CGR
                table.loc[i,sub] = len(mapping)
        return table
    
    def get_feature_names(self):
        """Returns the list of features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names