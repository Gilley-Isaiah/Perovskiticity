"""
The Perovskiticity program is designed to aid in the processing of structural data (in the form of a CIF,
etc.) of perovskites and related crystal structures.
Copyright (C) 2023 Isaiah Gilley, Mercouri Kanatzidis, Edward Sargent, Northwestern University
Contact: Isaiah Gilley, gilley@u.northwestern.edu, isaiahwgilley@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import time
import os
import math
import numpy
import scipy.spatial._qhull
from scipy.spatial import ConvexHull
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.dimensionality import get_dimensionality_larsen
from pymatgen.analysis.local_env import JmolNN
from itertools import combinations
from ccdc.diagram import DiagramGenerator
from ccdc.molecule import Molecule
from ccdc import search
import ccdc

numpy.set_printoptions(suppress=True)

jmol_radii =   {'Ac': 1.88, 'Ag': 1.59, 'Al': 1.35, 'Am': 1.51, 'Ar': 1.57, 'As': 1.21, 'At': 1.7, 'Au': 1.5, 'B': 0.83,
                'Ba': 1.34, 'Be': 0.35, 'Bh': 1.6, 'Bi': 1.54, 'Bk': 1.5, 'Br': 1.21, 'C': 0.68, 'Ca': 0.99, 'Cd': 1.69,
                'Ce': 1.83, 'Cf': 1.5, 'Cl': 0.99, 'Cm': 1.5, 'Co': 1.33, 'Cr': 1.35, 'Cs': 1.67, 'Cu': 1.52, 'Db': 1.6,
                'Dy': 1.75, 'Er': 1.73, 'Es': 1.5, 'Eu': 1.99, 'F': 0.64, 'Fe': 1.34, 'Fm': 1.5, 'Fr': 2, 'Ga': 1.22,
                'Gd': 1.79, 'Ge': 1.17, 'H': 0.23, 'He': 0.93, 'Hf': 1.57, 'Hg': 1.7, 'Ho': 1.74, 'Hs': 1.6, 'I': 1.4,
                'In': 1.63, 'Ir': 1.32, 'K': 1.33, 'Kr': 1.91, 'La': 1.87, 'Li': 0.68, 'Lr': 1.5, 'Lu': 1.72, 'Md': 1.5,
                'Mg': 1.1, 'Mn': 1.35, 'Mo': 1.47, 'Mt': 1.6, 'N': 0.68, 'Na': 0.97, 'Nb': 1.48, 'Nd': 1.81, 'Ne': 1.12,
                'Ni': 1.5, 'No': 1.5, 'Np': 1.55, 'O': 0.68, 'Os': 1.37, 'P': 0.75, 'Pa': 1.61, 'Pb': 1.54, 'Pd': 1.5,
                'Pm': 1.8, 'Po': 1.68, 'Pr': 1.82, 'Pt': 1.5, 'Pu': 1.53, 'Ra': 1.9, 'Rb': 1.47, 'Re': 1.35, 'Rf': 1.6,
                'Rh': 1.45, 'Rn': 2.4, 'Ru': 1.4, 'S': 1.02, 'Sb': 1.46, 'Sc': 1.44, 'Se': 1.22, 'Sg': 1.6, 'Si': 1.2,
                'Sm': 1.8, 'Sn': 1.46, 'Sr': 1.12, 'Ta': 1.43, 'Tb': 1.76, 'Tc': 1.35, 'Te': 1.47, 'Th': 1.79, 'Ti': 1.47,
                'Tl': 1.55, 'Tm': 1.72, 'U': 1.58, 'V': 1.33, 'W': 1.37, 'Xe': 1.98, 'Y': 1.78, 'Yb': 1.94, 'Zn': 1.45,
                'Zr': 1.56}

jmol_radii_ionic = {'H-': 1.54, 'Li+': 0.68, 'Be+': 0.44, 'Be2+': 0.35, 'B+': 0.35, 'B3+': 0.23, 'C4-': 2.6, 'C4+': 0.16,
                    'N3-': 1.71, 'N+': 0.25, 'N3+': 0.16, 'N5+': 0.13, 'O2-': 1.32, 'O-': 1.76, 'O+': 0.22, 'O6+': 0.09,
                    'F-': 1.33, 'F7+': 0.08, 'Ne+': 1.12, 'Na+': 0.97, 'Mg+': 0.82, 'Mg2+': 0.66, 'Al3+': 0.51,
                    'Si4-': 2.71, 'Si-': 3.84, 'Si+': 0.65, 'Si4+': 0.42, 'P3-': 2.12, 'P3+': 0.44, 'P5+': 0.35,
                    'S2-': 1.84, 'S2+': 2.19, 'S4+': 0.37, 'S6+': 0.3, 'Cl-': 1.81, 'Cl5+': 0.34, 'Cl7+': 0.27,
                    'Ar+': 1.54, 'K+': 1.33, 'Ca+': 1.18, 'Ca2+': 0.99, 'Sc3+': 0.732, 'Ti+': 0.96, 'Ti2+': 0.94,
                    'Ti3+': 0.76, 'Ti4+': 0.68, 'V2+': 0.88, 'V3+': 0.74, 'V4+': 0.63, 'V5+': 0.59, 'Cr+': 0.81,
                    'Cr2+': 0.89, 'Cr3+': 0.63, 'Cr6+': 0.52, 'Mn2+': 0.8, 'Mn3+': 0.66, 'Mn4+': 0.6, 'Mn7+': 0.46,
                    'Fe2+': 0.74, 'Fe3+': 0.64, 'Co2+': 0.72, 'Co3+': 0.63, 'Ni2+': 0.69, 'Cu+': 0.96, 'Cu2+': 0.72,
                    'Zn+': 0.88, 'Zn2+': 0.74, 'Ga+': 0.81, 'Ga3+': 0.62, 'Ge4-': 2.72, 'Ge2+': 0.73, 'Ge4+': 0.53,
                    'As3-': 2.22, 'As3+': 0.58, 'As5+': 0.46, 'Se2-': 1.91, 'Se-': 2.32, 'Se+': 0.66, 'Se4+': 0.5,
                    'Se6+': 0.42, 'Br-': 1.96, 'Br5+': 0.47, 'Br7+': 0.39, 'Rb+': 1.47, 'Sr2+': 1.12, 'Y3+': 0.893,
                    'Zr+': 1.09, 'Zr4+': 0.79, 'Nb+': 1.0, 'Nb4+': 0.74, 'Nb5+': 0.69, 'Mo+': 0.93, 'Mo4+': 0.7,
                    'Mo6+': 0.62, 'Tc7+': 0.979, 'Ru4+': 0.67, 'Rh3+': 0.68, 'Pd2+': 0.8, 'Pd4+': 0.65, 'Ag+': 1.26,
                    'Ag2+': 0.89, 'Cd+': 1.14, 'Cd2+': 0.97, 'In3+': 0.81, 'Sn4-': 2.94, 'Sn-': 3.7, 'Sn2+': 0.93,
                    'Sn4+': 0.71, 'Sb3-': 2.45, 'Sb3+': 0.76, 'Sb5+': 0.62, 'Te2-': 2.11, 'Te-': 2.5, 'Te+': 0.82,
                    'Te4+': 0.7, 'Te6+': 0.56, 'I-': 2.2, 'I5+': 0.62, 'I7+': 0.5, 'Cs+': 1.67, 'Ba+': 1.53,
                    'Ba2+': 1.34, 'La+': 1.39, 'La3+': 1.016, 'Ce+': 1.27, 'Ce3+': 1.034, 'Ce4+': 0.92, 'Pr3+': 1.013,
                    'Pr4+': 0.9, 'Nd3+': 0.995, 'Pm3+': 0.979, 'Sm3+': 0.964, 'Eu2+': 1.09, 'Eu3+': 0.95, 'Gd3+': 0.938,
                    'Tb3+': 0.923, 'Tb4+': 0.84, 'Dy3+': 0.908, 'Ho3+': 0.894, 'Er3+': 0.881, 'Tm3+': 0.87, 'Yb2+': 0.93,
                    'Yb3+': 0.858, 'Lu3+': 0.85, 'Hf4+': 0.78, 'Ta5+': 0.68, 'W4+': 0.7, 'W6+': 0.62, 'Re4+': 0.72,
                    'Re7+': 0.56, 'Os4+': 0.88, 'Os6+': 0.69, 'Ir4+': 0.68, 'Pt2+': 0.8, 'Pt4+': 0.65, 'Au+': 1.37,
                    'Au3+': 0.85, 'Hg+': 1.27, 'Hg2+': 1.1, 'Tl+': 1.47, 'Tl3+': 0.95, 'Pb2+': 1.2, 'Pb4+': 0.84,
                    'Bi+': 0.98, 'Bi3+': 0.96, 'Bi5+': 0.74, 'Po6+': 0.67, 'At7+': 0.62, 'Fr+': 1.8, 'Ra2+': 1.43,
                    'Ac3+': 1.18, 'Th4+': 1.02, 'Pa3+': 1.13, 'Pa4+': 0.98, 'Pa5+': 0.89, 'U4+': 0.97, 'U6+': 0.8,
                    'Np3+': 1.1, 'Np4+': 0.95, 'Np7+': 0.71, 'Pu3+': 1.08, 'Pu4+': 0.93, 'Am3+': 1.07, 'Am4+': 0.92}


# Special exceptions to jmol_radii
bond_lengths =         {'Ge': 3.44, 'Sn': 3.44, 'Pb': 3.05,
                        'Ag': 2.90, 'In': 2.62, 'Tl': 3.51, 'As': 2.34, 'Sb': 2.57, 'Bi': 2.91,
                        'F': 0.5, 'Cl': 0.8, 'Br': 0.95, 'I': 1.25}

# Assumption: Pb, Ag, In, Tl will never be outside of their usual oxidation state (+2, +1, +3, +1, respectively)
bond_lengths_oxidized = {'Ge': 1.5, 'Sn': 1.85, 'Pb':3.05,
                         'Ag': 2.90, 'In':2.62, 'Tl':3.51, 'As':1.65, 'Sb':1.65, 'Bi':1.75,
                         'F':0.5, 'Cl':0.8, 'Br':0.95, 'I':1.25}


def find_metal_oxidation_states(hit):
    """The find_metal_oxidation_states function attempts to find oxidation states for metals in a CCDC search hit
    object by checking the CCDC-assigned 'charge' of the components of the structure. Whenever charges are not assigned
    properly or are obfuscated by disorder, the returned oxidation state may not be correct.

    :param hit: A hit object from a CCDC Python API search.
    :return: An averaged 'metal charge' for the metals in the structure.
    """
    override_dict = {'WEBBAI': 4}
    if hit.identifier in override_dict:
        metal_charge = override_dict[hit.identifier]
    else:
        crystal = hit.entry.crystal
        # Hydrogens are removed to save time
        crystal.remove_hydrogens()
        total_inorganic_charge = 0
        inorganic_atoms_dict = {}
        for component in crystal.molecule.components:
            component_contains_metal = False
            for atom in component.atoms:
                if atom.is_metal:
                    component_contains_metal = True
                    break
            if component_contains_metal:
                formal_charge = component.formal_charge
                total_inorganic_charge += formal_charge

                for word in component.formula.split(" "):
                    elementsymbol = ""
                    elementcount = ""
                    wordstarted = False
                    wordfinished = False
                    for letter in word:
                        if letter.isalpha():
                            if not wordfinished:
                                elementsymbol += letter
                                wordstarted = True
                        elif letter.isdigit() or letter == ".":
                            wordfinished = True
                            if wordstarted:
                                elementcount += letter
                        else:
                            continue
                    if elementcount:
                        if elementsymbol not in inorganic_atoms_dict:
                            inorganic_atoms_dict.update({elementsymbol: int(elementcount)})
                        else:
                            inorganic_atoms_dict.update({elementsymbol: inorganic_atoms_dict[elementsymbol] + int(elementcount)})

        # Halides and metals are treated specially here. Halides are considered to always have -1 charge.
        metal_count = 0
        halide_count = 0
        for species in inorganic_atoms_dict:
            if species in ['F', 'Cl', 'Br', 'I']:
                halide_count += inorganic_atoms_dict[species]
            if species in ['Ge', 'Sn', 'Pb', 'Ag', 'In', 'Tl', 'As', 'Sb', 'Bi']:
                metal_count += inorganic_atoms_dict[species]
        metal_charge = (halide_count + total_inorganic_charge) / metal_count

    return metal_charge

def pymatgenize(hit):
    """The pymatgenize function turns a ccdc hit object into a Pymatgen Structure object which the below
    functions can utilize.

    :param hit: A hit object from a CCDC Python API search.
    :return: A Pymatgen structure object.
    """
    # We get the cell lengths, angles, atom labels, and coordinates for non-H atoms in the structure.
    crystal = hit.entry.crystal
    # H atoms are removed to save time.
    crystal.remove_hydrogens()
    cell_lengths = crystal.cell_lengths
    cell_angles = crystal.cell_angles
    lattice = Lattice.from_parameters(a=cell_lengths[0], b=cell_lengths[1], c=cell_lengths[2],
                                      alpha=cell_angles[0], beta=cell_angles[1], gamma=cell_angles[2])
    # Drop the symmetry to make sure all atoms in the unit cell are reported.
    low_symmetry_crystal = crystal.reduce_symmetry_to_p1()
    atomlabellist = []
    atomcoordinatelist = []
    # Some EXTRA atoms are reported with labels including '?'. We exclude these as well as carbons (to save time).
    for atom in low_symmetry_crystal.asymmetric_unit_molecule.atoms:
        if atom.atomic_symbol != 'C' and "?" not in atom.label:
            if not any([all(numpy.isclose(atom.fractional_coordinates, coordinate, atol=0.0001))
                        for coordinate in atomcoordinatelist]):
                atomlabellist.append(atom.atomic_symbol)
                atomcoordinatelist.append(numpy.round(atom.fractional_coordinates, 7))

    for index, coordinate in enumerate(atomcoordinatelist):
        for xyz, value in enumerate(coordinate):
            if value <0:
                atomcoordinatelist[index][xyz] += 1
            elif value >= 1:
                atomcoordinatelist[index][xyz] -= 1

    return Structure(lattice, atomlabellist, atomcoordinatelist)

def nmad(inputlist):
    """The nmad() function computes the normalized mean absolute deviation of a set of numbers.

    :param inputlist: A list of numbers.
    :return: The normalized mean absolute deviation (i.e the average distance from the average divided by
    the average).
    """

    # We first calculate the average value from the list.
    average = numpy.average(inputlist)
    # Then we make a list of the deviations from this average of each element in the list.
    devlist = []
    # Finally we calculate the average of this deviation list and divide it by the average calculated above.
    for item in inputlist:
        devlist.append(numpy.absolute(item - average))
    if average == 0:
        nmad_value = 0
    else:
        nmad_value = numpy.average(devlist) / average
    return nmad_value

def xgcd(a, b):
    """The xcd() function is an implementation of the extended Euclidean algorithm for quickly finding the
    greatest common denominator between two numbers.


    :param a: A number
    :param b: A number
    :return: The greatest common demoninator between the two numbers
    """
    if a < b:
        a, b = b, a
    u_i, u_j = 0, 1
    v_i, v_j = 1, 0
    while a % b != 0:
        div = a // b
        a, b = b, a % b
        u_i, u_j = u_j, u_i - div * u_j
        v_i, v_j = v_j, v_i - div * v_j
    return b

class Perovskite:
    """Perovskite class (borrowed from Pyrovskite) will be used to do structure calculations."""

    def __init__(self, structure, a=None, b=None, x=None, n=None, custom_b_cations=None,
                 custom_x_anions=None, suppress_output=True):
        """The __init__ function is called when the Perovskite class is invoked. It attempts to determine
        the B cation(s) and X anion(s) within a perovskite or perovskite-like structure.

        Note that structure is the only necessary parameter, and the others are optional.

        :param structure: A Pymatgen structure object.
        :param a: A list of elemental symbols (as strings) for A cations. These will be identified if they
        are not provided.
        :param b: A list of elemental symbols (as strings) for B cations. These will be identified if they
        are not provided.
        :param x: A list of elemental symbols (as strings) for X anions. These will be identified if they
        are not provided.
        :param n: An n-value (number) for a layered (2D) perovskite.
        :param custom_b_cations: A list of elemental symbols (strings) to be considered for identifying
        the B cation(s). If no list is provided, a default list will be used.
        :param custom_x_anions: A list of elemental symbols (strings) to be considered for identifying the
        X anion(s). If no list is provided, a default list will be used.
        :param suppress_output: A Boolean value. When set to True, it squelches console output.
        """
        self.structure = structure
        self.a = a
        self.b = b
        self.x = x
        self.n = n
        self.structure_radius = None
        self.polyhedra = None
        self.frac_polyhedra = None
        self.cpp = None
        self.perovset = None
        self.simplestructure = None
        self.connectivity_indices = None
        self.normalized_connectivity_indices = None
        self.tilting_angles = None
        self.distances_and_angles = None
        self.distortion_indices = None
        self.shared_X_sites = None
        self.cartesian_shared_X_sites = None
        self.adjacency = None
        self.dimensionality = None
        self.description = None
        self.non_octahedral = None
        self.coordination_numbers = None

        if custom_b_cations is not None:
            self.common_b = custom_b_cations
        else:
            # These are the B cations for which Perovskite() looks, if not otherwise specified.
            self.common_b = ['Pb', 'Sn', 'Ge', 'Ga', 'In', 'Tl', 'As', 'Sb', 'Bi', 'Ag']

        if custom_x_anions is not None:
            self.common_x = custom_x_anions
        else:
            # These are the X anions for which Perovskite() looks, if not otherwise specified.
            self.common_x = ['F', 'Cl', 'Br', 'I']

        unique_types = []
        # Make a list of the types of atoms in the structure.
        for ele in self.structure.types_of_specie:
            unique_types.append(str(ele))

        self.atom_types = unique_types

        # First check whether A, B sites have been identified by the input.
        if self.a is None:
            pass
        if self.b is None:
            # Since they haven't already been identified, we start to look for them.
            # Make a list of candidate B cations
            b_candidates = list(set(self.common_b) & set(self.atom_types))
            if not suppress_output:
                print(f"\nNo B-cation set; candidate B cations determined from structure:\n{b_candidates}\n")

            # Whichever elements from the candidate B cations that are present in the structure are
            # assigned as B cations.
            if len(b_candidates) == 1:
                self.b = b_candidates
            elif len(b_candidates) >= 2:
                if not suppress_output:
                    print(f"Two or more B-site candidates found, possible double perovskite system\n"
                          f"Setting B={b_candidates}")
                self.b = b_candidates
            else:
                raise ValueError("B-site cation not detected; provide manually.")

        # The X anion is more complicated because some anions (i.e. F, Cl, Br, I) appear in organic
        # molecules.
        if self.x is None:
            x_candidates = list(set(self.common_x) & set(self.atom_types))

            if not suppress_output:
                print(f"\nNo X-anion set; candidate X-anions determined from structure:\n{x_candidates}\n")

            # Again, if the candidates list has only one element, we simply use that one.
            if len(x_candidates) == 1:
                self.x = x_candidates

            # If there is more than one candidate, we check each instance of each candidate for whether it
            # is close (within 3.5 Angstroms) to a B cation. If it is close, we consider it an X anion.
            elif len(x_candidates) > 1:
                x_list = []
                if not x_list:
                    for site in self.structure.sites:
                        if site.species_string.rsplit(":")[0] in x_candidates:
                            neighborslist = self.structure.get_neighbors(site, 3.5)
                            for neighbor in neighborslist:
                                if neighbor.species_string.rsplit(":")[0] in self.b:
                                    x_list.append(site.species_string.rsplit(":")[0])
                    self.x = list(set(x_list))
                    if not suppress_output:
                        print(
                            f"{self.x} determined as X-anions.\n")
            elif len(x_candidates) == 0:
                raise ValueError("X-site anion not properly detected; provide manually.")

    def identify_polyhedra(self, oxidized_metal = False):
        """The identify_polyhedra() function applies to a Perovkite object (e.g.
        Perovskite.identify_polyhedra()) and calculates the centers (B cations) and vertices (X anions) of
        each polyhedron within the structure.

        Note that return_distances is deprecated. See the distances() and angles() function.

        In the future, this function may be used to assess disordered strutures by including site
        occupancies in the calculation.

        :param self: A Perovskite object.
        :param oxidized_metal: A boolean specifying whether the metal site is oxidized relative to its assumed
                                oxidation state (e.g. Sn4+ instead of Sn2+)

        :return: A list of lists of triples, where each interior list designates a polyhedron, the first
        triple is the coordinates of the B cation, and the following triples are the coordinates of X
        anions. Note that the triples take the form [a, b, c], where a, b, and c are coordinates (in
        either fractional or Cartesian form, depending on the value of frac).
        """
        b = self.b
        x = self.x
        perovset = set(b+x)
        polyhedra = []
        frac_polyhedra = []
        poly_indices = []
        structure = self.structure
        atomlist = []
        # Make a list of the types of atoms in the structure.
        for ele in structure.types_of_specie:
            atomlist.append(str(ele))
        atomset = set(atomlist)
        removeset = atomset.difference(perovset)
        # Generate a dummy structure from which to remove things for time-saving.
        simplestructure = structure
        simplestructure.remove_species(removeset)

        removeset = set()
        site_splitting = set()
        for site in simplestructure.sites:
            if site.species_string in self.b:
                if site not in removeset:
                    for neighbor in simplestructure.get_neighbors(site, 1.9):
                        if neighbor.species_string in self.b:
                            removeset.add(simplestructure.sites[neighbor.index])
                            site_splitting.add(site.species_string)
            if site.species_string in self.x:
                if site not in removeset:
                    for neighbor in simplestructure.get_neighbors(site, 1.9):
                        if neighbor.species_string in self.x:
                            removeset.add(simplestructure.sites[neighbor.index])
                            site_splitting.add(site.species_string)
        for site in removeset:
            simplestructure.remove(site)
        self.simplestructure = simplestructure


        # This block defines special radii used for the program.
        if oxidized_metal:
            atomic_radii_updates = bond_lengths_oxidized
        else:
            atomic_radii_updates = bond_lengths
        if all("+" in metal for metal in b):
            ionic_radii = True
        else:
            ionic_radii = False
        if ionic_radii:
            preferred_radii = jmol_radii_ionic
        else:
            preferred_radii = jmol_radii | atomic_radii_updates

        # This block uses the preferred radii to generate an assumed maximum radius for the structure
        blist = []
        xlist = []
        for metal in b:
            blist.append(preferred_radii[metal])
        for anion in x:
            xlist.append(preferred_radii[anion])
        if ionic_radii:
            self.structure_radius = (max(blist) + max(xlist)) * 1.2
        elif not ionic_radii:
            self.structure_radius = max(blist) + max(xlist)

        # We begin by appending the coordinates of each B cation to the polyhedra list.
        polyhedra_ct = 0
        for idx, site in enumerate(self.simplestructure.sites):
            if site.species_string.rsplit(":")[0] in b:
                frac_polyhedra.append([site.frac_coords])
                polyhedra.append([site.coords])
                poly_indices.append([idx])
                polyhedra_ct += 1

                # Now, we get the neighbors of each B cation a range based on the atoms
                neighbors = self.simplestructure.get_neighbors(site, self.structure_radius)
                x_neighbors = []
                for neighbor in neighbors:
                    if neighbor.species_string.rsplit(":")[0] in x:
                        x_neighbors.append([neighbor[0], neighbor[1], neighbor[3]])

                # Here, we sort the list of neighboring X anions by proximity then append them to the polyhedron
                sorted_x_neighbors = sorted(x_neighbors, key=lambda y: y[1])
                frac_polyhedra[polyhedra_ct - 1].extend([xi[0].frac_coords for xi in sorted_x_neighbors])
                polyhedra[polyhedra_ct - 1].extend([xi[0].coords for xi in sorted_x_neighbors])
            else:
                pass
        self.polyhedra = polyhedra
        self.frac_polyhedra = frac_polyhedra

        # Take note of the unique coordination numbers for the polyhedra
        # If any are not exactly 6, we mark the structure as 'non-octahedral'
        # Note that 'polyhedron' lists contain metals, so we subtract 1 to get the metal's coordination number
        self.coordination_numbers = []
        dummy_numbers_list = []
        self.non_octahedral = False
        for polyhedron in self.polyhedra:
            if (len(polyhedron) - 1) not in dummy_numbers_list:
                dummy_numbers_list.append(len(polyhedron) - 1)
            if len(polyhedron) > 7 or len(polyhedron) <7:
                self.non_octahedral = True
        dummy_numbers_list.sort(reverse=True)
        for number in dummy_numbers_list:
            self.coordination_numbers.append(str(number))



        # This block calculates the average charge per polyhedron (C/P) for the structure. It ignores structures with
        # multiple differently charged cations.
        charges =          {'Pb':2, 'Sn':2, 'Ge':2, 'As':3, 'Sb':3, 'Bi':3, 'Ag':1, 'In':3, 'Tl':1}
        oxidized_charges = {'Pb':2, 'Sn':4, 'Ge':4, 'As':5, 'Sb':5, 'Bi':5, 'Ag':1, 'In':3, 'Tl':1}
        try:
            if oxidized_metal:
                structurecharges = {oxidized_charges[x] for x in self.b}
            else:
                structurecharges = {charges[x] for x in self.b}
            bcationcharge = None
            if len(structurecharges) == 1:
                bcationcharge = list(structurecharges)[0]
            if bcationcharge:
                vertlist = list()
                for polyhedron in frac_polyhedra:
                    for vert in polyhedron[1:]:
                        if not all([0 <= site < 1 for site in vert]):
                            for idx, site in enumerate(vert):
                                if site < 0:
                                    vert[idx] += 1
                                if site >= 1:
                                    vert[idx] -= 1
                        vertlist.append(numpy.round(vert, 7))
                vertlist = numpy.unique(vertlist, axis=0)
                npolyhedra = len(polyhedra)
                poscharge = npolyhedra * bcationcharge
                negcharge = len(vertlist)
                self.cpp = (poscharge - negcharge) / npolyhedra
        except KeyError:
            self.cpp = None

    def connectivity(self):
        """The connectivity() function calculates a set of indices describing the amount and type of polyhedral
            sharing within a structure (i.e. terminal, corner-sharing, edge-sharing, or face-sharing). It is
            intended to help determine types of octahedral structures including perovskites and perovskitoids, but
            the theory of it applies to any polyhedral structure.

            Will not work properly unless Perovskite.identify_polyhedra() has been called FIRST and succeeded normally.

            :param self: A Perovskite object.

            It builds the property Perovskite.connectivity_indices, a list of numbers including [T, C, E, F]
            as well as the property Perovskite.tilting_angles, a list of interpolyhedral
            angles. Note that these are only calculated for vertices shared by two polyhedra. Vertices
            shared by more than two polyhedra are considered in the calculation of connectivity indices
            but not in the calculation of tilting angles.
            """
        simplestructure = self.simplestructure
        overall_sharing = [0,0,0,0]
        tilting_angles = []
        shared_x_sites = []
        for site in simplestructure.sites:
            # [t, c, e, f]
            # Make list of polyhedra that neighbor each other.
            if site.species_string in self.x:
                neighboring_polyhedra = []
                neighbors = simplestructure.get_neighbors(site, self.structure_radius)
                for neighbor in neighbors:
                    if neighbor.species_string in self.b:
                        b_neighbors = simplestructure.get_neighbors(neighbor[0], self.structure_radius)
                        neighboring_polyhedron = [neighbor[0].coords]
                        for b_neighbor in b_neighbors:
                            if b_neighbor.species_string in self.x:
                                neighboring_polyhedron.append(b_neighbor[0].coords)
                        neighboring_polyhedra.append(neighboring_polyhedron)

                if len(neighboring_polyhedra) == 0:
                    continue
                elif len(neighboring_polyhedra) == 1:
                    overall_sharing[0] += 1
                elif len(neighboring_polyhedra) > 1:
                    shared_x_sites.append([site.coords, neighboring_polyhedra])
                    neighboring_pairs_list = list(combinations(neighboring_polyhedra, 2))
                    for neighboring_pair in neighboring_pairs_list:
                        # We want to find the size of the intersection between the neighboring pair; the fastest
                        # way to do that is with sets, so we make the polyhedra into sets of tuples.
                        first_polyhedron = set(map(tuple, neighboring_pair[0]))
                        second_polyhedron = set(map(tuple, neighboring_pair[1]))
                        intersection = first_polyhedron.intersection(second_polyhedron)
                        pair_sharing = len(intersection)
                        if 0 < pair_sharing < 4:
                            overall_sharing[pair_sharing] += 1/len(neighboring_pairs_list)

                # If the x site is shared by exactly two polyhedra, we can calculate the B-X-B angle.
                if len(neighboring_polyhedra) == 2:
                    vector_1 = neighboring_polyhedra[0][0] - site.coords
                    vector_2 = neighboring_polyhedra[1][0] - site.coords
                    tilting_radians = numpy.arccos(numpy.dot(vector_1, vector_2) /
                                                   (numpy.linalg.norm(vector_1) * numpy.linalg.norm(vector_2)))
                    tilting_degrees = math.degrees(tilting_radians)
                    tilting_angles.append(numpy.round(tilting_degrees, decimals=2))

        # Tilting angles of 180 cause arccos to return NaN.
        for idx, angle in enumerate(tilting_angles):
            if math.isnan(angle):
                tilting_angles[idx] = 180

        self.shared_X_sites = shared_x_sites
        self.connectivity_indices = numpy.round(overall_sharing, decimals=7)
        self.normalized_connectivity_indices = self.connectivity_indices / sum(self.connectivity_indices)
        self.tilting_angles = tilting_angles

    def calc_adjacency(self):
        """The calc_adjacency() function calculates an 'adjacency' parameter for a Perovskite object, which is the.
            average fraction of adjacent 'shared' vertices for each shared halide in the structure. This parameter is
            useful for distinguishing between structures that otherwise have the same connectivity indices, such as
            (100)- and (110)-sliced 2D perovskites.

            For example, a (100)-sliced 2D perovskite with the ABX4 formula has connectivity indices of [0.5, 0.5, 0, 0]
            and an adjacency of 0.5, while a (110)-sliced 2D perovskite with the same formula has the same connectivity
            but an adjacency of 0.625.

            Will not work properly unless Perovskite.identify_polyhedra() AND Perovskite.connectivity() have been
            called FIRST and succeeded normally.

            :param self: A Perovskite object.
            :return: Returns nothing, but builds the Perovskite.adjacency property, which is either a string with 'N/A'
            if the structure has no sharing or a float 0 <= x <= 1 if the structure has some sharing.
            """
        # Don't try to calculate adjacency if there are no shared halides
        if tuple(self.normalized_connectivity_indices) == tuple([1, 0, 0, 0]):
            self.adjacency = "N/A"
        else:
            average_adjacency = 0
            # Note that self.shared_X_sites is a list of lists that look like [point, polyhedron]
            shared_x_sites = self.shared_X_sites
            shared_x_sites_coords_list = []
            # We need the list of shared sites by itself for comparison
            for site_and_neighboring_polyhedra_list in shared_x_sites:
                site_frac_coords = numpy.dot(site_and_neighboring_polyhedra_list[0], self.structure.lattice.inv_matrix)
                shared_x_sites_coords_list.append(site_frac_coords)

            # Now go through the shared sites and get their individual adjacencies
            for site_and_neighboring_polyhedra_list in shared_x_sites:
                site = site_and_neighboring_polyhedra_list[0]
                neighboring_polyhedra = site_and_neighboring_polyhedra_list[1]
                for polyhedron in neighboring_polyhedra:
                    # We will use the convex hull to figure out which X sites are adjacent
                    hull = ConvexHull(polyhedron[1:])
                    shared_index_set = set()
                    adjacent_index_set = set()
                    site_index = None
                    for index, vertex in enumerate(hull.points):
                        vertex_frac_coords = numpy.round(numpy.dot(vertex, self.structure.lattice.inv_matrix), 7)
                        if all(numpy.isclose(numpy.array(vertex), site, atol=0.0001)):
                            site_index = index
                        # The "% 1" is used here to make the check not care if a site is outside the unit cell.
                        # A quirk happens when floating point makes one of the numbers read as "-0", in which case
                        # the %1 modifier returns 1 instead of 0. We fix that by rounding the vertex_frac_coords above.
                        elif any([all(numpy.isclose(numpy.array(vertex_frac_coords) % 1, shared_x_site, atol=0.0001))
                                for shared_x_site in shared_x_sites_coords_list]):
                            shared_index_set.add(index)
                    for simplex in hull.simplices:
                        # These simplices are tuples of three indices, where the indices correspond to the position
                        # of the hull's vertices in the hull.points list. We're treating a vertex as 'adjacent' if it
                        # shares a simplex (face) with the one we're investigating.
                        if site_index in simplex:
                            adjacent_index_set.update(set(simplex).difference({site_index}))
                    adjacency = len(adjacent_index_set.intersection(shared_index_set)) / len(adjacent_index_set)
                    average_adjacency += adjacency /  (len(neighboring_polyhedra) * len(shared_x_sites))
            self.adjacency = average_adjacency

    def calc_distances_and_angles(self):
        """The distances_and_angles() function calculates bond distances, edge lengths, and interior angles of
            each polyhedron within a Perovskite.

            Note that the function only works with structures for which the polyhedra are well-ordered (no site
            splitting of polyhedral sites).

            Special Note: Edge length and interior angle calculations only work for polyhedra with triangular
            faces. Other polyhedra will require special consideration.

            Will not work properly unless Perovskite.identify_polyhedra() has been called FIRST and succeeded normally.

            :param self: A Perovskite object.
            :return: Returns nothing, but builds the property Perovskite.distances_and_angles,
            a list of lists of lists of numbers, [distances, edgelengths, angles] where distances is a
            list of bond distances within each polyhedron, edgelengths is a list of edge lengths within each
            polyhedron, and angles is a list of interior (edge) angles within each polyhedron.
            """
        distances = []
        angles = []
        edgelengths = []
        polyhedra = self.polyhedra

        # Calculate a list of edes by building the polyhedron from its vertices using scipy ConvexHull()
        # class and taking the edges from the simplices (triangular faces) that make up the polyhedron,
        # and then calculate the length of each edge. Note that this method only works for polyhedra with
        # triangular faces (e.g. tetrahedra, octahedra) and needs special consideration for other polyhedra.
        angles_and_edges_error = False
        try:
            for polyhedron in polyhedra:
                edgepairs = []
                hull = ConvexHull(polyhedron[1:])
                simplices = hull.simplices
                duplist = []
                for j in hull.vertices:
                    for k in hull.vertices:
                        # Make sure we don't calculate a vertex's distance to itself.
                        if j != k:
                            for simplex in simplices:
                                if j in simplex and k in simplex:
                                    # Make sure we don't add duplicates to the list.
                                    if [j, k] not in duplist and [k, j] not in duplist:
                                        edgepairs.append([hull.points[j], hull.points[k]])
                                        duplist.append([j, k])
                octedgelengths = []
                for edge in edgepairs:
                    octedgelengths.append(math.dist(edge[0], edge[1]))
                edgelengths.append(octedgelengths)

                # Calculate the interior (edge) angles from the list of edges. This is done by drawing vectors
                # from the center to either vertex of the edge and calculating the angle between the vectors.
                polyangles = []
                for edge in edgepairs:
                    edge.append(polyhedron[0])
                    u = edge[0] - edge[2]
                    v = edge[1] - edge[2]
                    angle_radians = numpy.arccos(numpy.dot(u, v) / (numpy.linalg.norm(u) * numpy.linalg.norm(v)))
                    angle_degrees = math.degrees(angle_radians)
                    polyangles.append(angle_degrees)
                angles.append(polyangles)

        except (scipy.spatial._qhull.QhullError, IndexError, ZeroDivisionError):
            angles_and_edges_error = True
            pass

        # Calculate the distances within each polyhedron directly.
        for polyhedron in polyhedra:
            octdistance = []
            for x in polyhedron[1:]:
                octdistance.append(math.dist(x, polyhedron[0]))
            distances.append(octdistance)

        if angles_and_edges_error:
            self.distances_and_angles = [distances, None, None]

        elif not angles_and_edges_error:
            self.distances_and_angles = [distances, edgelengths, angles]

    def calc_distortion_indices(self):
        """The calc_distortion_indices() function calculates the XT, XTX, and XX distortion indices of the
        polyhedra within a structure.

        These indices use the formalism established by Baur (e.g., Bauer WH, 1974, Acta Cryst. B40:1195–1215)
        and take the form:
                DI = SUM(i=1 up to n, |Pi-Pavg|/(n*Pavg))
        where P is the parameter of interest (bond distance for XT, interior angle for XTX, or edge-length for
        XX).

        Will not work properly unless Perovskite.identify_polyhedra() and Perovskite.calc_distances_and_angles
        have been called FIRST and succeeded normally.

        :param self: A Perovskite object.
        :return: No return, but builds the property Perovskite.distortion_indices, which is
        a list of polyhedral distortion indices XT, XTX, and XX for each polyhedron in the structure
        using the formalism established by Baur (e.g., Bauer WH, 1974, Acta Cryst. B40:1195–1215)
        Specifically, returns [dixt_list, dixtx_list, dixx_list], where dixt_list is a list of the normalized mean
        absolute deviations of polyhedral bond distances within each polyhedron of the structure, dixtx_list is a
        list of the normalized mean absolute deviations of the interior polyhedral angles of each polyhedron
        in the structure, and dixx_list is a list of the normalized mean absolute deviations of the polyhedral
        edge lengths of each polyhedron in the structure.
        """
        # See distances_and_angles() function for calculation of individual distances and angles.
        distance_angles_list = self.distances_and_angles
        dixt_list = []
        dixtx_list = []
        dixx_list = []
        # For each polyhedra, we append the normalized mean absolute deviation, i.e. the average distance from
        # the average divided by the average. We do this for the bond lengths (XT), angles (XTX),
        # and edge lengths (XX).
        for polyhedron in distance_angles_list[0]:
            dixt_list.append(nmad(polyhedron))
        if distance_angles_list[1]:
            for polyhedron in distance_angles_list[1]:
                dixx_list.append(nmad(polyhedron))
        if distance_angles_list[2]:
            for polyhedron in distance_angles_list[2]:
                dixtx_list.append(nmad(polyhedron))
        self.distortion_indices = [dixt_list, dixtx_list, dixx_list]

    def calc_dimensionality(self, oxidized_metal = False):
        """The calc_dimensionality function calculates the covalent (bonding)vdimensionality of a structure, using a
            method described by Larsen as implemented in Pymatgen's get_dimensionality_larsen() function.

            Will not work properly unless Perovskite.identify_polyhedra() has been called FIRST and succeeded normally.

            :param self: A Perovskite object.
            :param oxidized_metal: A boolean specifying whether the metal site is oxidized relative to its assumed
                                oxidation state (e.g. Sn4+ instead of Sn2+)
            :return: No return, but builds the property Perovskite.dimensionality, which is an
            integer representing the covalent dimensionality.
            """
        simplestructure = self.simplestructure
        if oxidized_metal:
            jmol_update = bond_lengths_oxidized
        else:
            jmol_update = bond_lengths
        bonded_structure = JmolNN(el_radius_updates=jmol_update, tol=0.0).get_bonded_structure(simplestructure)
        # Remove 'bonds' between x anions or between b cations
        edges_to_be_removed = []
        for edge in bonded_structure.graph.edges:
            from_atom = edge[0]
            to_atom = edge[1]
            # Note that we have to keep track of whether the edge crosses the unit cell boundaries and,
            # if it does, which boundary. This is stored in the 'to_jimage' value.
            to_jimage = bonded_structure.graph.edges[edge]['to_jimage']
            if (simplestructure.sites[from_atom].species_string in self.x
                    and simplestructure.sites[to_atom].species_string in self.x):
                edges_to_be_removed.append([from_atom, to_atom, to_jimage])
            elif (simplestructure.sites[from_atom].species_string in self.b
                  and simplestructure.sites[to_atom].species_string in self.b):
                edges_to_be_removed.append([from_atom, to_atom, to_jimage])
        for edge in edges_to_be_removed:
            bonded_structure.break_edge(from_index=edge[0], to_index=edge[1], to_jimage=edge[2])
        self.dimensionality = int(get_dimensionality_larsen(bonded_structure))

    def describe_structure(self):
        """The describe_structure function uses the connectivity indices, adjacency, and dimensionality of a structure
            to estimate the structure type.

                Will not work properly if Perovskite.identify_polyhedra(), Perovskite.connectivity(),
                Perovskite.calc_adjacency() and Perovskite.calc_dimensionality have not been called FIRST and succeeded.

                :param self: A Perovskite object.
                :return: No return, but builds the property Perovskite.description, a string representing
                the structure type.
                """
        structure_class = ''
        [terminal, corner, edge, face] = self.normalized_connectivity_indices
        if self.non_octahedral:
            structure_class = 'Non-octahedral'
        else:
            if not corner and not edge and not face:
                structure_class = "Isolated octahedra"
            elif corner and not edge and not face:
                structure_class = "Perovskite"
            elif not corner and edge and not face:
                structure_class = "Edge-sharing"
            elif not corner and not edge and face:
                structure_class = "Face-sharing"
            elif corner and (face or edge):
                structure_class = "Perovskitoid"
            elif not corner and face and edge:
                structure_class = "Face- and edge-sharing"
        if structure_class and isinstance(self.dimensionality, int):
            structure_description = str(self.dimensionality) + "D" + " " + structure_class
        elif structure_class and not isinstance(self.dimensionality, int):
            structure_description = "?D " + structure_class
        elif isinstance(self.dimensionality, int) and self.connectivity_indices == [0, 0, 0, 0]:
            structure_description = "? " + str(self.dimensionality) + "D"
        else:
            structure_description = "N/F"

        self.description = structure_description

        # For (100) 2D perovskites, the n values give adjacencies of 1/2, 4/5, 7/8, 10/11, 12/13, ...
        # As a sequence, this is (3n-2)/(3n-1) for n = 1,2,3,4,...
        if self.description == '2D Perovskite':
            for n, ratio in enumerate([2/(3*x + 1) for x in range(10)[1:]]):
                if self.description != '2D Perovskite':
                    break
                elif numpy.isclose(self.adjacency, (3*(n + 1) - 2)/(3*(n + 1) - 1), atol=0.0001):
                    if numpy.isclose(terminal, ratio, atol = 0.0001):
                        self.description = f'2D Perovskite n={n + 1}'

        # For (111) 2D perovskites, the n values give adjacencies of 1/2, 3/4, 5/6, 7/8, ...
        # As a sequence, this is (2q-3)/(2q-2) for q = 3,4,5,6... and 2/3 for q = 2
        if self.description == '2D Perovskite':
            for q, ratio in enumerate([4/(3*x + 0) for x in range(10)[2:]]):
                if self.description != '2D Perovskite':
                    break
                elif numpy.isclose(self.adjacency, (2*(q + 2) - 3)/(2*(q + 2) - 2), atol=0.0001):
                    if numpy.isclose(terminal, ratio, atol=0.0001):
                        self.description = f'2D Perovskite (111) q={q+2}'

        # For (110) 2D perovskite, the n values give adjacencies of 5/8, 11/14, 17/20, 23/26, ...
        # As a sequence, this is (6m-7)/(6m-4) for m = 2,3,4,5...
        if self.description == '2D Perovskite':
            for m, ratio in enumerate([4/(3*x + 2) for x in range(10)[2:]]):
                if self.description != '2D Perovskite':
                    break
                elif numpy.isclose(terminal, ratio, atol=0.0001):
                    if numpy.isclose(self.adjacency, (6*(m + 2) - 7)/(6*(m + 2) - 4), atol=0.0001):
                        self.description = f'2D Perovskite (110) m={m + 2}'
                # Note that (110) 2D perovskites come in other varieties, one common one of which is the '(k*l)' variety,
                # which gives adjacencies of 18/32 for 3x3, 26/48 for 4x4, 34/64 for 5x5, 28/48 for 2x3, 36/64 for 2x4,
                # 44/80 for 2*5, and 5/8 for 2x2, which seems like an outlier among this series
                # for k*k this gives (8*k-6)/(16*k-16) when k>=3
                # for 2*k this gives (8*k+4)/(16*k) for k>=2
                # Thicknesses (n values) are generally all n=1 for these special cases but not always
                elif numpy.isclose(terminal, 0.5, atol=0.0001):
                    if numpy.isclose(self.adjacency, (8 * (m + 2) - 6) / (16 * (m + 2) - 16), atol=0.0001):
                        self.description = f'2D Perovskite (110) {(m + 2)}*{(m + 2)}'
                    elif numpy.isclose(self.adjacency, (8*(m + 2) + 4)/(16*(m + 2)), atol=0.0001):
                        self.description = f'2D Perovskite (110) 2*{(m + 2)}'

        # Otherwise we give up trying to guess the type of 2D perovskite
        if self.description == '2D Perovskite':
            self.description = '2D Perovskite misc'

# For internal use
def sanitize_name(name):
    nameslist = name.split(" ")
    sanitizedlist = []
    for shortname in nameslist:
        shortname = shortname.replace("diaminomethaniminium", "guanidinium")
        shortname = shortname.replace("aminomethaniminium", "formamidinium")
        shortname = shortname.replace("ethanaminium", "ethylammonium")
        shortname = shortname.replace("ethan-1-aminium", "ethylammonium")
        shortname = shortname.replace("butan-1-aminium", "butylammonium")
        shortname = shortname.replace("propan-1-aminium", "propylammonium")
        shortname = shortname.replace("pentan-1-aminium", "pentylammonium")
        shortname = shortname.replace("hexan-1-aminium", "hexylammonium")
        if shortname.startswith("catena-"):
            shortname = shortname.replace("catena-[", "")
            shortname = shortname.replace("catena-(", "")
            nameslist[-1] = nameslist[-1][:-1]
        if shortname.startswith("catena"):
            shortname = shortname.replace("catena", "")
        if shortname.startswith("-"):
            shortname = shortname.replace("-", "")
        for opening in ["bis", "tris", "tetrakis", "pentakis", "hexakis", "heptakis", "octakis",
                        "nonakis", "decakis", "undecakis", "dodecakis", "tridecakis",
                        "tetradecakis", "pentadecakis", "hexadecakis", "heptadecakis",
                        "octadecakis", "nonadecakis", "icosakis", "henicosakis", "docosakis",
                        "tricosakis", "tetracosakis", "pentacosakis", "hexacosakis",
                        "heptacosakis", "octacosakis", "nonacosakis", "triacontakis",
                        "hentriacontakis", "dotriacontakis", "tritriacontakis",
                        "tetratriacontakis", "pentatriacontakis", "hexatriacontakis",
                        "heptatriacontakis", "octatriacontakis", "nonatriacontakis",
                        "tetracontakis", "hentetracontakis", "dotetracontakis", "tritetracontakis",
                        "tetratetracontakis", "pentatetracontakis", "hemikis", "sesquikis"]:
            if shortname.startswith(opening):
                shortname = shortname.replace(opening, "")

        if "cesium" in shortname:
            sanitizedlist.append("cesium")
            continue
        if "rubidium" in shortname:
            sanitizedlist.append("rubidium")
            continue

        if "lead" in shortname or "tin" in shortname or "germanium" in shortname or "solvate" in shortname or "plumb" \
                in shortname or "stann" in shortname or "german" in shortname or 'indium' in shortname or 'thallium' \
                in shortname or 'arsenic' in shortname or 'antimony' in shortname or 'bismuth' in shortname or 'muth' \
                in shortname or 'arsenate' in shortname or 'antimonate' in shortname or 'gallium' in shortname or \
                'gallate' in shortname or 'thallate' in shortname or 'indate' in shortname or 'silver' in shortname:
            continue
        while (shortname.startswith("[") and shortname.endswith("]")) or (shortname.startswith("(") and
                                                                shortname.endswith(")")):
            shortname = shortname[1:-1]
        sanitizedlist.append(shortname)
    return ", ".join(sanitizedlist)

def imagegeneration(hit):
    diagram_generator = DiagramGenerator()
    diagram_generator.settings.font_size = 12
    diagram_generator.settings.line_width = 1.6
    diagram_generator.settings.image_width = 500
    diagram_generator.settings.image_height = 500
    diagram_generator.settings.shrink_symbols = False
    csdentry = hit.entry

    molecules = csdentry.crystal.molecule.components
    newmolecule = ccdc.molecule.Molecule()
    for molecule in molecules:
        smilesstring = molecule.smiles
        if smilesstring is not None:
            if "Pb" in smilesstring or "Sn" in smilesstring or "Ge" in smilesstring:
                continue
            else:
                newmolecule.add_molecule(molecule)
    img = diagram_generator.image(newmolecule)

    if img:
        return img
    else:
        return diagram_generator.image(csdentry)

def execute_search(metals = 'Group14'):
    # Note, you can make an OR by putting a list inside ONE query
    # You can make an AND by putting the atoms in separate queries
    metals1 = list()
    metals2 = list()
    halides = list()
    forbidden_partners = list()
    forbidden_atoms = list()
    if metals == 'Group14':
        metals1 = ccdc.search.QueryAtom(['Ge', 'Sn', 'Pb'])
        metals2 = ccdc.search.QueryAtom(['Ge', 'Sn', 'Pb'])
        halides = ccdc.search.QueryAtom(['F', 'Cl', 'Br', 'I'])
        forbidden_partners = ccdc.search.QueryAtom(['B', 'N', 'C', 'O', 'S', 'P', 'Se', 'Si'])
        forbidden_atoms = ccdc.search.QueryAtom(['Li', 'Na', 'K', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Sc', 'Y',
                                                 'La', 'Ac', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'Cr', 'Mo', 'W', 'Mn',
                                                 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pd', 'Pt', 'Cu',
                                                 'Ag', 'Au', 'Zn', 'Cd', 'Hg', 'Al', 'Ga', 'In', 'Tl', 'As', 'Sb', 'Bi',
                                                 'Te', 'Po', 'Ce', 'Th', 'Pr', 'Pa', 'Nd', 'U', 'Pm', 'Np', 'Sm', 'Pu',
                                                 'Eu', 'Am', 'Gd', 'Cm', 'Tb', 'Bk', 'Dy', 'Cf', 'Ho', 'Es', 'Er', 'Fm',
                                                 'Tm', 'Md', 'Yb', 'No', 'Lr'])
    elif metals == 'Group13+15':
        metals1 = ccdc.search.QueryAtom(['Bi', 'Sb', 'As', 'Tl', 'In'])
        metals2 = ccdc.search.QueryAtom(['Bi', 'Sb', 'As', 'Tl', 'In'])
        halides = ccdc.search.QueryAtom(['F', 'Cl', 'Br', 'I'])
        forbidden_partners = ccdc.search.QueryAtom(['B', 'N', 'C', 'O', 'S', 'P', 'Se', 'Si'])
        forbidden_atoms = ccdc.search.QueryAtom(['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra', 'Sc', 'Y', 'La', 'Ac', 'Ti', 'Zr',
                                                  'Hf', 'Ta', 'W', 'Mn', 'Tc', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir',
                                                  'Ni', 'Pd', 'Pt', 'Cu', 'Au', 'Zn', 'Cd', 'Hg', 'Al', 'Ga', 'Ce', 'Pr',
                                                  'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm','Yb', 'Lu',
                                                  'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                                                  'No', 'Lr', 'Li', 'Na', 'K', 'Fr'])
    # Make a bond query (can be included or excluded later, checks for anything with a bond from first group to second group)
    bonds_query1 = ccdc.search.QueryBond('Single')
    allowed_query = ccdc.search.QuerySubstructure()
    allowed_query.add_atom(metals1)
    allowed_query.add_atom(halides)
    allowed_query.add_bond(bonds_query1, metals1, halides)

    forbidden_bonds_query = ccdc.search.QuerySubstructure()
    bonds_query2 = ccdc.search.QueryBond('Single')
    forbidden_bonds_query.add_atom(metals2)
    forbidden_bonds_query.add_atom(forbidden_partners)
    forbidden_bonds_query.add_bond(bonds_query2, metals2, forbidden_partners)

    # Make an atom query (can be included or excluded later, checks for those atoms)
    forbidden_atoms_query = ccdc.search.QuerySubstructure()
    forbidden_atoms_query.add_atom(forbidden_atoms)

    # It seems that additive groups (like allowed bonds here) need to be actually ran as a search with
    # max_hits_per_structure=1, otherwise you get a lot of duplicates, even if using the proper instruction later in the
    # combined search. I am not sure why this is necessary here and not for the excluded groups (like forbidden bonds group).
    allowed_bonds_search = ccdc.search.SubstructureSearch()
    allowed_bonds_search.add_substructure(allowed_query)
    allowed_bonds_search.search(max_hits_per_structure=1)

    forbidden_bonds_search = ccdc.search.SubstructureSearch()
    forbidden_bonds_search.add_substructure(forbidden_bonds_query)

    forbidden_atoms_search = ccdc.search.SubstructureSearch()
    forbidden_atoms_search.add_substructure(forbidden_atoms_query)

    combined_search = search.CombinedSearch(allowed_bonds_search & -forbidden_bonds_search & -forbidden_atoms_search)

    # Here, you can actually even leave out the max_hits_per_structure=1 command; it doesn't affect the result or runtime.
    hits = combined_search.search()
    print(len(hits))
    return hits

def export_data(hitlist, databasepath, reload=False, images=False):
    global_start_time = time.perf_counter()
    describe_structure_time = None
    timedict = {'reading':0, 'sanitize_name':0, 'pymatgenize':0, 'find_metal_oxidation_states': 0, 'Perovskite':0,
                'identify_polyhedra':0,'connectivity':0, 'calc_adjacency':0, 'calc_distances_and_angles':0,
                'calc_distortion_indices':0,'calc_dimensionality':0, 'describe_structure':0, 'wrapping up':0}
    bigdatalist = []
    skiplist = ['AVOSEK']
    failedlist = []
    databaselocation = databasepath + "\\" + "database.txt"
    header_line = ("CSD Code\tChemical Formula\tMoiety Formula\tOrganic Components\tDiagram\tYear\tDOI\t"
                  "Structure Description\tT\tC\tE\tF\tAdjacency\tB cation(s)\tX anion(s)\tCoordination\t"
                   "Metal Charge\tC/P\tColor\tDescription\tSpace Group\tR Factor\tDiff. Temp\tCell Lengths [a,b,c]\t"
                  "Cell Angles [alpha, beta, gamma]\tZ\tDensity\tBond Distances\tTilting Angles\tDI: XT\tDI: XTX\t"
                  "DI:XX\tFlag\tNotes\n")

    if os.path.exists(databaselocation):
        with open(databaselocation, 'r+') as outputfile:
            if reload:
                outputfile.truncate()
            else:
                linelist = outputfile.readlines()
                for line in linelist[1:]:
                    skiplist.append(line.split('\t')[0])
            outputfile.seek(0, os.SEEK_END)
            if not outputfile.tell():
                outputfile.write(header_line)
        outputfile.close()
    else:
        print("Couldn't find databasepath; defaulting to reload = True")
        with open(databaselocation, 'w') as outputfile:
            outputfile.write(header_line)
            outputfile.write("\n".join(bigdatalist))
        outputfile.close()

    for hit_number, hit in enumerate(hitlist):
        skip_outer = False
        if hit.entry.identifier in skiplist:
            continue
        else:
            skiplist.append(hit.entry.identifier)
            try:
                start_time = time.perf_counter()
                flag = ""
                notes = []

                for name in hit.entry.chemical_name.split(" "):
                    if 'cyclopentadienyl' in name:
                        skip_outer = True
                if skip_outer:
                    continue
                if hit.entry.chemical_name:
                    organics = sanitize_name(hit.entry.chemical_name)
                else:
                    organics = "N/F"
                sanitize_name_time = time.perf_counter()
                timedict['sanitize_name'] += sanitize_name_time - start_time

                csd_code = hit.entry.identifier
                formula = hit.entry.crystal.formula
                moiety_formula = ", ".join([component.formula for component in hit.entry.crystal.molecule.components])
                diagram = ""
                year = hit.entry.publication[3]
                if hit.entry.publication[5]:
                    doi = "http://dx.doi.org/" + hit.entry.publication[5]
                else:
                    doi = "N/F"
                color = hit.entry.color
                crystal_description = hit.entry.habit
                space_group = hit.entry.crystal.spacegroup_symbol
                if hit.entry.r_factor:
                    r_factor = numpy.round(hit.entry.r_factor, decimals = 2)
                else:
                    r_factor = "N/F"
                diffraction_temp = hit.entry.temperature
                diffraction_temp_value = None
                if diffraction_temp:
                    for word in diffraction_temp.split(" "):
                        if not diffraction_temp_value:
                            try:
                                if word.endswith("deg.C"):
                                    truncated_word = word[:-5]
                                    diffraction_temp_value = float(truncated_word)
                                elif word.endswith("K"):
                                    truncated_word = word[:-1]
                                    diffraction_temp_value = float(truncated_word)
                                else:
                                    diffraction_temp_value = float(word)
                            except ValueError:
                                continue
                    if diffraction_temp.endswith("K"):
                        diffraction_temp = str(numpy.round(diffraction_temp_value, decimals=1))
                    elif diffraction_temp.endswith("C"):
                        diffraction_temp = str(numpy.round(diffraction_temp_value + 273.15, decimals=1))
                else:
                    diffraction_temp = "N/F"
                # Figure out where to get precision for cell lengths and angles
                cell_lengths = [float(numpy.round(cell_length, decimals=3))
                                for cell_length in hit.entry.crystal.cell_lengths]
                cell_angles = [float(numpy.round(cell_angle, decimals=3))
                               for cell_angle in hit.entry.crystal.cell_angles]
                z_value = hit.entry.crystal.z_value
                density = hit.entry.crystal.calculated_density
                reading_time = time.perf_counter()
                timedict['reading'] += reading_time - sanitize_name_time

                if hit.entry.crystal.has_disorder:
                    flag = "yellow"
                    notes.append("disorder")

                hit_structure = pymatgenize(hit)
                pymatgenize_time = time.perf_counter()
                timedict['pymatgenize'] += pymatgenize_time - reading_time

                hit_perovskite = Perovskite(hit_structure)
                perovskite_time = time.perf_counter()
                timedict['Perovskite'] += perovskite_time - pymatgenize_time

                metals = hit_perovskite.b
                metal_charge = None
                try:
                    metal_charge = numpy.round(find_metal_oxidation_states(hit), decimals=2)
                except ZeroDivisionError:
                    if any(element in metals for element in ['Pb', 'Sn', 'Ge']):
                        metal_charge = 2
                    elif any(element in metals for element in ['Bi', 'As', 'Sb', 'In']):
                        metal_charge = 3
                    elif any(element in metals for element in ['Ag', 'Tl']):
                        metal_charge = 1
                fourplus_metals = False
                fiveplus_metals = False
                for metal in metals:
                    if metal_charge != "N/F":
                        if metal == 'Pb':
                            if not numpy.isclose(metal_charge, 2, atol=0.001):
                                metal_charge = "N/F"
                        elif metal in ['Sn', 'Ge']:
                            fourplus_metals = True
                            if (not numpy.isclose(metal_charge, 2, atol=0.001)
                                    and not numpy.isclose(metal_charge, 4, atol=0.001)):
                                metal_charge = "N/F"
                        elif metal in ['As', 'Sb', 'Bi']:
                            fiveplus_metals = True
                            if (not numpy.isclose(metal_charge, 3, atol=0.001)
                                    and not numpy.isclose(metal_charge, 5, atol=0.001)):
                                metal_charge = "N/F"
                        elif metal in ['Ag', 'In', 'Tl']:
                            if (not numpy.isclose(metal_charge, 1, atol=0.001)
                                    and not numpy.isclose(metal_charge, 3, atol=0.001)):
                                metal_charge = "N/F"

                find_metal_oxidation_states_time = time.perf_counter()
                timedict['find_metal_oxidation_states'] += find_metal_oxidation_states_time - perovskite_time

                if isinstance(metal_charge, str):
                    hit_perovskite.identify_polyhedra()
                elif fourplus_metals and numpy.isclose(metal_charge, 4, atol=0.001):
                        hit_perovskite.identify_polyhedra(oxidized_metal=True)
                elif fiveplus_metals and numpy.isclose(metal_charge, 5, atol=0.001):
                        hit_perovskite.identify_polyhedra(oxidized_metal=True)
                else:
                    hit_perovskite.identify_polyhedra()
                identify_polyhedra_time = time.perf_counter()
                timedict['identify_polyhedra'] += identify_polyhedra_time - find_metal_oxidation_states_time

                if isinstance(metal_charge, str):
                    hit_perovskite.calc_dimensionality()
                elif fourplus_metals and numpy.isclose(metal_charge, 4, atol=0.001):
                    hit_perovskite.calc_dimensionality(oxidized_metal=True)
                elif fiveplus_metals and numpy.isclose(metal_charge, 5, atol=0.001):
                    hit_perovskite.calc_dimensionality(oxidized_metal=True)
                else:
                    hit_perovskite.calc_dimensionality()
                calc_dimensionality_time = time.perf_counter()
                timedict['calc_dimensionality'] += calc_dimensionality_time - identify_polyhedra_time

                hit_perovskite.connectivity()
                connectivity_time = time.perf_counter()
                timedict['connectivity'] += connectivity_time - calc_dimensionality_time

                tilting_angles = "N/A"
                try:
                    hit_perovskite.calc_distances_and_angles()
                    calc_distances_and_angles_time = time.perf_counter()
                    timedict['calc_distances_and_angles'] += calc_distances_and_angles_time - connectivity_time

                    hit_perovskite.calc_adjacency()
                    calc_adjacency_time = time.perf_counter()
                    timedict['calc_adjacency'] += calc_adjacency_time - calc_distances_and_angles_time

                    hit_perovskite.calc_distortion_indices()
                    calc_distortion_time = time.perf_counter()
                    timedict['calc_distortion_indices'] += calc_distortion_time - calc_adjacency_time

                    hit_perovskite.describe_structure()
                    describe_structure_time = time.perf_counter()
                    timedict['describe_structure'] += describe_structure_time - calc_distortion_time

                    description = hit_perovskite.description
                    terminal_index = numpy.round(hit_perovskite.normalized_connectivity_indices[0], decimals=4)
                    corner_index = numpy.round(hit_perovskite.normalized_connectivity_indices[1], decimals=4)
                    edge_index = numpy.round(hit_perovskite.normalized_connectivity_indices[2], decimals=4)
                    face_index = numpy.round(hit_perovskite.normalized_connectivity_indices[3], decimals=4)
                    if hit_perovskite.adjacency == 'N/A':
                        adjacency = 'N/A'
                    else:
                        adjacency = numpy.round(hit_perovskite.adjacency, decimals=4)
                    b_cations = hit_perovskite.b
                    x_anions = hit_perovskite.x
                    coordination_numbers = hit_perovskite.coordination_numbers
                    if hit_perovskite.cpp:
                        cpp = numpy.round(hit_perovskite.cpp, decimals=4)
                    else:
                        cpp = 'N/A'

                    distances_list = hit_perovskite.distances_and_angles[0]
                    distances_list = [item for sublist in distances_list for item in sublist]
                    bond_distances = numpy.round([numpy.min(distances_list), numpy.max(distances_list),
                                                  numpy.average(distances_list), nmad(distances_list)], decimals=3)
                    if numpy.max(distances_list) > 4.5:
                        flag = 'red'
                        notes.append(f"Very long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")
                    elif numpy.max(distances_list) > 4.0:
                        if flag != 'red':
                            flag = 'orange'
                        notes.append(f"Long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")
                    elif numpy.max(distances_list) > 3.5:
                        if flag != 'red' and flag != 'orange':
                            flag = 'yellow'
                        notes.append(f"Slightly long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")

                    if hit_perovskite.tilting_angles:
                        tilting_list = hit_perovskite.tilting_angles
                        tilting_angles = numpy.round([numpy.min(tilting_list), numpy.max(tilting_list),
                                                      numpy.average(tilting_list), nmad(tilting_list)], decimals=3)
                        tilting_angles[3] *= 100

                    dilist = hit_perovskite.distortion_indices
                    di_xt = 100*numpy.round([min(dilist[0]), max(dilist[0]), numpy.average(dilist[0]), nmad(dilist[0])],
                                        decimals=3)
                    if hit_perovskite.non_octahedral:
                        di_xx = 'N/A'
                        di_xtx = 'N/A'
                    else:
                        di_xx = 100*numpy.round([min(dilist[1]), max(dilist[1]), numpy.average(dilist[1]), nmad(dilist[1])],
                                            decimals=3)
                        di_xtx = 100*numpy.round(
                            [min(dilist[2]), max(dilist[2]), numpy.average(dilist[2]), nmad(dilist[2])], decimals=3)
                except (scipy.spatial._qhull.QhullError, IndexError, ZeroDivisionError) as error:
                    notes.append(f"Convex Hull Error")
                    if flag != "red":
                        flag = "orange"
                    print(f"Qhull Error for {hit.identifier}, error {error}.")
                    hit_perovskite.describe_structure()

                    description = hit_perovskite.description
                    terminal_index = numpy.round(hit_perovskite.normalized_connectivity_indices[0], decimals=4)
                    corner_index = numpy.round(hit_perovskite.normalized_connectivity_indices[1], decimals=4)
                    edge_index = numpy.round(hit_perovskite.normalized_connectivity_indices[2], decimals=4)
                    face_index = numpy.round(hit_perovskite.normalized_connectivity_indices[3], decimals=4)
                    adjacency = "N/F"
                    b_cations = hit_perovskite.b
                    x_anions = hit_perovskite.x
                    coordination_numbers = hit_perovskite.coordination_numbers
                    if hit_perovskite.cpp:
                        cpp = numpy.round(hit_perovskite.cpp, decimals=4)
                    else:
                        cpp = "N/F"
                    distances_list = hit_perovskite.distances_and_angles[0]
                    distances_list = [item for sublist in distances_list for item in sublist]
                    bond_distances = numpy.round([numpy.min(distances_list), numpy.max(distances_list),
                                                  numpy.average(distances_list), nmad(distances_list)], decimals=3)
                    # Scale nmad by 100 so it can be a percentage
                    bond_distances[3] *= 100
                    if numpy.max(distances_list) > 4.5:
                        flag = 'red'
                        notes.append(f"Very long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")
                    elif numpy.max(distances_list) > 4.0:
                        if flag != 'red':
                            flag = 'orange'
                        notes.append(f"Long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")
                    elif numpy.max(distances_list) > 3.5:
                        if flag != 'red' and flag != 'orange':
                            flag = 'yellow'
                        notes.append(f"Slightly long B...X bond: {numpy.round(numpy.max(distances_list), decimals=3)}")
                    di_xt = 100*numpy.round([min(dilist[0]), max(dilist[0]), numpy.average(dilist[0]), nmad(dilist[0])],
                                        decimals=3)
                    di_xx = "N/F"
                    di_xtx = "N/F"

                smalldatalist = [str(csd_code), str(formula), str(moiety_formula), str(organics), str(diagram), str(year),
                                 str(doi), str(description), str(terminal_index), str(corner_index), str(edge_index),
                                 str(face_index), str(adjacency), ", ".join(b_cations), ", ".join(x_anions), ", ".join(coordination_numbers),
                                 str(metal_charge), str(cpp), str(color), str(crystal_description), str(space_group),
                                 str(r_factor), str(diffraction_temp), str([cell_lengths[0], cell_lengths[1], cell_lengths[2]]),
                                 str([cell_angles[0], cell_angles[1], cell_angles[2]]), str(z_value),
                                 str(numpy.round(density, 4)), str(bond_distances), str(tilting_angles), str(di_xt),
                                 str(di_xtx), str(di_xx), str(flag), ", ".join(notes)]
                datastring = '\t'.join(smalldatalist) + "\n"

                # Now we try to save an image for the structure
                if images:
                    if not os.path.exists(databasepath + "\\" + "images"):
                        os.mkdir(databasepath + "\\" + "images")
                    savelocation = databasepath + rf"\images\{csd_code}.png"
                    if not os.path.exists(savelocation):
                        try:
                            image = imagegeneration(hit)
                            image.save(fp=savelocation)
                            print(f"Saved image for {csd_code}")
                        except AttributeError:
                            print(f"Image generation failed altogether for {csd_code}; skipping.")
                            pass

                # Next we try to save a cif
                if not os.path.exists(databasepath + "\\" + "CIFs"):
                    os.mkdir(databasepath + "\\" + "CIFs")
                cif_text = hit.entry.to_string(format="cif")
                cif_file_location = databasepath + "\\" + "CIFs" + "\\" + f"{csd_code}.cif"
                if not os.path.exists(cif_file_location):
                    with open(cif_file_location, 'w') as cif_file:
                        cif_file.write(cif_text)
                    cif_file.close()

                with open(databaselocation, 'r+', encoding="utf-8") as outputfile:
                    outputfile.seek(0, os.SEEK_END)
                    outputfile.write(datastring)
                outputfile.close()

                wrapping_up_time = time.perf_counter()
                if describe_structure_time:
                    timedict['wrapping up'] += wrapping_up_time - describe_structure_time

                print(f"Finished with {hit.identifier}, #{hit_number}")
            except (ValueError, TypeError, AttributeError, ZeroDivisionError) as error:
                print(f"Issue with {hit.identifier}, #{hit_number} ({error}); continuing.")
                failedlist.append(hit.identifier)

    print("Finished with inputted data!")
    end_time = time.perf_counter()
    print("Distribution of computation time:")
    for key in timedict:
        print(f"{key} : {numpy.round(100 * timedict[key] / sum(timedict.values()), decimals=1)}%.")
    print(f"Total time: {numpy.round(end_time - global_start_time, decimals = 1)} seconds")

def process(hit_or_filepath):
    if str(type(hit_or_filepath)) == "<class 'ccdc.search.CombinedSearch.CombinedHit'>":
        structure = pymatgenize(hit_or_filepath)
    elif str(type(hit_or_filepath)) == "<class 'str'>":
        structure = Structure.from_file(hit_or_filepath)
    else:
        structure = None
    perov = Perovskite(structure)
    perov.identify_polyhedra()
    perov.connectivity()
    perov.calc_adjacency()
    perov.calc_distances_and_angles()
    perov.calc_distortion_indices()
    perov.calc_dimensionality()
    perov.describe_structure()
    print(f"Processed structure: {structure.formula}\n")
    print(f"Structure Description: {perov.description}")
    print(f"Coordination Numbers: {perov.coordination_numbers}")
    print(f"Connectivity Indices: {numpy.round(perov.normalized_connectivity_indices, 4)}")
    if perov.adjacency:
        print(f"Adjacency: {numpy.round(perov.adjacency, 4)}")
    else:
        print("Adjacency: N/A")
