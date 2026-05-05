import ccdc
import time
from ccdc import search

#For timing
start_time = time.perf_counter()

#Note, you can make an OR by putting a list inside ONE query
#You can make an AND by putting the atoms in separate queries
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
#Make a bond query (can be included or excluded later, checks for anything with a bond from first group to second group)
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

#Make an atom query (can be included or excluded later, checks for those atoms)
forbidden_atoms_query = ccdc.search.QuerySubstructure()
forbidden_atoms_query.add_atom(forbidden_atoms)

#It seems that additive groups (like allowed bonds here) need to be actually ran as a search with
#max_hits_per_structure=1, otherwise you get a lot of duplicates, even if using the proper instruction later in the
#combined search. I am not sure why this is necessary here and not for the excluded groups (like forbidden bonds group).
allowed_bonds_search = ccdc.search.SubstructureSearch()
allowed_bonds_search.add_substructure(allowed_query)
allowed_bonds_search.search(max_hits_per_structure=1)

forbidden_bonds_search = ccdc.search.SubstructureSearch()
forbidden_bonds_search.add_substructure(forbidden_bonds_query)

forbidden_atoms_search = ccdc.search.SubstructureSearch()
forbidden_atoms_search.add_substructure(forbidden_atoms_query)

combined_search = search.CombinedSearch(allowed_bonds_search & -forbidden_bonds_search & -forbidden_atoms_search)

#Here, you can actually even leave out the max_hits_per_structure=1 command; it doesn't affect the result or runtime.
hits = combined_search.search()
print(len(hits))


end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Time: {round(elapsed_time, 1)} seconds")

#substructure_search= ccdc.search.SubstructureSearch()
#substructure_search.add_substructure(short_query)
#hits = susbtructure_search.search()
#print(len(hits))

#The hits list contains SubstructureHit objects with properties:
    #identifier (CSD code)
    #entry (Entry object, see below)
    #crystal (Crystal object)
    #molecule (Molecule object)

#Entries have LOTS of properties. Here are some useful ones:
    #Identifiers:
        #identifier (CSD code)
        #chemical_name
        #deposition_date
        #publication
    #Descriptors:
        #calculated_density
        #color
        #formula
        #habit
        #is_organic
        #is_organometallic
        #is_polymeric
        #to_string
    #Crystallographic Stuff:
        #r_factor
        #radiation_source
        #refinement_goodness_of_fit
        #a bunch of other stuff for refinement, including _max_shift, _number_of_contstraints, _number_of_parameters,
        #_residual_electron_density_max, _residual_electron_density_min, _weighted_r_factor
        #reflection_max_theta
        #temperature
        #disorder_details
        #has_3d_structure (Boolean, whether entry has coordinates)
        #has_disorder

#Stuff you can get from a Crystal object:
    #Crystal.molecule
    #Crystal.cell_lengths - list of a, b, c
    #Crystal.cell_angles - list of alpha, beta, gamma
    #Crystal.spacegroup_symbol
    #Crystal.symmetry_operators
    #Crystal.is_centrosymmetric
    #Crystal.is_sohncke

#Stuff you can get from a Molecule object:
    #components (list of sub-molecules)
    #atoms (list of atoms)
    #formula
    #heaviest_component

#Stuff you can get from an Atom object:
    #coordiantes (see below)
    #atomic_symbol
    #atomic_number
    #bonds
    #is_chiral
    #label
    #occupancy
    #neighbors
    #partial_charge
    #

#To get coordinates:
    #hit.entry.crystal.molecule.atoms
    #for each atom the coordinate is atom.coordinates - list of 3, x, y, z in Angstroms (orthogonal space)
    #fractional coordinates given by fractional_coordinates
    #you can get coordinates by index atom.coordinates.x or by key atom.coordinates[0]
    #can get uncertainties with fractional_uncertainties instead of fractional_coordinates
    #can also get displacement parameters atom.displacement_parameters.values




