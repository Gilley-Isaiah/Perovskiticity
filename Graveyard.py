def calc_adjacency(self):
    """The calc_adjacency() function calculates an 'adjacency' parameter for a Perovskite object, which is the.
        average fraction of adjacent 'shared' vertices for each shared halide in the structure. This parameter is
        useful for distinguishing between structures that otherwise have the same connectivity indices, such as
        (100)- and (110)-sliced 2D perovskites.

        For example, a (100)-sliced 2D perovskite with the ABX4 formula has connectivity indices of [0.5, 0.5, 0, 0]
        and an adjacency of 0.5, while a (110)-sliced 2D perovskite with the same formula has the same connectivity
        but an adjacency of 0.625.

        Will not work properly unless Perovskite.identify_octahedra() has been called FIRST and succeeded normally.

        :param self: A Perovskite object.
        :return: Returns nothing, but builds the Perovskite.adjacency property, which is either a string with 'N/A'
        if the structure or a float 0 <= x <= 1 if the structure has some sharing.
        """
    # To start with, there are a lot of duplicates in our adjacency_list, so we are going to go through the sites
    # in the structure and average all of the adjacency entries for each site, then the total adjacency is the
    # average across all of the sites in the structure.

    # Some massaging of the self.shared_X_sites may be needed since sometimes there may be points in that set which
    # don't belong to any of the 'octahedra' inside the unit cell (e.g. if the octahedron to which they belong has
    # its metal outside the unit cell but the X site is in the unit cell).
    shared_x_sites = list(self.shared_X_sites)
    shunted_x_sites = []
    unique_octahedral_sites = numpy.unique([numpy.round(numpy.dot(site, self.structure.lattice.inv_matrix), decimals=7)
                                            for octahedron in self.octahedra for site in octahedron], axis=0)
    for index, point in enumerate(shared_x_sites):
        if any([all(numpy.isclose(point, coordinate, atol=0.0001)) for coordinate in unique_octahedral_sites]):
            continue
        else:
            # This product function makes cartesian products of sets with fixed length (repeat), so the below
            # line is generating every translation needed to move a point inside the unit cell into the 3x3x3
            # supercell.
            supercell_translations = list(product({-1, 0, 1}, repeat=3))
            translation_found = False
            for coordinate in unique_octahedral_sites:
                if translation_found:
                    break
                for translation in supercell_translations:
                    if all(numpy.isclose(point + translation, coordinate, atol=0.0001)):
                        shared_x_sites.append(point + translation)
                        shunted_x_sites.append(point)
                        print(f"Shunting {point} to {point + translation}.")
                        translation_found = True
                        break
    self.shared_X_sites = numpy.array(shared_x_sites)

    # hull.points[i] = octahedron[i+1]
    if self.normalized_connectivity_indices[0] == 1:
        self.adjacency = 'N/A'
    else:
        adjacency_list = []
        for index, octahedron in enumerate(self.octahedra):
            hull = ConvexHull(octahedron[1:])

            shared_idx_list = []
            shared_outside_unit_cell_idx_list = []
            for idx, point in enumerate(hull.points):
                # Note hull.points[i] = octahedron[i+1] because the indexes are preserved.
                frac_point = numpy.dot(octahedron[idx + 1], self.structure.lattice.inv_matrix)
                if any([all(numpy.isclose(frac_point, coordinate, atol=0.0001))
                        for coordinate in self.shared_X_sites]):
                    shared_idx_list.append(idx)
                    continue
                elif any([all(numpy.isclose(frac_point, coordinate, atol=0.0001))
                          for coordinate in shunted_x_sites]):
                    shared_idx_list.append(idx)
                    shared_outside_unit_cell_idx_list.append(idx)
                    continue
                elif any(numpy.round(frac_point, decimals=7) < 0) or any(numpy.round(frac_point, decimals=7) >= 1):
                    shunted_frac_point = [None, None, None]
                    for value, coordinate in enumerate(frac_point):
                        if numpy.round(coordinate, 7) >= 1:
                            shunted_frac_point[value] = numpy.round(coordinate - 1, 7)
                        elif numpy.round(coordinate, 7) < 0:
                            shunted_frac_point[value] = numpy.round(coordinate + 1, 7)
                        elif 1 > numpy.round(coordinate, 7) >= 0:
                            shunted_frac_point[value] = numpy.round(coordinate, 7)
                    if any([all(numpy.isclose(shunted_frac_point, coordinate, atol=0.0001))
                            for coordinate in self.shared_X_sites]):
                        shared_idx_list.append(idx)
                        shared_outside_unit_cell_idx_list.append(idx)

            for idx in shared_idx_list:
                if idx not in shared_outside_unit_cell_idx_list:
                    adjacent_vertices = []
                    adjacent_shared_vertices = []
                    for simplex in hull.simplices:
                        if idx in simplex:
                            for vertex in simplex:
                                if (vertex != idx) and (vertex not in adjacent_vertices):
                                    adjacent_vertices.append(vertex)
                                    if vertex in shared_idx_list:
                                        adjacent_shared_vertices.append(vertex)
                    point_adjacency = len(adjacent_shared_vertices) / len(adjacent_vertices)
                    adjacency_list.append([hull.points[idx], point_adjacency])

        unique_adjacency_list = numpy.unique([entry[0] for entry in adjacency_list], axis=0)
        adjacency_average = []

        for entry in unique_adjacency_list:
            count = 0
            adjacency = 0
            for entry2 in adjacency_list:
                if all([numpy.isclose(entry[x], entry2[0][x], atol=0.0001) for x in range(len(entry))]):
                    adjacency += entry2[1]
                    count += 1
            if count != 0:
                adjacency = adjacency / count
                adjacency_average.append(adjacency)

        if not adjacency_average:
            self.adjacency = 0
        elif adjacency_average:
            self.adjacency = numpy.round(float(numpy.average(adjacency_average)), 5)


def connectivity(self):
        """The connectivity() function calculates a set of indices describing the amount and type of polyhedral
            sharing within a structure (i.e. terminal, corner-sharing, edge-sharing, or face-sharing). It is
            intended to help determine types of octahedral structures including perovskites and perovskitoids but
            the theory of it applies to any polyhedral structure.

            Will not work properly unless Perovskite.identify_octahedra() has been called FIRST and succeeded normally.

            :param self: A Perovskite object.

            It builds the property Perovskite.connectivity_indices, a list of numbers including [T, C, E, F]
            as well as the property Perovskite.tilting_anlges, a list of interpolyhedral
            angles. Note that these are only calculated for vertices shared by two polyhedra. Vertices
            shared by more than two polyhedra are considered in the calculation of connectivity indices
            but not in the calculation of tilting angles.
            """
        # We begin by removing any organic components of the structure to save computing time. This is
        # necessary because we iterate multiple times over every atom in a 3x3x3 supercell, which can become
        # taxing for large unit cells with many atoms. Removing organic components typically eliminates >50% of
        # the atoms in the structure, dramatically reducing the number of calculations needed.
        structure = self.structure
        self.perovset = set(self.x + self.b)
        self.shared_X_sites = list()
        self.cartesian_shared_X_sites = list()

        terminal = 0
        corner_shared = 0
        edge_shared = 0
        face_shared = 0
        tilting_angles = []
        tiltinglist = []

        # We make a 3x3x3 supercell from the unit cell to account for 'sharing' across unit-cell boundaries.
        # There may be a more elegant way to do this involving the neighbors() function from Pymatgen.
        supercell = self.simplestructure.make_supercell(3, in_place=False)
        superperovskite = Perovskite(supercell, suppress_output=True)
        # We make a list of 'superoctahedra' from the supercell.
        superperovskite.identify_octahedra()
        superoctahedra = superperovskite.frac_octahedra
        # Now we make a list of only the octahedra with at least one atom inside the interior unit cell of the
        # 3x3x3 supercell. This amounts to having fractional coordinates between 1/3 and 2/3.
        octahedra = []
        for superoctahedron in superoctahedra:
            boolarray = []
            for site in superoctahedron:
                boolarray.append(all([round(1 / 3, 7) <= round(coord, 7) < round(2 / 3, 7)
                                      for coord in site]))
            if any(boolarray):
                octahedra.append(superoctahedron)
        # Most of the calculation happens within this for loop. We go through each octahedron from the list
        # and iterate over each X anion which is inside the interior unit cell of the 3x3x3 supercell.
        for octahedron in octahedra:
            for x in octahedron[1:]:
                if all([round(1 / 3, 7) <= round(coord, 7) < round(2 / 3, 7) for coord in x]):
                    # We will need to account for the fact that shared anions are going to be counted more
                    # than once. This problem is handled by tracking how many octahedra share the anion and
                    # later dividing by a factor based on this number.
                    shared = 1
                    # We now make a list of all the other octahedra that have this exact X anion.
                    sharedoctahedra = []
                    for superoctahedron in superoctahedra:
                        for superx in superoctahedron[1:]:
                            # We look through all the anions of the other octahedra and check whether they
                            # have the exact position of the anion of interest. If they do, we consider
                            # that anion to be shared between those octahedra.
                            if list(x) == list(superx) and not \
                                    list(octahedron[0]) == list(superoctahedron[0]):
                                shared += 1
                                sharedoctahedra.append(superoctahedron)
                    # If we find that no octahedra share that X anion, assume that it is terminal.
                    if not sharedoctahedra:
                        terminal += 1
                    else:
                        # Now, we need to figure out whether the X anion in question is part of a
                        # corner-sharing, edge-sharing, or face-sharing motif.
                        for sharedoctahedron in sharedoctahedra:
                            sharing = 0
                            # We count how many anions are shared between each octahedron that has the X
                            # anion  of interest. Again, we do this by comparing their fractional coordinates.
                            for x_2 in octahedron[1:]:
                                for superx_2 in sharedoctahedron[1:]:
                                    if list(x_2) == list(superx_2):
                                        sharing += 1
                            # If the two octahedra share only one anion, it must be corner-shared. If they
                            # share two, it is edge-shared, and if they share three, it is face-shared.
                            # If the two octahedra share more than three anions, it is likely something went
                            # wrong, and we throw an error to flag the structure for checking.
                            if sharing == 1:
                                corner_shared = corner_shared + 1 / (shared * (shared - 1))
                            if sharing == 2:
                                edge_shared = edge_shared + 1 / (shared * (shared - 1))
                            if sharing >= 3:
                                face_shared = face_shared + 1 / (shared * (shared - 1))

                        # Needs to be 3x-1 because x is a fractional coordinate in the supercell, and that expression
                        # transforms it into the fractional coordinate in the original cell.
                        self.shared_X_sites.append(3*x - 1)
                        self.cartesian_shared_X_sites.append(numpy.dot(3*x-1, self.structure.lattice.matrix))

                    # Now we write the coordinates (Cartesian) of the anions shared by two octahedra to a
                    # list for calculating tilting angles.
                    if shared == 2:
                        x_cart = numpy.dot(x, structure.lattice.matrix)
                        oct_cart = numpy.dot(octahedron[0], structure.lattice.matrix)
                        sharedoct_cart = numpy.dot(sharedoctahedra[0][0], structure.lattice.matrix)
                        tiltinglist.append([x_cart, oct_cart, sharedoct_cart])

        delcount = 0
        # We delete duplicates from the list.
        for j in range(len(tiltinglist)):
            for entry2 in tiltinglist:
                if tiltinglist[j - delcount][0][0] == entry2[0][0] and tiltinglist[j - delcount][0][1] == \
                        entry2[0][1] and tiltinglist[j - delcount][0][2] == entry2[0][2] \
                        and not ((tiltinglist[j - delcount][1][0] == entry2[1][0]
                                  and tiltinglist[j - delcount][1][1] == entry2[1][1] and
                                  tiltinglist[j - delcount][1][2] == entry2[1][2]) and
                                 (tiltinglist[j - delcount][2][0] == entry2[2][0] and
                                  tiltinglist[j - delcount][2][1] == entry2[2][1] and
                                  tiltinglist[j - delcount][2][2] == entry2[2][2])):
                    del (tiltinglist[j - delcount])
                    delcount += 1
        # Angles are calculated by drawing the two X -> B vectors and computing the angle between them.
        for entry in tiltinglist:
            u = entry[1] - entry[0]
            v = entry[2] - entry[0]
            angle_radians = numpy.arccos(numpy.dot(u, v) / (numpy.linalg.norm(u) * numpy.linalg.norm(v)))
            angle_degrees = math.degrees(angle_radians)
            tilting_angles.append(numpy.round(angle_degrees, decimals=2))
        for idx, angle in enumerate(tilting_angles):
            if math.isnan(angle):
                tilting_angles[idx] = 180

        shared_x_sites_no_duplicates = numpy.unique(self.shared_X_sites, axis=0)
        cartesian_x_sites_no_duplicates = numpy.unique(self.cartesian_shared_X_sites, axis=0)
        self.shared_X_sites = shared_x_sites_no_duplicates
        self.cartesian_shared_X_sites = cartesian_x_sites_no_duplicates
        self.connectivity_indices = numpy.round([terminal, corner_shared, edge_shared, face_shared], decimals=7)
        self.normalized_connectivity_indices = self.connectivity_indices / sum(self.connectivity_indices)
        self.tilting_angles = tilting_angles