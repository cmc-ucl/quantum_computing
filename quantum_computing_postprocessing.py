# Bruno Camino 19/07/2022

import numpy as np
import pandas as pd
import copy
import sys
sys.path.append('../')

from quantum_computing_functions import *

def cart2sph_array(array):
    sph_coordinates = []
    for line in array:
        x = line[0]
        y = line[1]
        z = line[2]
        hxy = np.hypot(x, y)
        r = np.hypot(hxy, z)
        el = np.arctan2(z, hxy)
        az = np.arctan2(y, x)
        if np.around(az,6) ==  np.around(2*np.pi,6) \
        or np.around(az,6) ==  -np.around(2*np.pi,6):
            az = 0.
        if np.around(az,6) < 0.:
            az = np.round(2*np.pi+az,6)
        sph_coordinates.append([round(r,6), round(el,6), round(az,6) ])   
    return sph_coordinates


def classifier(structures, max_shell=2):
    centered_sph_coords = []
    centered_sph_coords_structure = []
    neighbours_spatial_dist = []
    neighbours_spatial_dist_all = []
    shells = np.unique(np.round(structures[0].distance_matrix[0],decimals=6),return_counts=True)[0].tolist()

    for k,structure in enumerate(structures): 
        neighbours_spatial_dist = []
        
        for j in range(structure.num_sites):
            centered_sph_coords = []
            neighbours_spatial_dist_atom = []
            
            for n in range(max_shell+1):
                atom_indices = np.where(np.round(structure.distance_matrix[j],5) == np.round(shells[n],5))[0].tolist()
                centered_sph_coords = []
                for i in atom_indices:

                    translation_vector = structure.sites[j].distance_and_image(structure.sites[i])[1]
                    new_cart_coords = structure.cart_coords[i]+(translation_vector*structure.lattice.abc)
                    centered_cart_coords = new_cart_coords-structure.cart_coords[j] 

                    centered_sph_coords.append(cart2sph(centered_cart_coords[0],centered_cart_coords[1],centered_cart_coords[2]))        

                spatial_distribution = np.argsort(np.array(centered_sph_coords)[:,1]*10 +\
                                                np.array(centered_sph_coords)[:,0])


                neighbours_spatial_dist_atom.extend((np.array(structure.atomic_numbers)[np.array(atom_indices)[spatial_distribution]]).tolist())
            neighbours_spatial_dist.append(neighbours_spatial_dist_atom)
        neighbours_spatial_dist_all.append(neighbours_spatial_dist) 

        #Sort
        neighbours_spatial_dist_all_sorted = []
        sorting = []

        for k,structure in enumerate(structures):
            sorted_atoms = []
            for i in range(len(neighbours_spatial_dist_all[0])):
                sorted_atoms.append(int(''.join([str(x) for x in neighbours_spatial_dist_all[k][i]])))
            sorting.append(np.argsort(np.array(sorted_atoms)))    
            neighbours_spatial_dist_all_sorted.append((np.array(neighbours_spatial_dist_all)[k][np.argsort(np.array(sorted_atoms))]).tolist())
        neighbours_spatial_dist_all_sorted = np.array(neighbours_spatial_dist_all_sorted)    

        #Slice
        neighbours_spatial_dist_all_sorted_sliced = neighbours_spatial_dist_all_sorted[:,:,1:]

        #Flatten
        n_structures = neighbours_spatial_dist_all_sorted_sliced.shape[0]
        vector_len = neighbours_spatial_dist_all_sorted_sliced.shape[1] * neighbours_spatial_dist_all_sorted_sliced.shape[2]
        neighbours_spatial_dist_all_sorted_sliced_flat = \
        np.reshape(neighbours_spatial_dist_all_sorted_sliced, [n_structures,vector_len])

        #Reduce

        return neighbours_spatial_dist_all_sorted_sliced_flat  


def convert_df_binary2atom(dataframe,species):
    
    df  = copy.deepcopy(dataframe)
    
    if isinstance(df.columns[0], int):
        num_sites = sum([type(x) == int for x in df.columns])
    elif isinstance(df.columns[0], str):
        num_sites = sum([x.isdigit() for x in df.columns])
    
    
    
    convert = lambda x: np.array(species)[x]
    
    labels = df.iloc[:,0:num_sites].to_numpy()
    
    new_labels = convert(labels)

    for i in range(num_sites):
        if isinstance(df.columns[0], int):
            df.loc[:,int(i)] = new_labels[:,i]
        elif isinstance(df.columns[0], str):
            df.loc[:,str(i)] = new_labels[:,i]
    
    return df


def convert_df_binary2atom_discrete(dataframe,species,remove_unfeasible=True):
    
    num_species = len(species)
    
    df  = copy.deepcopy(dataframe)
    
    if type(df.columns[0]) is str:
        num_sites = int(sum([x.isdigit() for x in df.columns])/num_species)
    else:
        num_sites = int(sum([type(x) == int for x in dataframe.columns])/num_species)
    
    sites = df.iloc[:,0:num_sites].to_numpy()

    test_sum = sites[:,::num_species]

    for i in range(1,2):
        test_sum += sites[:,i::num_species]
    
    
    unfeasible = np.where(np.prod(test_sum,axis=1) != 1)[0]
    if isinstance(df.index[0], str): 
        unfeasible = [str(x) for x in unfeasible]
    elif isinstance(df.index[0], int):
        unfeasible = [int(x) for x in unfeasible]

    df.drop(unfeasible, inplace=True)
    
    convert = lambda x: np.array(species)[x]
    
    labels = df.iloc[:,0:num_sites*num_species].to_numpy()[:,::num_species]
    new_labels = convert(labels)
    if isinstance(df.columns[0], int): 
        columns_drop = [int(x) for x in np.arange(num_sites,num_sites*num_species)]
    elif isinstance(df.columns[0], str):
        columns_drop = [str(x) for x in np.arange(num_sites,num_sites*num_species)]

    
    df_1 = df.drop(columns_drop,axis=1,inplace=False)

    for i in range(new_labels.shape[1]):
        if isinstance(df.columns[0], int):
            df_1.loc[:,int(i)] = new_labels[:,i]
        elif isinstance(df.columns[0], str):
            df_1.loc[:,str(i)] = new_labels[:,i]
    
    return df_1


def df2structure(df,structure):
    
    from pymatgen.core.structure import Structure

    num_sites = structure.num_sites
    lattice = structure.lattice
    atom_position = structure.cart_coords
    
    configurations = df.iloc[:,0:num_sites].to_numpy()
    
    zero_elements = np.where(configurations == 0) 
    vacancies = False
    if len(zero_elements[0]) > 0:
        configurations[zero_elements] = 99
        vacancies = True
    
    all_structures = []
    for config in configurations:
        all_structures.append(Structure(lattice, config, atom_position, coords_are_cartesian=True))
    
    if vacancies == True:
        for structure in all_structures:
            structure.remove_species([99])
    
    return all_structures


def display_low_E_structures(structure,energies,configurations, min_energy = 0, view = False):
    
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.visualize import view
    import numpy as np
    import copy

    low_energy, low_config = find_unique_E_structures(energies,configurations,min_energy = min_energy)
    
    low_energy_structures = []
    
    for i in low_config:
        structure_2 = copy.deepcopy(structure)
        for j in np.where(configurations[i] == 0)[0]:
            structure_2.replace(j,1)
        low_energy_structures.append(AseAtomsAdaptor().get_atoms(structure_2))
        
        if view == True:
            view(AseAtomsAdaptor().get_atoms(structure_2))
    
    return low_energy_structures


def display_structures(structure, dataframe, index, discrete=False):
    
    #structure = pymatgen structure object
    #dataframe = sampleset.to_pandas_dataframe()
    #index = list of indices of structures to visualise

    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.visualize import view
    import copy

    num_atoms = structure.num_sites
    
    if discrete == True:
        for i in index:
            structure_2 = copy.deepcopy(structure)
            for j in dataframe.iloc[i,0:num_atoms].to_numpy():
                structure_2.replace(j,1)
    elif discrete == False:
        for i in index:
            structure_2 = copy.deepcopy(structure)
            for j in np.where(dataframe.iloc[i,0:num_atoms] == 0)[0]:
                structure_2.replace(j,1)
            
            view(AseAtomsAdaptor().get_atoms(structure_2))


def find_all_structures(dataframe, min_energy = 0., return_count = False, sort_config = False, sort_energy = False):
    
    import numpy as np

    n_atoms = len([i for i in dataframe.columns if type(i) == int])
    
    energies = dataframe['energy'].to_numpy()
    
    below_min_energy = np.where(energies < min_energy)[0]

    config = dataframe.iloc[:,0:n_atoms].to_numpy()[below_min_energy]

    energies = energies[below_min_energy]

    multiplicity = dataframe['num_occurrences'].to_numpy()[below_min_energy]
    
    if sort_config == True and sort_energy == False:
        config_sorted = []
        for i, conf in enumerate(config):
            config_sorted.append(int(''.join([str(x) for x in conf])))

        sorting = np.argsort(np.array(config_sorted))   
 
        config = config[sorting]
        energies = energies[sorting]
        multiplicity = multiplicity[sorting]
    
    elif sort_energy == True:
        print('Implement this first')
        
    return config, energies, multiplicity


def find_coordination(structure, configurations, structure_indices, return_analysis = False):
    # Find the coordination of each site

    import numpy as np
    import copy

    first_neighbour_dist = np.round(np.unique(structure.distance_matrix[0])[1],5)
    
    neighbours = []
    for i in structure_indices:
        neigh = []
        structure_2 = copy.deepcopy(structure)
        for j in np.where(configurations[i] == 0)[0]:
            structure_2.replace(j,1)
        
        for atom in range(structure_2.num_sites):
            bonded = np.where(build_adjacency_matrix(structure)[atom] == 1.)[0]
            if structure_2.atomic_numbers[atom] == 1:
                neigh.append(-1)
            else:
                neigh.append(int(np.sum(np.array(structure_2.atomic_numbers)[bonded])/6))
        
        neighbours.append(neigh)
    
    analysis = []
    max_bond = max_n_bonds = np.sum(adjacency_matrix[0])
    if return_analysis == True:
        for line in np.array(neighbours):
            analysis_tmp = []
            for i in range(max_bond):
                analysis_tmp.append(np.sum(line == max_bond-i))
            analysis.append(analysis_tmp)
        return neighbours, analysis

    
    return neighbours


def find_energy_distribution(dataframe, remove_broken_chains = False, only_feasible = False, vacancies = 0):
    
    if remove_broken_chains == True:
        df = dataframe[dataframe['chain_break_fraction'] == 0.]
    elif remove_broken_chains == False:
        df = dataframe
    
    if only_feasible == True:       
        num_atoms = sum([x.isdigit() for x in df.columns])    
        all_config = df.iloc[:,0:num_atoms].to_numpy()
        multiplicity = df['num_occurrences'].to_numpy()
        sum_vector = np.sum(all_config,axis=1)
        feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - vacancies),5))[0]        
        df = df.iloc[feasible_config]
    
    energy = df['energy']
    energy = np.round(np.array(energy),5)
    multiplicity = df['num_occurrences']
    multiplicity = np.array(multiplicity)

    unique_multiplicity = []

    unique_energy = np.unique(energy)
    
    for e in unique_energy:
        pos = np.where(energy == np.round(e,5))[0]
        unique_multiplicity.append(np.sum(multiplicity[pos]))
    
    return unique_energy, unique_multiplicity


def find_energy_distribution_discrete(dataframe, remove_broken_chains = False, only_feasible = False, vacancies = 0):
    
    if remove_broken_chains == True:
        df = dataframe[dataframe['chain_break_fraction'] == 0.]
    elif remove_broken_chains == False:
        df = dataframe
    
    if only_feasible == True:       
            num_atoms = sum([x.isdigit() for x in df.columns])
            all_config = df.iloc[:,0:num_atoms].to_numpy()
            multiplicity = df['num_occurrences'].to_numpy()
            acceptable_config = np.all((all_config[:,::2]+all_config[:,1::2])-np.ones(18)==0,axis=1)  

            num_atoms = sum([x.isdigit() for x in df.columns])
            all_config = df.iloc[:,0:num_atoms].to_numpy()
            multiplicity = df['num_occurrences'].to_numpy()
            sum_vector = np.sum(all_config[:,::2],axis=1)
            feasible_configurations = np.round(sum_vector,5) == np.round((num_atoms/2 - vacancies),5)
            feasible_config = np.where((feasible_configurations * acceptable_config) == True)[0]
            df = df.iloc[feasible_config]





    
    energy = df['energy']
    energy = np.round(np.array(energy),5)
    multiplicity = df['num_occurrences']
    multiplicity = np.array(multiplicity)

    unique_multiplicity = []

    unique_energy = np.unique(energy)
    
    for e in unique_energy:
        pos = np.where(energy == np.round(e,5))[0]
        unique_multiplicity.append(np.sum(multiplicity[pos]))
    
    return unique_energy, unique_multiplicity


def find_multiplicity(all_structures,descriptor_unique=[None]):
    
    descriptor = build_descriptor(all_structures)
    
    if len(descriptor_unique) > 0:
        descriptor_unique, descriptor_first, descriptor_count = \
                                    np.unique(descriptor, axis=0,return_counts=True, return_index=True)
    group_structures = []
    for desc in descriptor_unique:
        structure_desc = []
        for i,d in enumerate(descriptor):
            if np.all(np.array(desc) == np.array(d)):
                structure_desc.append(i)
        group_structures.append(len(structure_desc))
        
    group_structures = np.array(group_structures)
    
    return group_structures


def find_vacancy_distribution(dataframe, remove_broken_chains = False, vacancies = 0):
    
    if remove_broken_chains == True:
        df = dataframe[dataframe['chain_break_fraction'] == 0.]
    elif remove_broken_chains == False:
        df = dataframe
    
    
    num_atoms = sum([x.isdigit() for x in df.columns])    
    all_config = df.iloc[:,0:num_atoms].to_numpy()

    np.unique(np.sum(all_config,axis=1), return_counts=True)
    
    vacancies = num_atoms-np.sum(df.iloc[:,0:18].to_numpy(),axis=1)

    unique_vacancies = np.unique(vacancies)
    
    multiplicity = df['num_occurrences']
    multiplicity = np.array(multiplicity)

    unique_multiplicity = []

    
    for v in unique_vacancies:
        pos = np.where(vacancies == np.round(v,5))[0]
        unique_multiplicity.append(np.sum(multiplicity[pos]))

    return unique_vacancies, unique_multiplicity


def find_equivalent_energy_distribution(dataframe, energy,remove_broken_chains = False, sort_config = True):
    
    #Only tested for 1 vacancy
    
    if remove_broken_chains == True:
        df = dataframe[dataframe['chain_break_fraction'] == 0.]
    elif remove_broken_chains == False:
        df = dataframe
    
    n_atoms = sum([x.isdigit() for x in df.columns])
    
    energies = df['energy']
    energies = np.round(np.array(energies),6)
    
    all_config = df.iloc[:,0:n_atoms].to_numpy()
    all_multiplicity = df['num_occurrences'].to_numpy()

    config_pos = np.where(energies == np.round(energy,6))[0]

    configurations = all_config[config_pos]

    multiplicity = all_multiplicity[config_pos]
    
    if sort_config == True:
        config_sorted = []
        for i, conf in enumerate(configurations):
            config_sorted.append(int(''.join([str(x) for x in conf])))

        sorting = np.argsort(np.array(config_sorted))   
 
        config = configurations[sorting]
        mult = multiplicity[sorting]
    
        return config, mult
    
    elif sort_config == False:
        
        return configurations, multiplicity


def find_num_broken_bonds(dataframe, remove_broken_chains = False, only_feasible = True, vacancies = 0):
    # Find the number of broken bonds per structure
    
    if remove_broken_chains == True:
        df = dataframe[dataframe['chain_break_fraction'] == 0.]
    elif remove_broken_chains == False:
        df = dataframe

    num_atoms = sum([type(x) == int for x in dataframe.columns])
    
    unique_energies = find_energy_distribution(dataframe, remove_broken_chains = remove_broken_chains,
                        only_feasible = only_feasible, vacancies = vacancies)[0]
    
    num_broken_bonds =  - np.array(unique_energies) + (num_atoms * -1.5)
    
    return num_broken_bonds


def find_ratio_acceptable(dataframe, species, remove_broken_chains = False, total_reads = None):

    # This refers to the one-hot encoding

    if total_reads == None:
        
        total_reads = np.sum(dataframe['num_occurrences'].to_numpy())
        df = convert_df_binary2atom_discrete(dataframe,species,remove_unfeasible=True)
        reads_cleaned = np.sum(df['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads
    else:
        reads_cleaned = np.sum(dataframe['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads


def find_ratio_acceptable_old(dataframe, remove_broken_chains = False):
    
    # remove_broken_chains = True : ratio wrt the total number of non broken chains solutions
    # remove_broken_chains = False : ratio wrt the total number of reads
    
    if remove_broken_chains == True:
        total_reads = np.sum(dataframe[dataframe['chain_break_fraction'] == 0.]['num_occurrences'])
        df = dataframe[dataframe['chain_break_fraction'] == 0]
    elif remove_broken_chains == False:
        total_reads = np.sum(dataframe['num_occurrences'])
        df = dataframe
    
    if len(df) == 0:
        return 0.
    
    else:
        
        num_atoms = sum([x.isdigit() for x in df.columns])
        all_config = df.iloc[:,0:num_atoms].to_numpy()
        multiplicity = df['num_occurrences'].to_numpy()
        acceptable_config = np.where(np.all((all_config[:,::2]+all_config[:,1::2])-np.ones(18)==0,axis=1) 
                             == True )[0]

        total_acceptable = np.sum(multiplicity[acceptable_config])
        ratio_acceptable = total_acceptable/total_reads

        return np.round(ratio_acceptable,4)


def find_ratio_broken_chains(dataframe):
    # Return how many solutions returned a broken chain
    
    broken = np.sum(dataframe[dataframe['chain_break_fraction'] != 0.]['num_occurrences'].to_numpy())
    total = np.sum(dataframe['num_occurrences'].to_numpy())
    
    return np.round(broken/total,4)


def find_ratio_feasible(dataframe, species, concentration, remove_broken_chains = False, total_reads = None):
    
    # This refers to the concentration

    # remove_broken_chains = True : ratio wrt the total number of non broken chains solutions
    # remove_broken_chains = False : ratio wrt the total number of reads
    
    if total_reads == None:
        
        total_reads = np.sum(dataframe['num_occurrences'].to_numpy())
        df = remove_unfeasible_solutions(dataframe,species,concentration)
        reads_cleaned = np.sum(df['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads
    else:
        reads_cleaned = np.sum(dataframe['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads


    '''OLD
    if total_reads != None:
        total_reads = num_reads
        df = dataframe
    else:
        if remove_broken_chains == True:
            total_reads = np.sum(dataframe[dataframe['chain_break_fraction'] == 0.]['num_occurrences'])
            df = dataframe[dataframe['chain_break_fraction'] == 0]
        elif remove_broken_chains == False:
            total_reads = np.sum(dataframe['num_occurrences'])
            df = dataframe
    
    if len(df) == 0:
        return 0.
    
    else:
        if type(df.columns[0]) is str:
            num_atoms = sum([x.isdigit() for x in df.columns]) 
        else:
            num_atoms = sum([type(x) == int for x in dataframe.columns])
        all_config = df.iloc[:,0:num_atoms].to_numpy()
        multiplicity = df['num_occurrences'].to_numpy() 
        sum_vector = np.sum(all_config,axis=1)
        feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - num_vacancies),5))[0]

        total_feasible = np.sum(multiplicity[feasible_config])
        ratio_feasible = total_feasible/total_reads

        return np.round(ratio_feasible,4)'''


def find_ratio_feasible_discrete(dataframe, species, concentration, remove_broken_chains = False, total_reads = None):
    
    # remove_broken_chains = True : ratio wrt the total number of non broken chains solutions
    # remove_broken_chains = False : ratio wrt the total number of reads
    
    if total_reads == None:
        
        total_reads = np.sum(dataframe['num_occurrences'].to_numpy())
        dataframe = convert_df_binary2atom_discrete(dataframe, species, remove_unfeasible=True)
        df = remove_unfeasible_solutions(dataframe, species, concentration)
        reads_cleaned = np.sum(df['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads
    else:
        reads_cleaned = np.sum(dataframe['num_occurrences'].to_numpy())
        
        return reads_cleaned/total_reads

    
def find_ratio_ground_state(dataframe,num_vacancies, remove_broken_chains = False):
    
    # remove_broken_chains = True : ratio wrt the total number of non broken chains solutions
    # remove_broken_chains = False : ratio wrt the total number of reads
    
    if remove_broken_chains == True:
        total_reads = np.sum(dataframe[dataframe['chain_break_fraction'] == 0.]['num_occurrences'])
        df = dataframe[dataframe['chain_break_fraction'] == 0]
    elif remove_broken_chains == False:
        total_reads = np.sum(dataframe['num_occurrences'])
        df = dataframe
    
    if len(df) == 0:
        return 0.
    
    else:
        
        num_atoms = sum([x.isdigit() for x in df.columns])
        
        all_config = df.iloc[:,0:num_atoms].to_numpy()

        if num_vacancies == 0:
            gs_energy = -((3*num_atoms)/2)
        elif num_vacancies > 0:
            gs_energy = -((3*num_atoms)/2)+(3 + (num_vacancies-1)*2)
        all_config = df.iloc[:,0:num_atoms].to_numpy()

        multiplicity = df['num_occurrences'].to_numpy()

        energies = df['energy'].to_numpy()

        sum_vector = np.sum(all_config,axis=1)

        feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - num_vacancies),5))[0]

        ground_states = np.where(np.round(energies,5) == np.round(gs_energy,5))

        feasible_gs = np.intersect1d(ground_states,feasible_config)

        total_gs = np.sum(multiplicity[feasible_gs])

        ratio_gs = total_gs/total_reads

        return np.round(ratio_gs,4)


def find_symmetry_equivalent_structures(dataframe, structure, remove_unfeasible = False, species=None ,concentration=None,):
    #spglib-based analysis

    
    #Concentration follows the order given in species
    
    import copy 
    from pymatgen.analysis.structure_matcher import StructureMatcher 

    df = dataframe
    
    num_sites = structure.num_sites
    lattice = structure.lattice
    atom_position = structure.cart_coords
    
    '''if concentration is not None and species is not None:
        feasible_config = []
        all_config = df.iloc[:,0:num_sites].to_numpy()
        #sum_vector = np.sum(all_config,axis=1)
        #feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - vacancies),5))[0]
        for config in all_config:
            feasible = True
            for i in range(len(concentration)):
                feasible *= np.sum(config == species[i]) == concentration[i] 
            #print(feasible)
            feasible_config.append(feasible) 
        
        df = df.iloc[feasible_config,:]'''
    
    if remove_unfeasible == True and species is not None and concentration is not None:
        df = remove_unfeasible_solutions(dataframe,species,concentration)
    
    #configurations = df.iloc[:,0:num_sites].to_numpy()
    
    multiplicity = df['num_occurrences'].to_numpy()
    chain_break = df['chain_break_fraction'].to_numpy()
    energies = df['energy'].to_numpy()


    '''#Replace the C atom with an H atom where the vacancies are
    zero_elements = np.where(configurations == 0) 
    configurations[zero_elements] = 99'''
    
    all_structures = df2structure(df,structure)
    '''for config in configurations:
        all_structures.append(Structure(lattice, config, atom_position, coords_are_cartesian=True))'''

    
    '''#Build the descriptor - WIP
    descriptor = build_descriptor(all_structures)

    descriptor_unique, descriptor_first, descriptor_count = \
    np.unique(descriptor, axis=0,return_counts=True, return_index=True)

    group_structures = []
    for desc in descriptor_unique:
        structure_desc = []
        for i,d in enumerate(descriptor):
            if np.all(np.array(desc) == np.array(d)):
                structure_desc.append(i)
        group_structures.append(structure_desc)'''
    '''for structure in all_structures:
        SpacegroupAnalyzer()
    
    unique_multiplicity = []
    unique_chain_break = []
    unique_structure_index = []
    
    for x in group_structures:
        unique_structure_index.append(x[0])
        unique_multiplicity.append(np.sum(multiplicity[x]))
        unique_chain_break.append(np.average(chain_break[x],weights=multiplicity[x]))    
    
    df = df.iloc[unique_structure_index].copy()
    
    if len(df) == len(unique_multiplicity):
        df['num_occurrences'] = unique_multiplicity
        df['chain_break_fraction'] = unique_chain_break
        
        return df
    
    else:
        print('Some structures might be unfeasible, try using a smaller energy range (lower energy)')
        
        return None'''
    
    #Find the unique structures
    unique_structures = StructureMatcher().group_structures(all_structures)
    
    unique_structures_label = []
    
    #Find to which class the structures belong to
    for structure in all_structures:
        for i in range(len(unique_structures)):
            #print(unique_structures[i][0].composition.reduced_formula,structure.composition.reduced_formula)
            if StructureMatcher().fit(structure,unique_structures[i][0]) == True:
                unique_structures_label.append(i)
                break
    
    unique_structures_label = np.array(unique_structures_label)
    unique_multiplicity = []
    unique_chain_break = []
    for x in range(len(unique_structures)):
        multiplicity_tmp = multiplicity[np.where(unique_structures_label==x)[0]]
        unique_multiplicity.append(np.sum(multiplicity_tmp))
        unique_chain_break.append(np.average(chain_break[np.where(unique_structures_label==x)[0]],weights=multiplicity_tmp))
    
    df = df.iloc[np.unique(unique_structures_label,return_index=True)[1]]
    
    if len(df) == len(unique_multiplicity):
        df1 = df.copy(deep=True)
        df1['num_occurrences'] = unique_multiplicity
        df1['chain_break_fraction'] = unique_chain_break
        
        return df1
    
    else:
        print('Some structures might be unfeasible, try using a smaller energy range (lower energy)')
        
        return None


def find_unique_E_structures(dataframe, min_energy = 0, return_count = False):


    #PROBABLY OLD VERSION OFFIND EQUIVALENT ENERGY DISTRIBUTION

    # energy = list of all the energies returned by the anneal
    # configurations = list of all configurations corresponding to a certain energy
    
    import numpy as np

    n_atoms = len([i for i in dataframe.columns if type(i) == int])

    config = dataframe.iloc[:,0:n_atoms].to_numpy()

    energies = dataframe['energy'].to_numpy()

    below_min_energy = np.where(np.unique(energies,return_index=True)[0] < min_energy)[0]

    return np.unique(energies,return_index=True)[0][below_min_energy], np.unique(energies,return_index=True)[1][below_min_energy]


def lowest_energy_found(dataframe, bqm, limit = 25):
    
    from quantum_computing_functions import find_exact_solutions

    num_atoms = sum([type(x) == int for x in dataframe.columns])
    
    if num_atoms > limit:
        return None
    
    dataframe_low_e = np.min(dataframe['energy'].to_numpy())
    exact_low_e = np.min(find_exact_solutions(bqm)['energy'].to_numpy())
    
    if dataframe_low_e == exact_low_e:
        return True
    else:
        return False


def make_df(directory):

    import os

    dataframes = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f) and f.endswith(".json") and 'bqm' in f:
            #print(f)
            df, param = load_json(f,return_param = True, return_qubo = False)        
            param = param.transpose()
            n_vac = param['num_vacancies'].values[0]
            n_atoms = param['N atoms'].values[0]
            param['broken_chains'] = find_ratio_broken_chains(df)   
            param['ratio_feasible'] = find_ratio_feasible(df,param['num_vacancies'].values[0], remove_broken_chains=False)
            param['ratio_gs'] = find_ratio_ground_state(df,param['num_vacancies'].values[0], remove_broken_chains=False)
            param['energy_distribution_all'] = ', '.join(str(x) for x in find_energy_distribution(df)[0])
            param['multiplicity_all'] = ', '.join(str(x) for x in find_energy_distribution(df)[1])        
            param['energy_distribution'] = ', '.join(str(x) for x in find_energy_distribution(df, only_feasible=True, vacancies=param['num_vacancies'].values[0])[0])
            #param['num_broken_bonds'] = ', '.join(str(x) for x in find_num_broken_bonds(dataframe, remove_broken_chains = True, only_feasible = True, vacancies = param['num_vacancies'].values[0]))
            param['multiplicity'] = ', '.join(str(x) for x in find_energy_distribution(df, only_feasible=True, vacancies=param['num_vacancies'].values[0])[1])
            param['ratio l/w'] =  param['lagrange'] / param['weight_1'] 
            param['vacancies'] = ', '.join(str(x) for x in find_vacancy_distribution(df)[0])
            param['vacancies mult'] = ', '.join(str(x) for x in find_vacancy_distribution(df)[1])
            
            if 'chain_strength' not in param:
                param['chain_strength'] = -1
            #print(param['chain_strength'])
            if param['chain_strength'].all() == None:
                param['chain_strength'] = -1
            dataframes.append(param)

    df_results = pd.concat(dataframes)
    
    return df_results


def make_df_paper(dataframe,num_vacancies,symmetrised=False,structure=None,num_runs=1,only_feasible = False):
    
    import copy
    import string
    
    total_runs = np.sum(dataframe['num_occurrences'].to_numpy())
    
    df = dataframe

    num_atoms = sum([x.isdigit() for x in df.columns]) 
    all_config = df.iloc[:,0:num_atoms].to_numpy()
    sum_vector = np.sum(all_config,axis=1)
    feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - num_vacancies),5))[0]
    df = df.iloc[feasible_config,:]
        
    feasible_runs = np.sum(df['num_occurrences'].to_numpy())
    

    multiplicity = df['num_occurrences'].to_numpy()
    
    config = df.iloc[:,0:num_atoms].to_numpy()
    unique_config = np.unique(config,axis=0)
    
    chain_break = df['chain_break_fraction'].to_numpy()
    

    chain_break_average = []
    energy_average = []
    total_occurrences = []

    for c in unique_config:
        
        feasible = np.where((config == c).all(axis=1))[0]
        broken = df.iloc[feasible,18].to_numpy()
        energy = df.iloc[feasible,(num_atoms+1)].to_numpy()
        weights = df.iloc[feasible,(num_atoms+2)].to_numpy()
        
        chain_break_average.append(np.average(broken,weights=weights))
        energy_average.append(np.average(energy,weights=weights))
        total_occurrences.append(np.sum(weights)) 
        

    df = pd.DataFrame(unique_config, columns=[str(x) for x in np.arange(num_atoms,dtype=int)])
    df['chain_break_fraction'] = np.round(chain_break_average,4)
    df['energy'] = energy_average
    df['num_occurrences'] = total_occurrences
    
     
    if symmetrised == True and structure != None:
        
        structure_tmp = copy.deepcopy(structure)
        df = find_symmetry_equivalent_structures(df,structure_tmp)
    
    df.sort_values(by=['energy'], inplace=True)
    df['chain_break_fraction'] = np.round(df['chain_break_fraction'].to_numpy(),5)
    df['chain_break_fraction'] = df['chain_break_fraction'].apply(lambda x: x*100)
    if only_feasible == True: 
        df['num_occurrences'] = df['num_occurrences'].apply(lambda x: (x/feasible_runs)*100)
    elif only_feasible == False:
        df['num_occurrences'] = df['num_occurrences'].apply(lambda x: (x/total_runs)*100)
    
    df = df.rename(columns={"chain_break_fraction": "% broken chains", "energy": "Energy", "num_occurrences":"% occurrence"})

    if len(df.index) < 27:
        df.index = list(string.ascii_lowercase[0:len(df.index)])
    
    return df


def convert_discrete_df(dataframe,num_species,name_species,remove_unfeasible=True):

    # Convert a discrete df 
    
    import copy

    df = copy.deepcopy(dataframe)

    num_sites = sum([type(x) == int for x in df.columns])
    sites = df.iloc[:,0:num_sites].to_numpy()
    
    sum_vector = sites[:,0::num_species]
    for i in range(1,num_species):
        sum_vector += sites[:,i::num_species]
    unfeasible = np.where(np.prod(sum_vector,axis=1) != 1)[0]

    #unfeasible = np.where(np.prod((np.sum([sites[:,x*num_species:(x+1)*num_species] 
                                           #for x in range(num_species)],axis=0)),axis=1) != 1)[0]
    

    '''if ts == True:
        print('test this')
        unfeasible = np.concatenate((unfeasible,np.where(np.prod((np.sum([sites[:,x::num_species] 
                                       for x in range(num_species)],axis=0)),axis=1) != 1)[0]))
        unfeasible = np.unique(unfeasible)'''
                                       
    if remove_unfeasible == True:
        df.drop(unfeasible, inplace=True)

    if df.empty:
        print('The dataframe is empty, increase the theta value')
        return None
    

    new_labels_arr = df.iloc[:,0:num_sites].to_numpy()

    new_labels = []
    df.drop(np.arange(num_sites),axis=1,inplace=True)
    for line in new_labels_arr:
        new_labels_tmp=  [name_species[x] for x in np.where(line == 1)[0]%num_species]

        new_labels.append(new_labels_tmp)
    new_labels = np.array(new_labels)
    for i in range(new_labels.shape[1]):
        df.insert(i, i, new_labels[:,i])

    
    return df


'''def convert_df(df,remove_unfeasible=True):
    # Convert a discrete df 

    num_sites = sum([type(x) == int for x in df.columns])
    sites = df.iloc[:,0:num_sites].to_numpy()
    unfeasible = np.where(np.prod((sites[:,::2]+sites[:,1::2]),axis=1) != 1)[0]
    df.drop(unfeasible, inplace=True)
    
    new_labels = df.iloc[:,0:num_sites].to_numpy()[:,::2]
    df.drop(np.arange(num_sites),axis=1,inplace=True)

    
    for i in range(new_labels.shape[1]):
        df.insert(i, i, new_labels[:,i])
    
    return df'''


def remove_unfeasible_solutions(dataframe,species,concentration):

    df = copy.deepcopy(dataframe)


    if isinstance(df.columns[0], int):
        num_sites = sum([type(x) == int for x in df.columns])
    elif isinstance(df.columns[0], str):
        num_sites = sum([x.isdigit() for x in df.columns])

    feasible_config = []
    all_config = df.iloc[:,0:num_sites].to_numpy()
    #sum_vector = np.sum(all_config,axis=1)
    #feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - vacancies),5))[0]
    for config in all_config:
        feasible = True
        for i in range(len(concentration)):
            feasible *= np.sum(config == species[i]) == concentration[i] 
        #print(feasible)
        feasible_config.append(feasible) 
    
    df = df.iloc[feasible_config,:]

    return df


def time_to_solution(dataframe,num_vacancies, anneal_time, remove_broken_chains = False, discrete = False):

    import numpy as np

    if discrete == False:
        ratio = find_ratio_feasible(dataframe,num_vacancies, remove_broken_chains = remove_broken_chains)
    elif discrete == True:
        ratio = find_ratio_feasible_discrete(dataframe,num_vacancies, remove_broken_chains = remove_broken_chains)

    tts = (np.log10(1-0.99) / np.log10(1-ratio) ) * anneal_time

    return tts



