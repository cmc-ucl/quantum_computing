# Bruno Camino 19/07/2022

from tkinter.messagebox import QUESTION
import numpy as np

def display_low_E_structures(structure,energies,configurations, min_energy = 0, view = False):
    
    from pymatgen.io.ase import AseAtomsAdaptor
    from ase.visualize import view
    import numpy as np

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


def find_ratio_broken_chains(dataframe):
    # Return how many solutions returned a broken chain
    
    broken = np.sum(dataframe[dataframe['chain_break_fraction'] != 0.]['num_occurrences'].to_numpy())
    total = np.sum(dataframe['num_occurrences'].to_numpy())
    
    return np.round(broken/total,4)


def find_ratio_feasible(dataframe,num_vacancies, remove_broken_chains = False):
    
    # only the non broken chains solutions will be considered, regardless of remove_broken_chains
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
        
        sum_vector = np.sum(all_config,axis=1)

        feasible_config = np.where(np.round(sum_vector,5) == np.round((num_atoms - num_vacancies),5))[0]

        total_feasible = np.sum(multiplicity[feasible_config])
        ratio_feasible = total_feasible/total_reads

        return np.round(ratio_feasible,4)


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

