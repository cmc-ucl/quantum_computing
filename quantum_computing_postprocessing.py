# Bruno Camino 19/07/2022

from tkinter.messagebox import QUESTION
import numpy as np

def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    if np.around(az,6) ==  np.around(2*np.pi,6) \
    or np.around(az,6) ==  -np.around(2*np.pi,6):
        az = 0.
    if np.around(az,6) < 0.:
        az = np.round(2*np.pi+az,6)
    return [round(az,6), round(el,6), round(r,6)]


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


def find_ratio_acceptable(dataframe, remove_broken_chains = False):
    
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


def find_ratio_feasible(dataframe,num_vacancies, remove_broken_chains = False):
    
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


def find_ratio_feasible_discrete(dataframe,num_vacancies, remove_broken_chains = False):
    
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

        acceptable_config = np.all((all_config[:,::2]+all_config[:,1::2])-np.ones(18)==0,axis=1)
        
        sum_vector = np.sum(all_config[:,::2],axis=1)

        feasible_configurations = np.round(sum_vector,5) == np.round((num_atoms/2 - num_vacancies),5)

        feasible_config = np.where((feasible_configurations * acceptable_config) == True)[0]

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


def convert_df(df,remove_unfeasible=True):
    
    num_sites = sum([type(x) == int for x in df.columns])
    sites = df.iloc[:,0:num_sites].to_numpy()
    unfeasible = np.where(np.prod((sites[:,::2]+sites[:,1::2]),axis=1) != 1)[0]
    df.drop(unfeasible, inplace=True)
    
    new_labels = df.iloc[:,0:num_sites].to_numpy()[:,::2]
    df.drop(np.arange(num_sites),axis=1,inplace=True)

    
    for i in range(new_labels.shape[1]):
        df.insert(i, i, new_labels[:,i])
    
    return df


def time_to_solution(dataframe,num_vacancies, anneal_time, remove_broken_chains = False, discrete = False):

    import numpy as np

    if discrete == False:
        ratio = find_ratio_feasible(dataframe,num_vacancies, remove_broken_chains = remove_broken_chains)
    elif discrete == True:
        ratio = find_ratio_feasible_discrete(dataframe,num_vacancies, remove_broken_chains = remove_broken_chains)

    tts = (np.log10(1-0.99) / np.log10(1-ratio) ) * anneal_time

    return tts
