# Bruno Camino 19/07/2022

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


def find_unique_E_structures(dataframe, min_energy = 0, return_count = False):
    # energy = list of all the energies returned by the anneal
    # configurations = list of all configurations corresponding to a certain energy
    
    import numpy as np

    n_atoms = len([i for i in dataframe.columns if type(i) == int])

    config = dataframe.iloc[:,0:n_atoms].to_numpy()

    energies = dataframe['energy'].to_numpy()

    below_min_energy = np.where(np.unique(energies,return_index=True)[0] < min_energy)[0]

    return np.unique(energies,return_index=True)[0][below_min_energy], np.unique(energies,return_index=True)[1][below_min_energy]