# Bruno Camino 19/07/2022

import numpy as np

def build_adjacency_matrix(structure):
    # structure = pymatgen Structure object
    
    import numpy as np
    
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    shells = np.unique(distance_matrix_pbc[0])

    adjacency_matrix = np.round(distance_matrix_pbc,5) == np.round(shells[1],5)
    adjacency_matrix = adjacency_matrix.astype(int)
    
    return adjacency_matrix


def build_constrained_quadratic_model(structure,use_coord = True, num_vacancies = 0, 
                          weight_1=10, weight_2 = 1, lagrange = 1000):
    # structure = pymatgen Structure object
    # weight_1 = weight for the bond energy objective
    # weight_1 = weight for the bond energy objective
    # lagrange = weight for the number of vacancies constraint
    
    from dimod import BinaryQuadraticModel, ConstrainedQuadraticModel, Binary, quicksum
    from dwave.system import LeapHybridCQMSampler
    
    if num_vacancies == 0:
        print('No constraints to apply - Unconstrained quadratic model used instead')
        
        return build_quadratic_model(structure,use_coord = use_coord, num_vacancies = 0, 
                          weight_1=weight_1, weight_2 = weight_2, lagrange = lagrange)
    elif num_vacancies < 0:
        print('Please select a positive integer number of vacancies')
     
        return None
    
    elif num_vacancies > 0:
        print('Constrained quadratic model')
    
        atoms = [Binary(i) for i in range(structure.num_sites)]

        atoms = np.array(atoms)

        cqm = ConstrainedQuadraticModel()

        c = np.arange(structure.num_sites)

        adjacency_matrix = build_adjacency_matrix(structure)

        Q = np.triu(-weight_1*adjacency_matrix.astype(int),0)

        bqm = BinaryQuadraticModel.from_qubo(Q)

        bqm = BinaryQuadraticModel.from_qubo(Q)

        if use_coord == True:
            for i in range(structure.num_sites):  
                neighbours = np.where(adjacency_matrix[i,:] == 1)[0]
                for j in range(len(neighbours)):
                    for k in range(len(neighbours)):
                        if k > j:
                            bqm.add_interaction(c[neighbours[j]],c[neighbours[k]],weight_2)

        cqm.set_objective(bqm)

        #set the constraint: number of vacancies
        cqm.add_constraint(quicksum(atoms[i] for i in range(structure.num_sites)) \
                           == (structure.num_sites - num_vacancies), label='Num vacancies')

        return cqm
    
    else:
        print('Something is wrong with the number of vacancies')
        
        return None


def build_discrete_quadratic_model_ip(structure,species,concentrations, parameters, max_neigh = 1,):
    # structure = pymatgen Structure object
    # species = list of atomic numbers
    # interaction = interactomic potential given as a matrix element by row
    # concentrations = list of concentrations
    
    # https://github.com/dwave-examples/graph-partitioning
    
    from dimod import DiscreteQuadraticModel
    
    c = np.arange(structure.num_sites)
    
    adjacency_matrix = build_adjacency_matrix(structure)
    
    dqm = DiscreteQuadraticModel()
     
    n_species = len(species)
    concentration_1 = concentrations[0]
    
    n_atoms_species = [int(np.rint(x*structure.num_sites)) for x in concentrations[:-1]]
    n_atoms_species.append(int(np.rint(structure.num_sites-np.sum(c))))


    
    #Make this better (or input?)
    interaction_dict = {}
    index = -1
    for i in range(len(species)):
        for j in range(i,len(species)):
            index += 1
            interaction_dict[(i, j)] = interaction[index]
    
    #This creates a (num_sites*num_species) x (num_sites*num_species) matrix
    #DOES IT INCLUDE THE ONE-HOT ENCODING?
    
    for i in range(structure.num_sites):
        dqm.add_variable(n_species, label=i)
    
    #define linked atoms and bond strength
    for i in range(structure.num_sites):
        for j in np.where(adjacency_matrix[i] == 1)[0]:
            #dqm.set_quadratic(i, j, {(0, 0): cu_cu, (0, 1): cu_zn, (1, 1): zn_zn})
            dqm.set_quadratic(i, j, interaction_dict)
    
    # This imposes a double constraint (on both species)
    for i,conc in enumerate(n_atoms_species):
        concentration_1_v = [(s,i,1) for s in c] #make loop
        print(concentration_1_v)
    
        #Define the concentration as a constraint
        dqm.add_linear_equality_constraint(
                    concentration_1_v,
                    constant= -conc,
                    )
    
    '''OLD ONE
    concentration_1_v = [(i,0,1) for i in X]
    
    #Define the concentration as a constraint
    dqm.add_linear_equality_constraint(
                concentration_1_v,
                constant= -concentration_1,
                lagrange_multiplier = lagrange
                )'''
    
    
    return dqm


def build_graph(structure):
    # Generate the graph from the pymatgen structure
    
    from pymatgen.analysis.graphs import StructureGraph
    from pymatgen.analysis.local_env import NearNeighbors, MinimumDistanceNN
    from networkx import Graph
    
    G = StructureGraph.with_local_env_strategy(structure,MinimumDistanceNN())
    
    return Graph(G.graph)


def build_quadratic_model(structure,use_coord = True, num_vacancies = 0, 
                          weight_1=10, weight_2 = 1, lagrange = 1000):
    # structure = pymatgen Structure object
    # weight_1 = weight for the bond energy objective
    # weight_1 = weight for the bond energy objective
    # lagrange = weight for the number of vacancies constraint
    
    from dimod import BinaryQuadraticModel
    
    X = np.arange(structure.num_sites)
    
    adjacency_matrix = build_adjacency_matrix(structure)

    Q = np.triu(-weight_1*adjacency_matrix.astype(int),0)
    
    bqm = BinaryQuadraticModel.from_qubo(Q)
    
    if use_coord == True:
        for i in range(structure.num_sites):  
            neighbours = np.where(adjacency_matrix[i,:] == 1)[0]
            for j in range(len(neighbours)):
                for k in range(len(neighbours)):
                    if k > j:
                        bqm.add_interaction(X[neighbours[j]],X[neighbours[k]],weight_2)
    

    if num_vacancies == 0:
        print('Unconstrained quadratic model used')
        
        return bqm
    elif num_vacancies > 0:
        print('Unconstrained quadratic model + contraints used')
        
        c_n_vacancies = [(i,1) for i in X]

        bqm.add_linear_equality_constraint(
                c_n_vacancies,
                constant= -(structure.num_sites-num_vacancies),
                lagrange_multiplier = lagrange
                )

        return bqm
    
    else:
        print('Please select a positive integer number of vacancies')
     
        return None


def build_quadratic_model_discrete(structure,species,concentrations, parameters, weight=100, max_neigh = 1):
    '''# structure = pymatgen Structure object
    # weight_1 = weight for the bond energy objective
    # weight_1 = weight for the bond energy objective
    # lagrange = weight for the number of vacancies constraint'''
    
    
    from dimod import BinaryQuadraticModel, Binary
    
    num_sites = structure.num_sites
    num_species = len(species)
    
    X = np.arange(structure.num_sites)
    
    adjacency_matrix = build_adjacency_matrix(structure)

    Q = np.triu(adjacency_matrix.astype(int),0)
    
    bqm = BinaryQuadraticModel.empty(vartype='BINARY')

    
    ################## Add one-hot encoding ###########################

    # -1 diagonal
    # 2 off diagonal
    J = np.array([[0]*num_sites*num_species]*(num_sites*num_species))
    for i in range(num_sites*num_species):
        for j in range(i,num_sites*num_species,num_species):            
            J[i,j] = -1
            for k in range(1,num_species-i%num_species):
            #for k in range(1,num_species):
                if j+k < num_sites*num_species:
                    #print(i,j+k)
                    J[i,j+k] = +2
    J = J * weight
    
    
    ################## Calculate the potentials ###############################
    
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    
    # Generate an all False matrix
    distance_matrix_filter = (distance_matrix == -1)

    
    # Only add the atoms within the shells up to max_neigh 
    for neigh in range(1,max_neigh+1):
        distance_matrix_filter +=  distance_matrix == shells[neigh]  

    # Buckingham
    #loop 
    
    ip_matrix = np.array([[0.]*num_sites*num_species]*num_sites*num_species)
    parameters = np.array(parameters)
    for i in range(num_sites):
        for j in range(i,num_sites):
            if distance_matrix_filter[i,j] == True:
                index = -1
                for k in range(num_species):
                    for l in range(k,num_species):
                        index += 1
                        #print(i,j)
                        param = parameters[index]
                        #print(i*num_species+k,j*num_species+l)
                        ip_matrix[i*num_species+k,j*num_species+l] = param[0] * np.exp((-distance_matrix[i,j])/(param[1]))- ((param[2])/((distance_matrix[i,j])**6))
        
    
    
    ################## Add concentration constraint ###############################
    n_species = len(species)
    
    n_atoms_species = np.multiply(concentrations,num_sites)
    print(n_atoms_species)
    C = np.array([[0.]*num_sites*num_species]*(num_sites*num_species))
    for i in range(num_sites*num_species):
        for j in range(i,num_sites*num_species,num_species):
            #print(i,j, j%num_species)
            C[i,j] = 1-(n_atoms_species[j%num_species])
            for k in range(1,num_species-i%num_species):
            #for k in range(1,num_species):
                if j+k < num_sites*num_species:
                    #print(i,j+k)
                    C[i,j+k] = +2
    
    #print(J+C+ip_matrix)
    #return J
    return bqm


def build_qubo_matrix(bqm, transpose= True):
    
    # Returns the qubo matrix from a bqm model (cqm notr supported)
    
    # Build a n_atoms x n_atoms matrix containing only 0s    
    num_items = len(bqm.to_numpy_vectors().linear_biases)   
    qubo_matrix = np.array([[0]*num_items]*num_items)
    
    # Add the h_ii elements (diagonal)
    np.fill_diagonal(qubo_matrix,bqm.to_numpy_vectors().linear_biases)
    
    # Add the J_ij (off diagonal elements)
    num_non_zero_elements = len(bqm.to_numpy_vectors().quadratic.row_indices)

    row_indices = bqm.to_numpy_vectors().quadratic.row_indices
    col_indices = bqm.to_numpy_vectors().quadratic.col_indices

    biases = bqm.to_numpy_vectors().quadratic.biases

    #Assign the quadratic biases to the i,j elements
    for i in range(num_non_zero_elements):
        qubo_matrix[row_indices[i]][col_indices[i]] = biases[i]
        
    #Transpose the matrix (the model builds the lower left side of the matrix, 
    #it is hermitian, so it doesn't matter)  
    if transpose == True:
        qubo_matrix = qubo_matrix.transpose()
    
    return qubo_matrix


def find_exact_solutions(bqm):
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        
        from dimod import ExactSolver
        
        return ExactSolver().sample(bqm).to_pandas_dataframe()
    
    elif 'ConstrainedQuadraticModel' in str(type(bqm)):
        
        from dimod import ExactCQMSolver
        
        return ExactCQMSolver().sample_cqm(bqm).to_pandas_dataframe()


def load_json(file_name, return_param = True, return_qubo = True):
    
    import numpy as np
    import pandas as pd
    import json
    
    # Read a previously saved dataframe
    
    with open(file_name) as f:
        data = json.load(f)
    
    param = data.pop('parameters')
    qubo_matrix = np.array(param.pop('qubo_matrix'))
    param_df = pd.DataFrame(param,index=['Values']).transpose()
    
    dataframe = pd.DataFrame.from_dict(data)
    
    if return_param == True and return_qubo == True:
        return dataframe, param_df, qubo_matrix
    elif return_param == True and return_qubo == False:
        return dataframe, param_df
    elif return_param == False and return_qubo == False:
        return dataframe


def plot_buckingham_potential(A, p, C, r_range=[0,5],coulomb= []):
    
    import matplotlib.pyplot as plt
    import numpy as np

    color = ['b','g','r','c','m','y','k','tab:orange']
    
    fig, ax = plt.subplots()
    
    x = np.arange(r_range[0],r_range[1],0.1)
    
    c=0
    if len(coulomb) > 0:
        c = 1 #Change this
    y = A * np.exp(-x/p) - (C / (x**6))  + c 
    
    ax.set_xlim([0, 4])
    ax.set_ylim([-1, 300])
    ax.plot(x,y,'-')
    

def run_anneal(bqm,num_reads = 1000, time_limit=5, chain_strength = None, label='Test anneal', dataframe = False, 


               remove_broken_chains = True, return_config_E = False):
    
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        from dwave.system import EmbeddingComposite, DWaveSampler

        sampler = EmbeddingComposite(DWaveSampler())
        
        if chain_strength != None:
            result = sampler.sample(bqm, num_reads = num_reads, chain_strength = chain_strength,  label=label)
        else:
            result = sampler.sample(bqm, num_reads = num_reads, label=label)
        
    elif 'ConstrainedQuadraticModel' in str(type(bqm)):
        from dwave.system import LeapHybridCQMSampler
        
        cqm_sampler = LeapHybridCQMSampler()
        
        result = cqm_sampler.sample_cqm(bqm,time_limit=time_limit)
    
    elif 'DiscreteQuadraticModel' in str(type(bqm)):
        
        from dwave.system import LeapHybridDQMSampler

        dqm_sampler = LeapHybridDQMSampler()
        
        result = dqm_sampler.sample_dqm(test,time_limit=time_limit)
      
    
    if dataframe == True:
        result_df = result.to_pandas_dataframe()
        
        if 'BinaryQuadraticModel' in str(type(bqm)):
            if remove_broken_chains == True and return_config_E == True:
                result_df = result_df[result_df['chain_break_fraction'] == 0.]
            
            if return_config_E == True:
                n_atoms = len(bqm.to_numpy_vectors()[0])

                config = result_df.iloc[:,0:n_atoms].to_numpy()
                energies = result_df['energy'].to_numpy()

                return result_df, config, energies
            
        else:
            return result_df
        
    else:
        return result


def save_json(structure,sampleset, bqm, use_coord = True, num_vacancies = 0, 
                          weight_1=10, weight_2 = 1, lagrange = 1000,
              num_reads = 1000, time_limit=5, label='Test anneal', 
              remove_broken_chains = True, file_path = 'data', file_name = '', save_qubo = True,
              chain_strength = None):
    
    # save the dataframe as a json file
    
    import json
    from datetime import datetime,timezone
    from os import path
    import time
    
    dataframe = sampleset.to_pandas_dataframe()
    
    if 'chain_strength' in sampleset.info['embedding_context']:
        chain_strength = sampleset.info['embedding_context']['chain_strength']
    else:
        chain_strength = -1
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        model = 'bqm'
        time_limit = 0
    elif 'ConstrainedQuadraticModel' in str(type(bqm)):
        model = 'cqm'
        num_reads = 0
    
    date_time = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    time_stamp = int(time.time())
    
    if save_qubo == True:
        qubo_matrix = build_qubo_matrix(bqm).flatten().tolist()
    elif save_qubo == False:
        qubo_matrix = None
    
        
    param_dict = {'date_time': date_time,
                  'time_stamp': time_stamp,
                  'structure': structure.composition.formula,
                  'N atoms' : structure.num_sites,
                    'model': model,
                  'use_coord' : use_coord,
                  'num_vacancies': num_vacancies,
                  'weight_1': weight_1,
                  'weight_2' : weight_2,
                  'lagrange': lagrange,
                     'num_reads' : num_reads, 
                  'time_limit': time_limit,
                  'label':label, 
                  'remove_broken_chains' : remove_broken_chains,
                  'chain_strength' : chain_strength,
                  #'qubo_matrix': qubo_matrix,
                  'qpu_anneal_time_per_sample': sampleset.info['timing']['qpu_anneal_time_per_sample'],
                  'qubo_matrix': qubo_matrix
                     
    }
    
    json_string = dataframe.to_json()    
    json_object = json.loads(json_string)
    json_object['parameters'] = param_dict


    name = file_name + '_%s_%s_v%s_c%s_w1%s_w2%s_l%s_r%s_t%s_%s.json'%(structure.composition.formula,
                                                                    model,num_vacancies,str(use_coord)[0],
                                                                    weight_1,weight_2,lagrange,num_reads,                                                                    
                                                                    time_limit,time_stamp)
    
    file_name = path.join(file_path,name)

    with open(file_name, 'w') as f:
        json.dump(json_object, f)


def save_json_old(structure,dataframe, bqm, use_coord = True, num_vacancies = 0, 
                          weight_1=10, weight_2 = 1, lagrange = 1000,
              num_reads = 1000, time_limit=5, label='Test anneal', 
              remove_broken_chains = True, file_path = 'data', file_name = '', save_qubo = True,
             low_exact = None):
    
    # save the dataframe as a json file
    
    import json
    from datetime import datetime,timezone
    from os import path
    import time
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        model = 'bqm'
        time_limit = 0
    elif 'ConstrainedQuadraticModel' in str(type(bqm)):
        model = 'cqm'
        num_reads = 0
    
    date_time = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    time_stamp = int(time.time())

    
    if save_qubo == True:
        qubo_matrix = build_qubo_matrix(bqm).flatten().tolist()
    elif save_qubo == False:
        qubo_matrix = None
    
    if chain_strength == None:
        chain_strength = -1
        
    param_dict = {'date_time': date_time,
                  'time_stamp': time_stamp,
                  'structure': structure.composition.formula,
                  'N atoms' : structure.num_sites,
                    'model': model,
                  'use_coord' : use_coord,
                  'num_vacancies': num_vacancies,
                  'weight_1': weight_1,
                  'weight_2' : weight_2,
                  'lagrange': lagrange,
                     'num_reads' : num_reads, 
                  'time_limit': time_limit,
                  'label':label, 
                  'remove_broken_chains' : remove_broken_chains,
                  'chain_strength' : chain_strength,
                  'low_exact': low_exact,
                  #'qubo_matrix': qubo_matrix,
                  'qpu_anneal_time_per_sample': sampleset.info['timing']['qpu_anneal_time_per_sample'],
                  
                  'qubo_matrix': qubo_matrix
                     
    }
    
    json_string = dataframe.to_json()
    
    json_object = json.loads(json_string)
    json_object['parameters'] = param_dict

    name = file_name + '_%s_%s_v%s_c%s_w1%s_w2%s_l%s_r%s_t%s_%s.json'%(structure.composition.formula,
                                                                    model,num_vacancies,str(use_coord)[0],
                                                                    weight_1,weight_2,lagrange,num_reads,
                                                                    
                                                                    time_limit,time_stamp)
    
    file_name = path.join(file_path,name)

    with open(file_name, 'w') as f:
        json.dump(json_object, f)        