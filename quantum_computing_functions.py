# Bruno Camino 19/07/2022

import numpy as np
import copy
import time


def build_adjacency_matrix(structure):
    # structure = pymatgen Structure object
    
    import numpy as np
    
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    shells = np.unique(distance_matrix_pbc[0])

    adjacency_matrix = np.round(distance_matrix_pbc,5) == np.round(shells[1],5)
    adjacency_matrix = adjacency_matrix.astype(int)
    
    return adjacency_matrix


def build_adj_adjacency_matrix(structure):
    
    import numpy as np
  
    ''' Old version
    A = build_adjacency_matrix(structure)
    B = np.zeros((structure.num_sites,structure.num_sites))
    
    for i in range(structure.num_sites):  
        neighbours = np.where(A[i,:] == 1)[0]
        for j in range(len(neighbours)):
            for k in range(len(neighbours)):
                if k > j:
                    B[neighbours[j],neighbours[k]] = 1'''
    distance_matrix_pbc = np.round(structure.distance_matrix,5)

    shells = np.unique(distance_matrix_pbc[0])

    B = np.round(distance_matrix_pbc,5) == np.round(shells[1],5)
    B = np.triu(B)
    B = B.astype(int)
    
    return B


def build_constrained_quadratic_model(structure,use_coord = False, num_vacancies = 0, 
                          weight_1=10, weight_2 = 0, lagrange = 1000):
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


def build_descriptor(structures,max_shell=3):
    shells = np.unique(np.round(structures[0].distance_matrix[0],decimals=6),return_counts=True)[0].tolist()
    neighbours_spatial_dist = []
    neighbours_spatial_dist_all = []
    
    for k,structure in enumerate(structures): 
        neighbours_spatial_dist = []

        for j in range(structure.num_sites):
            centered_sph_coords = []
            neighbours_spatial_dist_atom = []

            for m,n in enumerate(range(max_shell+1)):
                centered_sph_coords = []

                neighbours = structure.get_neighbors_in_shell(structure.sites[j].coords,shells[n],0.2)
                new_cart_coords = [x.coords.tolist() for x in neighbours]
                atom_numbers = [x.specie.number for x in neighbours]
                
                #SORT DESCRIPTOR
                neighbours_spatial_dist_atom.extend(np.sort(np.array(atom_numbers)))

            neighbours_spatial_dist.append(neighbours_spatial_dist_atom)

        neighbours_spatial_dist_all.append(neighbours_spatial_dist)     

    #ALL AT ONCE
    neighbours_spatial_dist_all = np.array(neighbours_spatial_dist_all)

    neighbours_spatial_dist_all_sorted = []
    sorting = []

    for k,structure in enumerate(range(neighbours_spatial_dist_all.shape[0])):
        sorted_atoms = []
        for i in range(neighbours_spatial_dist_all.shape[1]):
            sorted_atoms.append(int(''.join([str(x) for x in neighbours_spatial_dist_all[k][i]])))
        sorting.append(np.argsort(np.array(sorted_atoms))) 
        neighbours_spatial_dist_all_sorted.append((np.array(neighbours_spatial_dist_all)[k][np.argsort(np.array(sorted_atoms))]).tolist())
    neighbours_spatial_dist_all_sorted = np.array(neighbours_spatial_dist_all_sorted)   

    neighbours_spatial_dist_all_sorted_sliced = neighbours_spatial_dist_all_sorted[:,:,1:]

    n_structures = neighbours_spatial_dist_all_sorted_sliced.shape[0]
    vector_len = neighbours_spatial_dist_all_sorted_sliced.shape[1] * neighbours_spatial_dist_all_sorted_sliced.shape[2]
    neighbours_spatial_dist_all_sorted_sliced_flat = \
    np.reshape(neighbours_spatial_dist_all_sorted_sliced, [n_structures,vector_len])

    neighbours_spatial_dist_all_sorted_sliced_reduced = \
    neighbours_spatial_dist_all_sorted_sliced[neighbours_spatial_dist_all_sorted_sliced != 8]

    vector_len = int(neighbours_spatial_dist_all_sorted_sliced_reduced.shape[0]/n_structures)

    neighbours_spatial_dist_all_sorted_sliced_reduced = \
    np.reshape(neighbours_spatial_dist_all_sorted_sliced_reduced,[n_structures,vector_len])

    descriptor = np.array(neighbours_spatial_dist_all_sorted_sliced_reduced)
    
    return descriptor


def build_descriptor_new(structures,max_radius=5):
    structure = structures[0]
    the_descriptor = []

    for i in range(structure.num_sites):

        site_distance = []
        unit_cell_neighbour = []
        centered_cart_coords = []

        for site in structure.get_all_neighbors(max_radius,sites=[structure.sites[i]])[0]:

            site_distance.append(site.distance(structure.sites[i]))
            centered_cart_coords.append(site.coords-structure.cart_coords[i])


            for j,site2 in enumerate(structure.sites):
                if site.is_periodic_image(site2) == True:
                    unit_cell_neighbour.append(j)
        unit_cell_neighbour = np.array(unit_cell_neighbour)

        centered_sph_coords = cart2sph_array(centered_cart_coords)

        spatial_distribution = np.argsort(np.array(centered_sph_coords)[:,0]*100+\
                                        np.array(centered_sph_coords)[:,1]*10 +\
                                        np.array(centered_sph_coords)[:,2])

        the_descriptor.append(unit_cell_neighbour[spatial_distribution])
    the_descriptor = np.array(the_descriptor)
    #print(the_descriptor)
    '''for k in unit_cell_neighbour[spatial_distribution]:
        print(np.round(structure.distance_matrix[0][k],4),
            np.round(structure.cart_coords[k]-structure.cart_coords[0],4))'''
    descriptor_sorted = []
    descriptor_non_sorted = []
    descriptor_atom_number = []

    for structure in structures:
        descriptor_sorted.append(np.array(structure.atomic_numbers)[the_descriptor][np.argsort(structure.atomic_numbers),:])
        descriptor_non_sorted.append(np.array(structure.atomic_numbers)[the_descriptor])

        descriptor_atom_number.append(np.c_[structure.atomic_numbers,np.array(structure.atomic_numbers)[the_descriptor][np.argsort(structure.atomic_numbers),:]])
    descriptor_sorted = np.array(descriptor_sorted)   
    descriptor_non_sorted = np.array(descriptor_non_sorted)   
    descriptor_atom_number = np.array(descriptor_atom_number)

    neighbours_spatial_dist_all = copy.deepcopy(descriptor_atom_number)

    #ALL AT ONCE
    neighbours_spatial_dist_all = np.array(neighbours_spatial_dist_all)

    neighbours_spatial_dist_all_sorted = []
    sorting = []

    for k,structure in enumerate(range(neighbours_spatial_dist_all.shape[0])):
        sorted_atoms = []
        for i in range(neighbours_spatial_dist_all.shape[1]):
            sorted_atoms.append(int(''.join([str(x) for x in neighbours_spatial_dist_all[k][i]])))
        sorting.append(np.argsort(np.array(sorted_atoms))) 
        neighbours_spatial_dist_all_sorted.append((np.array(neighbours_spatial_dist_all)[k][np.argsort(np.array(sorted_atoms))]).tolist())
    neighbours_spatial_dist_all_sorted = np.array(neighbours_spatial_dist_all_sorted)   

    neighbours_spatial_dist_all_sorted_sliced = neighbours_spatial_dist_all_sorted[:,:,1:]

    n_structures = neighbours_spatial_dist_all_sorted_sliced.shape[0]
    vector_len = neighbours_spatial_dist_all_sorted_sliced.shape[1] * neighbours_spatial_dist_all_sorted_sliced.shape[2]
    neighbours_spatial_dist_all_sorted_sliced_flat = \
    np.reshape(neighbours_spatial_dist_all_sorted_sliced, [n_structures,vector_len])

    neighbours_spatial_dist_all_sorted_sliced_reduced = \
    neighbours_spatial_dist_all_sorted_sliced[neighbours_spatial_dist_all_sorted_sliced != 8]

    vector_len = int(neighbours_spatial_dist_all_sorted_sliced_reduced.shape[0]/n_structures)

    neighbours_spatial_dist_all_sorted_sliced_reduced = \
    np.reshape(neighbours_spatial_dist_all_sorted_sliced_reduced,[n_structures,vector_len])

    descriptor = np.array(neighbours_spatial_dist_all_sorted_sliced_reduced)
    
    return descriptor


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


def build_quadratic_model(structure,use_coord = False, coord_const = None, num_vacancies = 0, 
                          alpha=1, beta = 1, lambda_1 = 2):
    # structure = pymatgen Structure object
    # weight_1 = weight for the bond energy objective
    # weight_1 = weight for the bond energy objective
    # lagrange = weight for the number of vacancies constraint
    
    from dimod import BinaryQuadraticModel
    
    X = np.arange(structure.num_sites)
    
    adjacency_matrix = build_adjacency_matrix(structure)

    Q = np.triu(-alpha*adjacency_matrix,0)
    
    bqm = BinaryQuadraticModel.from_qubo(Q)
    
    if use_coord == True:
        for i in range(structure.num_sites):  
            neighbours = np.where(adjacency_matrix[i,:] == 1)[0]
            for j in range(len(neighbours)):
                for k in range(len(neighbours)):
                    if k > j:
                        bqm.add_interaction(X[neighbours[j]],X[neighbours[k]],beta)
        if coord_const is not None and type(coord_const) == int:
            for i in range(structure.num_sites):
                bqm.add_linear(X[i],beta*np.sum(adjacency_matrix[i,:]))

    if num_vacancies == 0:
        print('Unconstrained quadratic model used')
        
        return bqm
    elif num_vacancies > 0:
        print('Unconstrained quadratic model + contraints used')
        
        c_n_vacancies = [(i,1) for i in X]

        bqm.add_linear_equality_constraint(
                c_n_vacancies,
                constant= -(structure.num_sites-num_vacancies),
                lagrange_multiplier = lambda_1
                )

        return bqm
    
    else:
        print('Please select a positive integer number of vacancies')
     
        return None


def build_bqm_discrete(structure,species,concentrations, parameters, weight=100, max_neigh = 1):
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


def build_quadratic_model_discrete_OLD(structure,species,concentrations, parameters, weight=100, max_neigh = 1):
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


def build_qubo_discrete_vacancies(structure,num_vac, alpha = 1, lambda_1 = 2, theta=100):
    
    num_sites = structure.num_sites
    num_atoms = num_sites - num_vac
    A = build_adjacency_matrix(structure)
    
    Q = np.zeros((2*num_sites,2*num_sites))
    for i in range(0,2*num_sites,2): #xc
        Q[i,i] = lambda_1*(1-2*num_atoms) - theta
        #print(i,lambda_1*(1-2*num_atoms) - theta)
    for i in range(1,2*num_sites,2): #xv
        Q[i,i] = lambda_1*(1-2*num_vac) - theta
        #print(i,lambda_1*(1-2*num_vac) - theta)
    for i in range(0,2*num_sites,2): #xcxv
        Q[i,i+1] = 2*theta
        #print(i,lambda_1*(1-2*num_vac) - theta)
    for i in range(0,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j] = 2*lambda_1
    for i in range(1,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j] = 2*lambda_1
    for i in range(0,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j+1] = alpha*A[int(i/2),int(j/2)]
            Q[i+1,j] = alpha*A[int(i/2),int(j/2)]
    return Q


def build_qubo_vacancies(structure, num_vac=0, coord_obj=False, coord_const=False, alpha = 1, lambda_1 = 2, beta=1):
    
    num_sites = structure.num_sites
    num_atoms = num_sites - num_vac
    
    A = np.triu(build_adjacency_matrix(structure),0)
    
    L = np.ones((structure.num_sites,structure.num_sites))*2*lambda_1
    L = np.triu(L,1)

    Q = np.zeros((num_sites,num_sites))

    for i in range(0,num_sites): #diag
        Q[i,i] = lambda_1*(1-2*num_atoms) 
    
    if type(coord_const) == int:
        for i in range(0,num_sites): #diag      
            Q[i,i] = Q[i,i]-beta*np.sum(A[i,:])

    if coord_obj == True:
        B = np.triu(build_adj_adjacency_matrix(structure),0)
        Q = Q - beta*B
    elif type(coord_const) == int:
        B = np.triu(build_adj_adjacency_matrix(structure),0)
        Q = Q - coord_const*beta*B
        
    Q = Q -alpha*A+L

    
    
    return Q


def build_qubo_matrix(bqm, transpose= True):
    
    # Returns the qubo matrix from a bqm model (cqm notr supported)
    
    # Build a n_atoms x n_atoms matrix containing only 0s    
    num_items = len(bqm.to_numpy_vectors().linear_biases)   
    qubo_matrix = np.array([[0.]*num_items]*num_items)
    
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


def classical_energy(x,q):
    # x is the binary vector
    # q is the qubo matrix

    E_tmp = np.matmul(x,q)
    E_classical = np.sum(x*E_tmp)
    
    return E_classical


def classical_solver(len_x, bqm, sort = True, discrete = False):
    
    # Calculate the energy of all possible solutions for a binary vector of len len_x
    # whose Hamiltonian is bqm
    # bqm can be either the bqm or the QUBO matrix
    
    import itertools

    x = np.array(list(itertools.product([0, 1], repeat=len_x)))
    
    if sort == True:
        sorting = np.argsort(np.sum(x,axis=1))
        
        x_1 = x[sorting]
    else:
        x_1 = x

    if discrete == True:
        
        xx = np.array([[0,1]*len_x]*x.shape[0])
        np.append(np.array(x)[:],np.array(x)[:],axis=1)

        X = np.zeros((x.shape[0],2*x.shape[1]),dtype=int)
        
        for i in range(x.shape[0]):
            X[i] = np.vstack([x_1[i,:], x_1[i,:]]).flatten(order='F')

        x_classical = np.array(np.logical_xor(X,xx),dtype=int)
    
    elif discrete == False:
        x_classical = x_1
   
    
    if 'numpy.ndarray' in str(type(bqm)): 
        qubo_matrix = bqm
    else:
        qubo_matrix = build_qubo_matrix(bqm)
    
    E_tmp = np.matmul(x_classical,qubo_matrix)
    E_classical = np.sum(x_classical*E_tmp,axis=1)
    
    return x_classical, E_classical


def find_exact_solutions(bqm):
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        
        from dimod import ExactSolver
        
        return ExactSolver().sample(bqm).to_pandas_dataframe()
    
    elif 'ConstrainedQuadraticModel' in str(type(bqm)):
        
        from dimod import ExactCQMSolver
        
        return ExactCQMSolver().sample_cqm(bqm).to_pandas_dataframe()


def generate_all_structures(structure_inp,num_vacancies):
    
    import itertools
    
    num_sites = structure_inp.num_sites
    
    num_atoms = num_sites - num_vacancies
    
    X = np.array(list(itertools.product([0, 1], repeat=num_sites)))
    indices = np.where(np.sum(X,axis=1)==num_atoms)[0]
    X_final = X[indices]

    all_structures = []
    for config in X_final:
        structure = copy.deepcopy(structure_inp)
        for j in np.where(config==0)[0]:
            structure.replace(j,1)
        all_structures.append(structure)    
    return all_structures


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


def qubo_classical_solver(Q, sort = True, discrete = False, return_df = True):
    
    """Calculate the energy of all possible solutions for the QUBO model.

    Extended description of function.

    Args:
        Q (numpy.array): QUBO matrix
        sort (bool): return sorted solutions
        discrete (bool): define if the QUBO model is discrete
        return_df (bool): return Pandas dataframe if True ot x_classical, E_classical if False 

    Returns:
        Pandas.dataframe: all possible solutions + Energy

    """
    
    
    import itertools
    import pandas as pd
    
    len_x = Q.shape[0]
        
    x = np.array(list(itertools.product([0, 1], repeat=len_x)))
    
    if sort == True:
        x_tmp = []
        for xx in x:
            x_tmp.append(int(''.join(xx.astype('str'))))
        sorting = np.argsort(x_tmp)        
        x_1 = x[sorting]
    else:
        x_1 = x

    if discrete == True: #TO TEST
        
        xx = np.array([[0,1]*len_x]*x.shape[0])
        np.append(np.array(x)[:],np.array(x)[:],axis=1)

        X = np.zeros((x.shape[0],2*x.shape[1]),dtype=int)
        
        for i in range(x.shape[0]):
            X[i] = np.vstack([x_1[i,:], x_1[i,:]]).flatten(order='F')

        x_classical = np.array(np.logical_xor(X,xx),dtype=int)
    
    elif discrete == False:
        x_classical = x_1
   

    
    E_tmp = np.matmul(x_classical,Q)
    E_classical = np.sum(x_classical*E_tmp,axis=1)
    
    if return_df == True:
        df = pd.DataFrame(x_classical)
        df['Energy'] = E_classical
        
        return df 
    else:
        return x_classical, E_classical   


def run_anneal(bqm,num_reads = 1000, time_limit=5, chain_strength = None, label='Test anneal', dataframe = False, 
               remove_broken_chains = False, return_config_E = False, annealing_time=20):
    
    
    if 'BinaryQuadraticModel' in str(type(bqm)):
        from dwave.system import EmbeddingComposite, DWaveSampler

        sampler = EmbeddingComposite(DWaveSampler())
        
        if chain_strength != None:
            result = sampler.sample(bqm, num_reads = num_reads, chain_strength = chain_strength, annealing_time=annealing_time,
                                      label=label, return_embedding=True)
        else:
            result = sampler.sample(bqm, num_reads = num_reads, label=label, annealing_time=annealing_time,
                                        return_embedding=True)
        
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
              chain_strength = None, concentration=None, potential=None):
    
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


def save_json_concentration(sampleset, structure, bqm, species, concetration, 
                          lambda_1, theta , file_name, num_reads = 1000, label='Test anneal', 
               file_path = 'data', save_qubo = True, chain_strength = None, concentration=None, potential=None):
    
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


def save_json_discrete_cp(structure,sampleset, bqm, theta = 10, cp_1 = 0, cp_2 = 0, 
                            num_reads = 1000, time_limit=5, label='Test anneal', 
                            remove_broken_chains = False, file_path = 'data', file_name = '', save_qubo = True,
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
                  'theta': theta,
                  'cp_1': cp_1,
                  'cp_2': cp_2,
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


    name = file_name + '_%s_%s_cp1%s_cp2%s_t%s_r%s_t%s_%s.json'%(structure.composition.formula,
                                                                    model,cp_1,cp_2,
                                                                    theta,num_reads,time_limit,time_stamp)
    
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


### ALL DISCRETE FUNCTIONS

def build_binary_vector(atomic_numbers,atom_types=None):
    """Summary line.

    Extended description of function.

    Args:
        atomic_numbers (list): List of atom number of the sites in the structure
        atom_types (list): List of 2 elements. List element 0 = atomic_number of site == 0, 
                           list element 1 = atomic_number of site == 1

    Returns:
        List: Binary list of atomic numbers

    """
    
    atomic_numbers = np.array(atomic_numbers)
    num_sites = len(atomic_numbers)
    
    if atom_types == None:
        species = np.unique(atomic_numbers)
    else:
        species = atom_types
    
    binary_atomic_numbers = np.zeros(num_sites,dtype=int)
    
    for i,species_type in enumerate(species):

        sites = np.where(atomic_numbers == species_type)[0]

        binary_atomic_numbers[sites] = i
    
    return binary_atomic_numbers

     

    
def build_qubo_discrete_constraints(structure, species, concentration=None, chem_potential = None,
                                    lambda_1 = 2, theta=100):
    
    num_sites = structure.num_sites
    num_species = len(species)
    num_elements = num_sites*num_species
    
    
    Q = np.zeros((num_elements,num_elements))
    
    if concentration is not None and type(concentration) is list:
        
        for n in range(num_species):
            for i in range(n,num_elements,num_species): #i-i k-k diagonal
                Q[i,i] = lambda_1*(1-2*concentration[n]) - theta   
                for k in range(1,num_species-n%num_species): #i-i k-l off-diagonal
                    if i+k < num_elements:
                        Q[i,i+k] = 2*theta
        for n in range(num_elements):                
            for j in range(n+num_species,num_elements,num_species): #i-i k-k diagonal

                    Q[n,j] = 2*lambda_1
    
    elif chem_potential is not None and type(chem_potential) is list:
        
        for n in range(num_species):
            for i in range(n,num_elements,num_species): #i-i k-k diagonal
                Q[i,i] = chem_potential[n] - theta   
                for k in range(1,num_species-n%num_species): #i-i k-l off-diagonal
                    if i+k < num_elements:
                        Q[i,i+k] = 2*theta
    return Q


def build_qubo_binary_constraints(structure, concentration=None, chem_potential = None,
                                    lambda_1 = 2):
    """Summary line.

    Extended description of function.

    Args:
        structure (pymatgen Structure object): Description of arg1
        concentration (int): concentration of the species corresponding to |1>
        chem_potential (float): chemical potential of the species corresponding to |1>

    Returns:
        numpy.array: QUBO matrix containing the contraints only

    """
    
    if concentration is not None and chem_potential is not None:
        print('Please select either concentration or chemical potential')
        return None
    
    num_sites = structure.num_sites
    #num_species = len(species)
    num_elements = num_sites#*num_species
    
    
    Q = np.zeros((num_elements,num_elements))
    
    if concentration is not None and type(concentration) is int:
        
        for i in range(0,num_sites): #diag
            Q[i,i] = lambda_1*(1-2*concentration) 
            for j in range(i+1,num_sites):
                Q[i,j] = 2*lambda_1            
    
    elif chem_potential is not None and type(chem_potential) is float:
        np.fill_diagonal(Q,chem_potential)
        
            
    return Q


def build_qubo_discrete_interaction(structure, species, parameters, alpha=1, atomic_contribution=None, max_neigh = 1):
    
    #returns an N_sites x N_sites matrix where the i,j element represent the interaction between i and j
    #parameters is a list of list where the 
        # first row represents the first neightbour interaction 
        # second row represents the second neightbour interaction and so on

        #Within the same row, the k-element represent the interaction between species i+j
        #(Think of it as an upper triangular matrix)
        

    num_sites = structure.num_sites
    num_species = len(species)
    num_elements = num_sites*num_species

    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    #distance_matrix_filter
    
    interaction_matrix = np.zeros((num_elements,num_elements))
    
    #loop 
    if atomic_contribution != None:
        for i in range(num_species):
            interaction_matrix[np.arange(i,num_sites*num_species,num_species),np.arange(i,num_sites*num_species,num_species)] = atomic_contribution[i]
    
    
    parameters = np.array(parameters)
    for i in range(num_sites):
        for j in range(i,num_sites):
            g = distance_matrix_filter[i,j]
            if g > 0:
                for k in range(num_species):
                    #for l in range(k,num_species): for upper diagonal
                    for l in range(num_species):
                        param = alpha*parameters[g-1][k+l] #the k+l sum is uniquely defining the pair potential
                        interaction_matrix[i*num_species+k,j*num_species+l] = param
                        #interaction_matrix[i*num_species-k,j*num_species+l] = param
                
    return interaction_matrix


def build_qubo_binary_interaction(structure, parameters, max_neigh = None):
    
    #returns an N_sites x N_sites matrix where the i,j element represent the interaction between i and j
    #parameters is a list of list where the 
        # first row represents the first neightbour interaction 
        # second row represents the second neightbour interaction and so on

        #Within the same row, the k-element represent the interaction between species i+j
        #(Think of it as an upper triangular matrix)
        

    num_sites = structure.num_sites
    #num_species = len(species)
    num_elements = num_sites#*num_species
    
    if max_neigh == None:
        max_neigh = len(parameters)
        
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    
    distance_matrix_filter = np.zeros((num_sites,num_sites),int)

    for i,s in enumerate(shells[0:max_neigh+1]):
        row_index = np.where(distance_matrix == s)[0]
        col_index = np.where(distance_matrix == s)[1]
        distance_matrix_filter[row_index,col_index] = i
    distance_matrix_filter = np.triu(distance_matrix_filter,0)
    interaction_matrix = np.zeros((num_elements,num_elements))
    
    #I DON'T THINK THIS IS NEEDED FOR THE BINARY 
    # if atomic_contribution != None:
    #    for i in range(num_species):
    #        interaction_matrix[np.arange(i,num_sites*num_species,num_species),np.arange(i,num_sites*num_species,num_species)] = atomic_contribution[i]
    
    parameters = np.array(parameters)
    diag_element = parameters[:,1]-parameters[:,0] # DeltaEcc/cn
    off_diag_element = (parameters[:,2]-parameters[:,0])-2*(parameters[:,1]-parameters[:,0]) #DeltaEnn/cc-2DeltaEcc/cn

    for i in range(num_sites):
        for j in range(0,num_sites):
            g = distance_matrix_filter[i,j]
            #print(i,j,g)
            if g > 0:
                interaction_matrix[i,i] += diag_element[g-1]
                interaction_matrix[j,j] += diag_element[g-1]
                interaction_matrix[i,j] += off_diag_element[g-1]
                
    return interaction_matrix


def build_qubo_binary(structure , parameters,
    concentration = None, chem_potential=None, max_neigh = 1, lambda_1 = 1):

    #combine the build_qubo_discrete_constraints and build_ip_matrix to make the QUBO matrix and convert to bqm

    from dimod import BinaryQuadraticModel, Binary
    
    Q = build_qubo_binary_interaction(structure, parameters, max_neigh = max_neigh)+ \
        build_qubo_discrete_constraints(structure,concentration=concentration, chem_potential=chem_potential,\
        lambda_1 = lambda_1) 
   
    #bqm = BinaryQuadraticModel.from_qubo(Q)

    return Q


def build_quadratic_model_discrete(structure ,species, parameters,
    concentration = None, chem_potential=None, max_neigh = 1, alpha=1,lambda_1 = 2, theta=10):

    #combine the build_qubo_discrete_constraints and build_ip_matrix to make the QUBO matrix and convert to bqm

    from dimod import BinaryQuadraticModel, Binary
    
    Q = build_qubo_discrete_interaction(structure, species, parameters, alpha=alpha, max_neigh = max_neigh)+ \
        build_qubo_discrete_constraints(structure,species,concentration=concentration, chem_potential=chem_potential,\
        lambda_1 = lambda_1, theta=theta) 
   
    bqm = BinaryQuadraticModel.from_qubo(Q)

    return bqm


def build_ip_parameters(structure, buckingham):
    #This function takes the IP parameters and returns a number (list) for the QUBO matrix
    '''
    num_sites = structure.num_sites
    num_species = len(species)
    num_elements = num_sites*num_species
    
    distance_matrix = np.round(structure.distance_matrix,5)
    shells = np.unique(np.round(distance_matrix,5))
    
    # Generate an all False matrix
    distance_matrix_filter = (distance_matrix == -1)

    
    # Only add the atoms within the shells up to max_neigh 
    for neigh in range(1,max_neigh+1):
    distance_matrix_filter +=  distance_matrix == shells[neigh] 
    
    ip_matrix = np.zeros((num_elements,num_elements))
    parameters = np.array(parameters)
    for i in range(num_sites):
        for j in range(i,num_sites):
            if distance_matrix_filter[i,j] == True:
                index = -1
                for k in range(num_species):
                    for l in range(k,num_species):
                        index += 1
                        param = parameters[index]                       
                        if test == True:
                            ip_matrix[i*num_species+k,j*num_species+l] = \
                            adjacency_matrix[i,j]*param
                        else:
                            ip_matrix[i*num_species+k,j*num_species+l] = \
                            param[0] * np.exp((-distance_matrix[i,j])/(param[1]))- \
                            ((param[2])/((distance_matrix[i,j])**6))            
                        return ip_matrix'''
    return None


def build_qubo_discrete_vacancies(structure,num_vac, alpha = 1, lambda_1 = 2, theta=100):

    
    # PROBABLY REPLACED BY ABOVE FUNCTIONS


    num_sites = structure.num_sites
    num_atoms = num_sites - num_vac
    A = build_adjacency_matrix(structure)
    
    Q = np.zeros((2*num_sites,2*num_sites))
    
    for i in range(0,2*num_sites,2): #xc
        Q[i,i] = lambda_1*(1-2*num_atoms) - theta
        #print(i,lambda_1*(1-2*num_atoms) - theta)
    for i in range(1,2*num_sites,2): #xv
        Q[i,i] = lambda_1*(1-2*num_vac) - theta
        #print(i,lambda_1*(1-2*num_vac) - theta)
    for i in range(0,2*num_sites,2): #xcxv
        Q[i,i+1] = 2*theta
        #print(i,lambda_1*(1-2*num_vac) - theta)
    for i in range(0,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j] = 2*lambda_1
    for i in range(1,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j] = 2*lambda_1
    for i in range(0,2*num_sites,2): 
        for j in range(i+2,2*num_sites,2):
            Q[i,j+1] = alpha*A[int(i/2),int(j/2)]
            Q[i+1,j] = alpha*A[int(i/2),int(j/2)]
    return Q


def save_json_discrete(structure,sampleset, bqm, n_dopant_atom, lambda_1 = 0, theta = 0, 
                            num_reads = 1000, time_limit=5, label='Test anneal', 
                            remove_broken_chains = False, file_path = 'data', file_name = '', save_qubo = True,
                            chain_strength = None, concentration = None, potential=None):
    
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
    elif 'numpy.ndarray' in str(type(bqm)):
        model = 'Q'
        time_limit = 0
    
    date_time = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    time_stamp = int(time.time())
    
    if save_qubo == True:
        if model == 'Q':
            qubo_matrix = bqm.flatten().tolist()
        elif model == 'bqm':
            qubo_matrix = build_qubo_matrix(bqm).flatten().tolist()
    elif save_qubo == False:
        qubo_matrix = None
    
    
        
    param_dict = {'date_time': date_time,
                  'time_stamp': time_stamp,
                  'structure': structure.composition.formula,
                  'N atoms' : structure.num_sites,
                    'model': model,
                  'theta': theta,
                  'lambda': lambda_1,
                     'num_reads' : num_reads, 
                  'time_limit': time_limit,
                  'label':label, 
                  'remove_broken_chains' : remove_broken_chains,
                  'chain_strength' : chain_strength,
                  #'qubo_matrix': qubo_matrix,
                  'qpu_anneal_time_per_sample': sampleset.info['timing']['qpu_anneal_time_per_sample'],
                  'qubo_matrix': qubo_matrix,
                  'concentration': concentration,
                  'potential': potential
                     
    }
    
    json_string = dataframe.to_json()    
    json_object = json.loads(json_string)
    json_object['parameters'] = param_dict


    name = file_name + '_%s_%s_%s_l%s_t%s_r%s_t%s_%s.json'%(structure.composition.formula, str(n_dopant_atom),
                                                                    model,lambda_1,
                                                                    theta,num_reads,time_limit,time_stamp)
    
    file_name = path.join(file_path,name)

    with open(file_name, 'w') as f:
        json.dump(json_object, f)


def save_json_discrete_potential(structure,sampleset, bqm, theta = 0, index = None,
                                 
                            num_reads = 1000, time_limit=5, label='Test anneal', 
                            remove_broken_chains = False, file_path = 'data', file_name = '', save_qubo = True,
                            chain_strength = None, concentration = None, potential=None):
    
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
    elif 'numpy.ndarray' in str(type(bqm)):
        model = 'Q'
        time_limit = 0
    
    date_time = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    time_stamp = int(time.time())
    
    if save_qubo == True:
        if model == 'Q':
            qubo_matrix = bqm.flatten().tolist()
        elif model == 'bqm':
            qubo_matrix = build_qubo_matrix(bqm).flatten().tolist()
    elif save_qubo == False:
        qubo_matrix = None
    
    
        
    param_dict = {'date_time': date_time,
                  'time_stamp': time_stamp,
                  'structure': structure.composition.formula,
                  'N atoms' : structure.num_sites,
                    'model': model,
                  'theta': theta,
                     'num_reads' : num_reads, 
                  'time_limit': time_limit,
                  'label':label, 
                  'remove_broken_chains' : remove_broken_chains,
                  'chain_strength' : chain_strength,
                  #'qubo_matrix': qubo_matrix,
                  'qpu_anneal_time_per_sample': sampleset.info['timing']['qpu_anneal_time_per_sample'],
                  'qubo_matrix': qubo_matrix,
                  'concentration': concentration,
                  'potential': potential
                     
    }
    
    json_string = dataframe.to_json()    
    json_object = json.loads(json_string)
    json_object['parameters'] = param_dict


    name = file_name + '_%s_%s_i%s_t%s_r%s_t%s_%s.json'%(structure.composition.formula, model, index,
                                                        theta,num_reads,time_limit,time_stamp)
    
    file_name = path.join(file_path,name)

    with open(file_name, 'w') as f:
        json.dump(json_object, f)

def save_json_binary_potential(structure,sampleset, bqm, index = None,
                            num_reads = 1000, time_limit=5, label='Test anneal', 
                            remove_broken_chains = False, file_path = 'data', file_name = '', save_qubo = True,
                            chain_strength = None, concentration = None, potential=None):
    
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
    elif 'numpy.ndarray' in str(type(bqm)):
        model = 'Q'
        time_limit = 0
    
    date_time = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    time_stamp = int(time.time())
    
    if save_qubo == True:
        if model == 'Q':
            qubo_matrix = bqm.flatten().tolist()
        elif model == 'bqm':
            qubo_matrix = build_qubo_matrix(bqm).flatten().tolist()
    elif save_qubo == False:
        qubo_matrix = None
    
    
        
    param_dict = {'date_time': date_time,
                  'time_stamp': time_stamp,
                  'structure': structure.composition.formula,
                  'N atoms' : structure.num_sites,
                    'model': model,
                     'num_reads' : num_reads, 
                  'time_limit': time_limit,
                  'label':label, 
                  'remove_broken_chains' : remove_broken_chains,
                  'chain_strength' : chain_strength,
                  #'qubo_matrix': qubo_matrix,
                  'qpu_anneal_time_per_sample': sampleset.info['timing']['qpu_anneal_time_per_sample'],
                  'qubo_matrix': qubo_matrix,
                  'concentration': concentration,
                  'potential': potential
                     
    }
    
    json_string = dataframe.to_json()    
    json_object = json.loads(json_string)
    json_object['parameters'] = param_dict


    name = file_name + '_%s_%s_i%s_t%s_r%s_%s.json'%(structure.composition.formula, model, index,
                                                        num_reads,time_limit,time_stamp)
    
    file_name = path.join(file_path,name)

    with open(file_name, 'w') as f:
        json.dump(json_object, f)