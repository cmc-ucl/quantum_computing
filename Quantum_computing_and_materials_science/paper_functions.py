def write_crystal_files_vacancies(dataframe,structure,input_file,input_name,directory,
                        dimensionality=2,slurm_file_crystal=None, return_slurm = False):
        
    # dataframe: dataframe containing the structures (it is advised to only have the symmetry irreducible ones)
    # structure: pymatgen structure 
    # input_file: location of the sample input file
    # input_name: name of the input files generated
    # directory: directory where the files should be saved
    # dimensionality: dimensionality of the system
    # slurm_file_crystal: sample slurm file
    # return_slurm: whether to return the slurm file or not
    
    import copy
    import numpy as np
    
    from CRYSTALpytools.crystal_io import Crystal_input
    from CRYSTALpytools.convert import cry_pmg2gui
    
    if slurm_file_crystal != None:
        file = open(slurm_file_crystal, 'r')
        slurm_file_crystal_data = file.readlines()
        file.close()    
    else:
        slurm_file_crystal_data = slurm_file_crystal
        
    crystal_input = Crystal_input().from_file(input_file)
    
    df = dataframe
    num_atoms = sum([x.isdigit() for x in df.columns])  
    
    for i in df.index:
        array = df.loc[i].to_numpy()[0:num_atoms]
        structure_tmp = copy.deepcopy(structure)        
        structure_tmp.remove_sites(np.where(array == 0)[0])
        vac = structure.num_sites - structure_tmp.num_sites
        structure_gui = cry_pmg2gui(structure_tmp,pbc=[True, True, False],symmetry=False)
        name = input_name+'_%sv_%s'%(str(vac),i)
        structure_gui.write_gui(directory+'/%s.gui'%name,symm=False)
        crystal_input.to_file(directory+'/%s.d12'%name)
        
    
    if return_slurm == True:
        return slurm_file_crystal_data