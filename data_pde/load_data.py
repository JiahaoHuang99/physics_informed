def load_data(DATASET_NAME, dataset_params):
    """
    select the dataset
    """
    # Airfoil Dataset
    if DATASET_NAME in ['AIRFOIL', 'AIRFOIL_DEEPMIND']:
        from data_pde.dataset_airfoil_deepmind import LoadAirfoilDatasetDGL as dataset_class

    # Burgers' Dataset
    if DATASET_NAME in ['BURGERS', 'BURGERS_PDEBench']:
        from data_pde.dataset_burgers_pdebench import LoadBurgersDatasetDatasetDGL as dataset_class

    # Convection Diffusion Dataset
    elif DATASET_NAME in ['CONVECTION_DIFFUSION_D2', 'CONVECTION_DIFFUSION_Aalto_D2']:
        from data_pde.dataset_convection_diffusion_aalto_d2 import LoadConvectionDiffusionDatasetDGL as dataset_class

    # Incompressible Navier-Stokes Dataset
    elif DATASET_NAME in ['INCOMPRESSIBLE_NAVIER_STOKES_2D_D2', 'INCOMPRESSIBLE_NAVIER_STOKES_2D_PDEBench_D2']:
        from data_pde.dataset_incompressible_navier_stokes_2d_pdebench_d2 import LoadIncompressibleNavierStokes2DDatasetDGL as dataset_class

    # Darcy flow Dataset
    elif DATASET_NAME in ['DARCY_FLOW', 'DARCY_FLOW_PDEBench']:
        raise NotImplementedError
        from data_pde.dataset_darcy_flow_pdebench import LoadDarcyFlowDatasetDGL as dataset_class

    elif DATASET_NAME in ['DARCY_FLOW_PDEBench_D2']:
        from data_pde.dataset_darcy_flow_pdebench_d2 import LoadDarcyFlowDatasetDGL as dataset_class

    elif DATASET_NAME in ['DARCY_FLOW_CalTech']:
        from data_pde.dataset_darcy_flow_caltech import LoadDarcyFlowDatasetDGL as dataset_class

    # HEAT Dataset
    elif DATASET_NAME in ['HEAT', 'HEAT_Aalto']:
        from data_pde.dataset_heat_aalto import LoadHeatDatasetDGL as dataset_class

    # Poisson Dataset
    elif DATASET_NAME in ['POISSON', 'POISSON_KAISERSLAUTERN']:
        raise NotImplementedError
        from data_pde.dataset_poisson_kaiserslautern import LoadPoissonDatasetDGL as dataset_class

    # Reaction Diffusion Dataset
    elif DATASET_NAME in ['DIFFUSION_REACTION_2D', 'DIFFUSION_REACTION_2D_PDEBench']:
        from data_pde.dataset_diffusion_reaction_2d_pdebench_d2 import LoadDiffusionReaction2DDatasetDGL as dataset_class

    # Shallow Water Dataset
    elif DATASET_NAME in ['SHALLOW_WATER_2D', 'SHALLOW_WATER_2D_PDEBench']:
        from data_pde.dataset_shallow_water_2d_pdebench_d2 import LoadShallowWater2DDatasetDGL as dataset_class

    # Incompressible Navier-Stokes Dataset
    elif DATASET_NAME in ['COMPRESSIBLE_NAVIER_STOKES_2D_D2', 'COMPRESSIBLE_NAVIER_STOKES_2D_D2_PDEBench']:
        from data_pde.dataset_compressible_navier_stokes_2d_pdebench_d2 import LoadCompressibleNavierStokes2DDatasetDGL as dataset_class

    else:
        raise ValueError('Dataset {} not found'.format(DATASET_NAME))

    return dataset_class(dataset_params)