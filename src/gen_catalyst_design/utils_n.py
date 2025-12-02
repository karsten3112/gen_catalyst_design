def embed_elements_as_onehot(elements:list, element_pool:list):
    mapping_dict = {element:i for i, element in enumerate(element_pool)}
    return torch.stack([get_onehot(element=element, mapping_dict=mapping_dict) for element in elements])

def embed_cluster_as_onehots(atoms:Atoms, element_pool:list):
    elements = atoms.get_chemical_symbols()
    return embed_elements_as_onehot(elements=elements, element_pool=element_pool)

def get_onehot(element:str, mapping_dict:dict):
    onehot = F.one_hot(torch.tensor(mapping_dict[element]), len(mapping_dict))
    return onehot

def get_graph_from_datadict(datadict:dict, template_atoms:Atoms, element_pool:list, condition_key:str=None):
    from gen_catalyst_design.discrete_space_diffusion import Graph
    edges_list = get_edges_list_from_connectivity(template_atoms.info["connectivity"])
    edge_index = torch.tensor(edges_list, dtype=torch.long).reshape(2,-1)
    graph = Graph(
        x=embed_elements_as_onehot(elements=datadict["elements"], element_pool=element_pool),
        edge_index=edge_index,
        pos=torch.tensor(template_atoms.positions)
    )
    if condition_key is not None:
        if condition_key in datadict:
            graph.y = datadict[condition_key]
        else:
            raise Exception(f"condition key {condition_key} is not available in datadict, having: {datadict.keys()}")
    return graph

def get_dataset_from_datadicts(datadicts:list, template_atoms:Atoms, element_pool:list, condition_key:str=None):
    from gen_catalyst_design.discrete_space_diffusion import GraphDataset
    graph_list = [
        get_graph_from_datadict(
            datadict=datadict, 
            template_atoms=template_atoms, 
            element_pool=element_pool, 
            condition_key=condition_key
        )
        for datadict in datadicts
    ]
    return GraphDataset(graph_list=graph_list)


def get_dataloaders_from_datadicts(
        data_dicts:list, 
        element_pool:list,
        template_atoms:Atoms,
        batch_size:int=42,
        condition_key:str="class", 
        train_val_split:float=0.1,
        loader_kwargs:dict={} 
    ):
    random.shuffle(data_dicts)
    split_index = int((1-train_val_split)*len(data_dicts))
    train_dataset = get_dataset_from_datadicts(
        datadicts=data_dicts[:split_index],
        template_atoms=template_atoms,
        element_pool=element_pool,
        condition_key=condition_key
    )
    val_dataset = get_dataset_from_datadicts(
        datadicts=data_dicts[split_index:],
        template_atoms=template_atoms,
        element_pool=element_pool,
        condition_key=condition_key
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )
    return train_loader, val_loader