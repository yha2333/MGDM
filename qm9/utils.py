import torch


def compute_mean_mad(dataset, properties, dataset_name):
    if dataset_name == 'qm9' or dataset_name == 'qm9_second_half' or dataset_name == 'qm9_first_half':
        return compute_mean_mad_from_dataloader(dataset, properties)
    # elif dataset_name == 'qm9_second_half' or dataset_name == 'qm9_second_half':
        # return compute_mean_mad_from_dataloader(dataloaders['valid'], properties)
    else:
        raise Exception('Wrong dataset name')


def compute_mean_mad_from_dataloader(dataset, properties):
    property_norms = {}
    for property_key in properties:
        values = dataset.data[property_key]
        mean = torch.mean(values)
        ma = torch.abs(values - mean)
        mad = torch.mean(ma)
        max_value = torch.max(values)
        min_value = torch.min(values)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['mad'] = mad
        property_norms[property_key]['max'] = max_value
        property_norms[property_key]['min'] = min_value
    return property_norms
    #
    # property_norms = {}
    #
    # # 初始化统计变量
    # for property_key in properties:
    #     property_norms[property_key] = {'sum': 0, 'sum_abs_diff': 0, 'count': 0}
    #
    # # 迭代 DataLoader 获取数据
    # for batch in dataset:
    #     for property_key in properties:
    #         values = batch[property_key]
    #         count = len(values)
    #
    #         # 更新统计变量
    #         property_norms[property_key]['sum'] += torch.sum(values)
    #         property_norms[property_key]['sum_abs_diff'] += torch.sum(torch.abs(values - torch.mean(values)))
    #         property_norms[property_key]['count'] += count
    #
    # # 计算最终的统计值
    # for property_key in properties:
    #     mean = property_norms[property_key]['sum'] / property_norms[property_key]['count']
    #     mad = property_norms[property_key]['sum_abs_diff'] / property_norms[property_key]['count']
    #     max_value = torch.max(values)
    #     min_value = torch.min(values)
    #
    #     property_norms[property_key]['mean'] = mean
    #     property_norms[property_key]['mad'] = mad
    #     property_norms[property_key]['max'] = max_value
    #     property_norms[property_key]['min'] = min_value
    #
    # return property_norms

edges_dic = {}


def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx * n_nodes)
                        cols.append(j + batch_idx * n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges


def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars


def prepare_context(conditioning, minibatch, property_norms):
    # batch_size = minibatch['batch'][-1] + 1
    context_node_nf = 0
    context_list = []
    for key in conditioning:
        properties = minibatch[key]
        properties = (properties - property_norms[key]['mean']) / property_norms[key]['mad']
        if len(properties.size()) == 1:
            # Global feature.
            # assert properties.size() == (batch_size,)
            properties = properties.index_select(0, minibatch['batch'])
            context_list.append(properties.unsqueeze(1))
            context_node_nf += 1
        elif len(properties.size()) == 2 or len(properties.size()) == 3:
            # Node feature.
            # assert properties.size(0) == batch_size

            context_key = properties

            # Inflate if necessary.
            if len(properties.size()) == 2:
                context_key = context_key.unsqueeze(2)

            context_list.append(context_key)
            context_node_nf += context_key.size(2)
        else:
            raise ValueError('Invalid tensor size, more than 3 axes.')
    # Concatenate
    context = torch.cat(context_list, dim=1)
    # Mask disabled nodes!
    assert context.size(1) == context_node_nf
    return context
