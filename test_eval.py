import argparse
import math

from eval import retrieve_qm9_smiles, analyze_stability_for_molecules, retrieve_geom_smiles

from configs.datasets_config import get_dataset_info
from models.epsnet import *
from utils.datasets import *
from utils.misc import *
from utils.reconstruct import *
from utils.sample import *

from torch import nn

from collections import OrderedDict

def RMSD(probe, ref):
    rmsd = 0.0
    # print(amap)
    assert len(probe) == len(ref)
    atom_num = len(probe)
    for i in range(len(probe)):
        posp = probe[i]
        posf = ref[i]
        rmsd += dist_2(posp, posf)
    rmsd = math.sqrt(rmsd / atom_num)
    return rmsd


def dist_2(atoma_xyz, atomb_xyz):
    dis2 = 0.0
    for i, j in zip(atoma_xyz, atomb_xyz):
        dis2 += (i - j) ** 2
    return dis2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='qm9, geom')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--ckpt', type=str, help='path for loading the checkpoint')
    parser.add_argument('--save_traj', action='store_true',
                        help='whether store the whole trajectory for sampling')
    parser.add_argument('--save_sdf', type=bool, default=False,
                        help='whether save the molecule as sdf format')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument("--context", nargs='+', default=[],
                        help='arguments : homo | lumo | alpha | gap | mu | Cv')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=100)
    # Parameters for model sampling
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=1000,
                        help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=float('inf'),
                        help='enable global gradients only when noise is low')
    parser.add_argument('--local_start_sigma', type=float, default=float('inf'),
                        help='enable local gradients only when noise is low')
    parser.add_argument('--w_global_pos', type=float, default=1.0,
                        help='weight for global pos gradients')
    parser.add_argument('--w_local_pos', type=float, default=1.0,
                        help='weight for local pos gradients')
    parser.add_argument('--w_global_node', type=float, default=4.0,
                        help='weight for global node gradients')
    parser.add_argument('--w_local_node', type=float, default=5.0,
                        help='weight for local node gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='generalized',
                        help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                        help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # Load checkpoint

    ckpt = torch.load(args.ckpt)

    args.dataset = 'qm9' if 'qm9' in args.ckpt else 'geom'
    config = ckpt['config']

    seed_all(args.seed)  # config.train.seed 3407 11 2021
    log_dir = os.path.dirname(os.path.dirname(args.ckpt))

    # args.global_start_sigma = 1 # float('inf')
    # args.local_start_sigma = 1
    args.n_steps = ckpt['config'].model.num_diffusion_timesteps


    # data_list
    dataset_info = get_dataset_info(args.dataset, False)
    num_samples = args.num_samples
    batch_size = args.batch_size
    data_list, _ = construct_dataset(num_samples, batch_size, dataset_info)

    # Logging
    tag = 'result'
    # Sample 10000 molecules for evaluation
    if num_samples < 10000:
        tag = 'test'

    output_dir = get_new_log_dir(
        log_dir,
        args.sampling_type + "_{}_sigma{}_build_all_bonds_".format(args.seed, args.global_start_sigma) + tag,
        tag=tag
    )
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Model
    logger.info('Building model...')
    logger.info(ckpt['config'].model['network'])
    model = get_model(ckpt['config'].model).to(args.device)
    model.load_state_dict(ckpt['model'],strict=False)

    # 遍历 ckpt['model'] 中的所有键，调整 global_tokens 的形状
    # for key, value in ckpt['model'].items():
    #     if 'global_tokens' in key:
    #         # 获取当前模型中对应的参数
    #         model_param = model
    #         for sub_key in key.split('.')[:-1]:  # 逐级获取子模块
    #             model_param = getattr(model_param, sub_key)
    #         model_param = getattr(model_param, key.split('.')[-1])  # 获取最终的参数
    #
    #         # 调整形状
    #         if value.shape[0] > model_param.shape[0]:
    #             print(f"{key}: {value.shape} -> {model_param.shape}")
    #             # 填充
    #             pad_size = value.shape[0] - model_param.shape[0]
    #             pad_tensor = torch.zeros((pad_size, model_param.shape[1]), dtype=model_param.dtype,
    #                                      device=model_param.device)
    #             # 获取父模块
    #             parent_module = model
    #             for sub_key in key.split('.')[:-1]:  # 逐级获取父模块
    #                 parent_module = getattr(parent_module, sub_key)
    #             # 将 torch.cat 的结果包装成 nn.Parameter
    #             new_global_tokens = nn.Parameter(torch.cat([model_param, pad_tensor], dim=0))
    #             # 修改父模块的参数
    #             setattr(parent_module, key.split('.')[-1], new_global_tokens)
    #             print(f"{key}: {value.shape} ->11 {getattr(parent_module, key.split('.')[-1]).shape}")
    #         elif value.shape[0] < model_param.shape[0]:
    #             # 截取
    #             parent_module = model
    #             for sub_key in key.split('.')[:-1]:  # 逐级获取父模块
    #                 parent_module = getattr(parent_module, sub_key)
    #             # 将截取结果包装成 nn.Parameter
    #             new_global_tokens = nn.Parameter(model_param[:value.shape[0], :])
    #             setattr(parent_module, key.split('.')[-1], new_global_tokens)
    #
    # # 加载调整后的 state_dict
    # model.load_state_dict(ckpt['model'], strict=False)
    # 手动调整 state_dict
    # updated_state_dict = OrderedDict()
    # for key, value in ckpt['model'].items():
    #     if 'global_tokens' in key:
    #         # 调整形状
    #         if value.shape[0] > 32:  # 如果检查点中的形状大于当前模型
    #             value = value[:32, :]  # 截取前 32 行
    #         elif value.shape[0] < 32:  # 如果检查点中的形状小于当前模型
    #             pad_size = 32 - value.shape[0]
    #             pad_tensor = torch.zeros((pad_size, value.shape[1]), dtype=value.dtype, device=value.device)
    #             value = torch.cat([value, pad_tensor], dim=0)  # 填充到 32 行
    #     updated_state_dict[key] = value

    # 加载调整后的 state_dict
    # model.load_state_dict(updated_state_dict, strict=False)

    print(ckpt['config'].model)

    model.eval()

    # Metric
    sa_list = []
    valid = 0
    smile_list = []
    results = []
    sum_rmsd = 0

    clip_local = 12
    stable = 0
    logger.info('dataset:%s' % args.dataset)
    logger.info('sample num:%d' % num_samples)
    logger.info('sample method:%s' % args.sampling_type)
    logger.info('w_global_pos:%.1f' % args.w_global_pos)
    logger.info('w_global_node:%.1f' % args.w_global_node)
    logger.info('w_local_pos:%.1f' % args.w_local_pos)
    logger.info('w_local_node:%.1f' % args.w_local_node)
    show_detail = True
    sdf = args.save_sdf

    position_list = []
    atom_type_list = []
    mols_dict = []

    FINISHED = True

    for n, datas in enumerate(tqdm(data_list)):
        with torch.no_grad():
            context = None
            start_time = time.time()
            batch = Batch.from_data_list(datas).to(args.device)
            FINISHED = True
            while FINISHED:
                try:
                    pos_gen, pos_gen_traj, atom_type, atom_traj = model.langevin_dynamics_sample(
                        atom_type=batch.x,
                        # atom_type = batch.atom_feat_full.float(),
                        pos_init=batch.pos,
                        bond_index=batch.edge_index,
                        bond_type=None,
                        batch=batch.batch,
                        num_graphs=batch.num_graphs,
                        extend_order=False,
                        n_steps=args.n_steps,
                        step_lr=1e-6,
                        w_global_pos=args.w_global_pos,
                        w_global_node=args.w_global_node,
                        w_local_pos=args.w_local_pos,
                        w_local_node=args.w_local_node,
                        global_start_sigma=args.global_start_sigma,
                        clip=args.clip,
                        clip_local=clip_local,
                        sampling_type=args.sampling_type,
                        eta=args.eta,
                        context=context

                    )
                    pos_list = unbatch(pos_gen, batch.batch)
                    atom_list = unbatch(atom_type, batch.batch)
                    current_num_samples = (n + 1) * batch_size
                    secs_per_sample = (time.time() - start_time) / current_num_samples
                    print('\t %d/%d Molecules generated at %.2f secs/sample' % (
                        current_num_samples, num_samples, secs_per_sample))
                    for m in range(batch_size):
                        pos = pos_list[m]
                        atom_type = atom_list[m]

                        # charge
                        atom_type = atom_type[:, :-1]
                        charge = atom_type[:, -1]

                        atom_type = torch.argmax(atom_type, dim=1)
                        position_list.append(pos.cpu().detach())
                        atom_type_list.append(atom_type.cpu().detach())

                        a = 0

                        mol = build_molecule(pos, atom_type, dataset_info)
                        smile = mol2smiles(mol)

                        if show_detail:
                            print("g pos:", pos)
                            print("generated element:", atom_type)
                            print("generated smile:", smile)
                        result = {'atom_type': atom_type, 'pos': pos, 'smile': smile}
                        results.append(result)
                        if smile is not None:
                            valid += 1
                            if "." not in smile:
                                stable += 1

                            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                            mol = largest_mol
                            smile = mol2smiles(mol)
                            smile_list.append(smile)

                            if sdf:
                                conf = Chem.Conformer(mol.GetNumAtoms())
                                for i in range(mol.GetNumAtoms()):
                                    conf.SetAtomPosition(i, (float(pos[i][0]), float(pos[i][1]), float(pos[i][2])))
                                mol.AddConformer(conf)
                                sdf_dir = './results/ddpm'
                                if not os.path.exists(sdf_dir):
                                    os.mkdir(sdf_dir)
                                writer = Chem.SDWriter(os.path.join(sdf_dir, '%s.sdf' % 'full_{}_{}'.format(n, m)))
                                writer.write(mol, confId=0)
                                writer.close()
                    FINISHED = False

                except FloatingPointError:
                    clip_local = 10
                    logger.warning('Retrying with local clipping.')

            print('----------------------------')
            # print('diversity:', diversity(smile_list)) # It would take very long time
            logger.info("The %dth validity:%.4f" % (n + 1, valid / ((n + 1) * batch_size)))
            logger.info("The %dth stable:%.4f" % (n + 1, stable / ((n + 1) * batch_size)))
            logger.info("The %dth Uniq:%.4f" % (n + 1, len(set(smile_list)) / ((n + 1) * batch_size)))
            print('----------------------------')


    validity_dict = analyze_stability_for_molecules(position_list, atom_type_list, dataset_info)
    print(validity_dict)
    print("Final validity:", valid / num_samples)
    print("Final stable:", stable / num_samples)
    print("Final unique:", len(set(smile_list)) / num_samples)
    print(len(set(smile_list)) / valid)
    logger.info("Final validity:%.4f" % (valid / num_samples))
    logger.info("Final stable:%.4f" % (stable / num_samples))
    logger.info("Final unique:%.4f" % (len(set(smile_list)) / num_samples))

    uniq = list(set(smile_list))

    if args.dataset == 'qm9':
        dataset_smile_list = retrieve_qm9_smiles()
    else:
        dataset_smile_list = retrieve_geom_smiles()

    novel = []
    for smile in uniq:
        if smile not in dataset_smile_list:
            novel.append(smile)

    print(len(novel))
    novelty = len(novel) / len(uniq)
    logger.info("Final novelty:%.4f" % novelty)
    logger.info("Final novel:%.4f" % len(novel))

    save = False
    if num_samples == 10000:
        save = True

    if save:
        save_path = os.path.join(output_dir, 'samples_all.pkl')
        logger.info('Saving samples to: %s' % save_path)
        save_smile_path = os.path.join(output_dir, 'samples_smile.pkl')

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        with open(save_smile_path, 'wb') as f:
            pickle.dump(smile_list, f)
