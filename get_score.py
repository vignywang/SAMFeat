import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io import loadmat
from tqdm import tqdm_notebook as tqdm
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

methods = ['SAMFeat_V3_3_cat_PAM_B14']
names = methods
colors = ['red']
linestyles = ['-']

methods = ['samfeat', 'mtldesc','hesaff', 'hesaffnet', 'delf-new', 'superpoint', 'd2-net', 'd2-net-trained']
names = ['SAMFeat', 'MTLDesc','Hes. Aff. + Root-SIFT', 'HAN + HN++', 'DELF', 'SuperPoint', 'D2-Net', 'D2-Net Trained']
colors = ['black', 'red','black','blue','green','purple', 'orange', 'brown', 'red', 'green', 'brown', 'purple', 'green', 'purple']
linestyles = ['-','-','-', '-', '-', '--', '--', '--', '--', '--', '--', '--', '--', '--', '--']

top_k = None
n_i = 52
n_v = 56
dataset_path = '/media/data/wjq/SAMFeat/evaluation_hpatch/output/SAMFeat_V3_3_cat_PAM_B14'
HPatches_path = '/media/data/wjq/HPatches'
output_path = '/media/data/wjq/SAMFeat/vis_pdf'
output_pdf_name = 'compare.pdf'
lim = [1, 15]
rng = np.arange(lim[0], lim[1] + 1)


def match_p(p, p_th):  # p N*M
    score, index = torch.topk(p, k=1, dim=-1)
    _, index2 = torch.topk(p, k=1, dim=-2)
    mask_th, index, index2 = score[:, 0] > p_th, index[:, 0], index2.squeeze(0)
    mask_mc = index2[index] == torch.arange(len(p)).cuda()
    mask = mask_th & mask_mc
    index1, index2 = torch.nonzero(mask).squeeze(1), index[mask]
    return index1, index2

def LG_process_list(input_list):
    output_list = []
    for index, value in enumerate(input_list):
        if value != -1:
            output_list.append([index, value.item()])
    return np.array(output_list)

def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def benchmark_features(read_feats, SG=True):
    seq_names = sorted(os.listdir(dataset_path))

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a, img_shape = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b, img_shape = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])


            matches = mnn_matcher(
               torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device))



            homography = np.loadtxt(os.path.join(HPatches_path, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def summary(stats):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((n_i + n_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (n_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (n_v * 5))
    )


def generate_read_function(method, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(dataset_path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors'], aux['img_shape']
        else:
            assert ('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k:]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]

    return read_function


def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)


def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert ('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k:]
        return keypoints[ids, :], descriptors[ids, :]

if top_k is None:
    cache_dir = 'cache-compare'
else:
    cache_dir = 'cache-top'
if not os.path.isdir(cache_dir):
    os.mkdir(cache_dir)
errors = {}
i = 0
for method in methods:
    output_file = os.path.join(cache_dir, method + '.npy')
    print(names[i])
    i += 1
    if method == 'hesaff':
        read_function = lambda seq_name, im_idx: parse_mat(
            loadmat(os.path.join(dataset_path, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
    else:
        if method == 'delf' or method == 'delf-new':
            read_function = generate_read_function(method, extension='png')
        else:
            read_function = generate_read_function(method)
    if os.path.exists(output_file):
        print('Loading precomputed errors...')
        errors[method] = np.load(output_file, allow_pickle=True)
    else:
        errors[method] = benchmark_features(read_function, False)
        np.save(output_file, errors[method])
    summary(errors[method][-1])

plt_lim = [1, 10]
plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls, linewidth=2, label=name)
print([(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng])
plt.title('Overall')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylabel('MMA')
plt.ylim([0, 1])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.legend()

plt.subplot(1, 3, 2)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=2, label=name)
plt.title('Illumination')
plt.xlabel('threshold [px]')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

plt.subplot(1, 3, 3)
for method, name, color, ls in zip(methods, names, colors, linestyles):
    i_err, v_err, _ = errors[method]
    plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=2, label=name)
plt.title('Viewpoint')
plt.xlim(plt_lim)
plt.xticks(plt_rng)
plt.ylim([0, 1])
plt.gca().axes.set_yticklabels([])
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=20)

if top_k is None:
    plt.savefig(os.path.join(output_path, output_pdf_name), bbox_inches='tight', dpi=300)
else:
    plt.savefig(os.path.join(output_path,'hseq-top.pdf'), bbox_inches='tight', dpi=300)