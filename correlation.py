import os
import pickle
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
matplotlib.use('Agg')
sns.set()

def process():
    # test_idx = pickle.load(open('./project/DisenSemi/Our/models/test_idx.pkl', 'rb'))
    # embs = []
    # for i in range(6,7):
    #     # emb_test = pickle.load(open('./project/DisenSemi/Our/models/outs_u_test_'+str(i)+'.pkl', 'rb'))
    #     # emb_val = pickle.load(open('./project/DisenSemi/Our/models/outs_u_val_'+str(i)+'.pkl', 'rb'))
    #     emb_test = pickle.load(open('./project/DisenSemi/Our/models/outs_s_test_'+str(i)+'.pkl', 'rb'))
    #     emb_val = pickle.load(open('./project/DisenSemi/Our/models/outs_s_val_'+str(i)+'.pkl', 'rb'))
    #     # print(len(list(emb_test.values())[:3]))
    #     # print(len(emb_test))
    #     # for emb_ in emb_test.values():
    #     #     print(emb_)
    #     #     print(emb_[:5,:])
    #     emb_test = [torch.cat(emb_,dim=-1)[:,:] for emb_ in emb_test.values()]
    #     emb_val = [torch.cat(emb_,dim=-1)[:,:] for emb_ in emb_val.values()]
    #     # emb = emb_test + emb_val
    #     # emb = emb_val
    #     emb = emb_test
    #     emb = torch.cat(emb, dim=0).cpu().numpy()
    #     print(emb.shape)
    #     embs.append(emb)
    # embs = np.concatenate(embs, axis=0)
    # print(embs.shape)
    # print(embs)
    # choose = random.choice(test_idx).item()
    # emb = embs[choose].numpy()
    name = 'cora'
    seed = 1
    model_output = np.load('./z_predicty_y'+'_'+name+'_'+str(seed)+'.npz')
    node_hid = model_output['arr_0'] # Shape: [n_nodes, hid_dim]
    embs = node_hid
    correlation = np.zeros((embs.shape[1], embs.shape[1]))
    for i in range(embs.shape[1]):
        print(i)
        for j in range(embs.shape[1]):
            cof = scipy.stats.pearsonr(embs[:, i], embs[:, j])[0]
            correlation[i][j] = cof

    print('plot')
    plot_corr(np.abs(correlation))
    # plot_corr(correlation)

def plot_corr(data):
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    # matplotlib.rcParams['axes.labelsize'] = 16
    # matplotlib.rcParams['xtick.labelsize'] = 16
    # matplotlib.rcParams['ytick.labelsize'] = 16
    ax = sns.heatmap(data, vmin=0.0, vmax=1.0, cmap='YlGnBu')
    ax.add_patch(
     patches.Rectangle(
         (0, 0),
         300.0,
         300.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (300, 300),
         300.0,
         300.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (600, 600),
         300.0,
         300.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (900, 900),
         300.0,
         300.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )
    ax.add_patch(
     patches.Rectangle(
         (1200, 1200),
         300.0,
         300.0,
         edgecolor='red',
         fill=False,
         lw=2,
         linestyle='--'
     ) )

    plt.subplots_adjust(top=0.975, right=0.99, left=0.09, bottom=0.09)
    plt.savefig('fig6_c.pdf')
    plt.show()
    plt.close()

process()