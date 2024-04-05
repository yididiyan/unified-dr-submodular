import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
import argparse

plt.rcParams['font.size'] = 18

def total_regret(offline_values, online_values):
    # change cum average to cum sum
    offline_cum_sum = np.array(offline_values) * np.arange(1, len(offline_values) + 1)
    online_cum_sum = np.array(online_values) * np.arange(1, len(online_values) + 1)

    # compute log of the total regret at each time step
    log_total_regret = np.log(offline_cum_sum - online_cum_sum)

    return log_total_regret
def log_total_regret(offline_values, online_values):
    """
    Both inputs are running averages across time
    :param offline_values:
    :param online_values:
    :return:
    """
    return np.log(total_regret(offline_values, online_values))

def load_data(output_dir, grad_noise=0.1, Nms=None):

    def pull_files(prefix):
        files_ = [glob.glob(
            f'{output_dir}/{prefix}/n_{n}__m_{m}*{ "" or grad_noise }*.pkl'
        ) for n, m in Nms]
        print(files_)
        files_ = [ f[0] for f in files_  if f]

        return files_

    ## load the pickles
    gmfw_files = {
        '0': [],
        '1/8': [],
        '1/4': [],
        '3/8': [],
        '1/2': [],
        '3/4': [],
    }
    gmfw = {
        '0': [],
        '1/8': [],
        '1/4': [],
        '3/8': [],
        '1/2': [],
        '3/4': [],
    }

    meta_files = {'0': [], '1/8': [], '1/4': [], '1/2': [], '3/4': [], '1': [], '3/2': []}
    meta = {'0': [], '1/8': [], '1/4': [], '1/2': [], '3/4': [], '1': [],  '3/2': []}



    (gmfw_files['0'],
     gmfw_files['1/8'],
     gmfw_files['1/4'],
     gmfw_files['3/8'],
     gmfw_files['1/2'],
     gmfw_files['3/4'],
     sbfw_files, mono_files,
     # meta_files['1/2'],
     meta_files['0'],
     meta_files['3/4'],
     meta_files['1'],
     meta_files['3/2']
     ) = [
            pull_files('gmfw/beta_0'),
            pull_files('gmfw/beta_0.125'),
            pull_files('gmfw/beta_0.25'),
            pull_files('gmfw/beta_0.375'),
            pull_files('gmfw/beta_0.5'),
            pull_files('gmfw/beta_0.75'),
            pull_files('sbfw'),
            pull_files('mono'),
            # pull_files('meta/beta_0.5')),
            pull_files('meta/beta_0'),
            pull_files('meta/beta_0.75'),
            pull_files('meta/beta_1'),
            pull_files('meta/beta_1.5')
        ]

    odc_files = pull_files('odc')

    meta_files['1/8'], meta_files['1/4'] = [
        pull_files('meta_beta/0.125'),
        pull_files('meta_beta/0.25'),
    ]




    sbfw, mono, odc  = [], [], []

    # import ipdb;ipdb.set_trace()
    for key in gmfw_files.keys():
        for f in gmfw_files[key]:
            with open(f, 'rb') as f_:
                gmfw[key].append(pickle.load(f_))


    for f in sbfw_files:
        with open(f, 'rb') as f_:
            sbfw.append(pickle.load(f_))

    for f in mono_files:
        with open(f, 'rb') as f_:
            mono.append(pickle.load(f_))

    for f in odc_files:
        with open(f, 'rb') as f_:
            odc.append(pickle.load(f_))

    for key in meta_files.keys():

        for f in meta_files[key]:
            with open(f, 'rb') as f_:
                meta[key].append(pickle.load(f_))


    return meta, mono, sbfw, gmfw, odc




def load_averaged_data(output_dir, T, seeds, grad_noise=0.1, Nms=[]):
    """

    :param output_dir: the prefix of the output directory
    :param seeds: the varying seeds used
    :param grad_noise: gradient noise
    :return:
    """

    meta_all, mono_all, sbfw_all, gmfw_all, odc_all = [], [], [], [], []

    for seed in seeds:
        meta, mono, sbfw, gmfw, odc = load_data(f'{output_dir}_{seed}/{T}', grad_noise=grad_noise, Nms=Nms)

        meta_all.append(meta)
        mono_all.append(mono)
        sbfw_all.append(sbfw)
        gmfw_all.append(gmfw)
        odc_all.append(odc)

    # import ipdb; ipdb.set_trace()
    return meta_all, mono_all, sbfw_all, gmfw_all, odc_all

if __name__ == '__main__':
    import os

    import numpy as np
    import argparse

    import matplotlib.pyplot as plt


    DEFAULT_SEEDS = list(range(1, 11))
    DEFAULT_Ts = [20, 40, 80, 160, 320, 500]

    parser = argparse.ArgumentParser('argument ')
    parser.add_argument('--output-dir', action='store', type=str, default='./')
    parser.add_argument('--Ts', action='store', nargs='+', type=int, default=DEFAULT_Ts)
    parser.add_argument('--seeds', action='store', nargs='+', type=int, default=DEFAULT_SEEDS)

    args = parser.parse_args()

    legend_font_size = 12
    plt.rcParams['font.size'] = 18

    root_dir = args.output_dir

    plots_dir = f'{root_dir}/plots'  # create directory for plots

    os.makedirs(plots_dir, exist_ok=True)

    T = args.Ts

    seeds = args.seeds
    grad_noise = 0.1  # 0.1

    color_choices = {
        '0': 'tab:red',
        '1/8': 'tab:orange',
        '1/4': "tab:green",
        '3/8': 'cyan',
        '1/2': 'black',
        '3/4': 'tab:blue',
        '3/2': 'black',
        '1': 'tab:red'
    }

    n_m_pairs = [(25, 15), (40, 20), (50, 50)]

    # Instantaneous regret plots



    def plot(x, y, ci, label=None, **kwargs):
        plt.plot(x, y, label=label, **kwargs)
        plt.fill_between(x, (y - ci), (y + ci), color=kwargs.get('color') or 'b', alpha=.1)


    n_m_pairs = [(25, 15), (40, 20), (50, 50)]

    # T for instantaneous plots
    T_instant = T[-1]
    seeds = seeds

    meta_t, mono_t, sbfw_t, gmfw_t, odc_t = load_averaged_data(
        output_dir=f'{root_dir}/output',
        T=T_instant,
        seeds=seeds,
        grad_noise=grad_noise, Nms=n_m_pairs)

    for s in range(len(seeds)):
        for i in range(3):
            sbfw_t[s][i] = (sbfw_t[s][i]['offline_values'] - sbfw_t[s][i]['online_values'])

    for t in range(len(seeds)):
        gmfw_keys = gmfw_t[t].keys()

        for i in range(3):
            for key in gmfw_keys:
                if gmfw_t[t][key]:
                    gmfw_t[t][key][i] = (gmfw_t[t][key][i]['offline_values'] - gmfw_t[t][key][i]['online_values'])


    for j in range(len(n_m_pairs)):  # experiments

        labels = []


        #
        plot(list(range(T_instant)),
             np.array([sbfw_t[i][j] for i in range(len(seeds))]).mean(0),
             np.array([sbfw_t[i][j] for i in range(len(seeds))]).std(0),
             label='SBFW', linestyle='-', color='tab:blue'
             )
        labels.append('SBFW')

        for key in gmfw_t[0].keys():
            if key in ['0', '1/4',  '1/2']:
                plot(list(range(T_instant)),
                     np.array([gmfw_t[i][key][j] for i in range(len(seeds))]).mean(0),
                     np.array([gmfw_t[i][key][j] for i in range(len(seeds))]).std(0),
                     label=fr'GMFW($\beta={key}$)', color=color_choices[key]
                )
                labels.append(f'GMFW(beta={key})')





        xticks = list(range(0, T_instant + 1, 250))

        plt.xticks(xticks, xticks)

        plt.title(fr'$n={n_m_pairs[j][0]}, m={n_m_pairs[j][1]}$')
        if j == 0:
            plt.ylabel(r'$\frac{1}{e}$-regret/$t$')
        plt.xlabel(fr'Iteration index $t$')

        plt.legend(prop={'size': legend_font_size}, loc="upper left", mode="expand", ncol=3)
        plt.savefig(f'{plots_dir}/fig_n_{n_m_pairs[j][0]}__m_{n_m_pairs[j][1]}_.pdf', dpi=500, bbox_inches='tight')

        plt.figure()

    # Cumulative regret plots
    for exp_idx in range(len(n_m_pairs)):
        print(f'Experiment # {exp_idx}')
        plot_data = []
        err_data = []

        labels = []
        colors = []
        linestyles = []

        for idx, t in enumerate(T):
            plot_data_t = []
            err_data_t = []
            meta_t, mono_t, sbfw_t, gmfw_t, odc_t = load_averaged_data(
                output_dir=f'{root_dir}/output',
                T=t,
                seeds=seeds,
                grad_noise=grad_noise, Nms=n_m_pairs)


            plot_data_t.append(t * np.mean(
                [item[exp_idx]['offline_values'][-1] - item[exp_idx]['online_values'][-1] for item in sbfw_t]))
            err_data_t.append(t * np.std(
                [item[exp_idx]['offline_values'][-1] - item[exp_idx]['online_values'][-1] for item in sbfw_t]))
            if idx == 0:
                labels.append('SBFW')
                colors.append('tab:blue')
                linestyles.append('-')







            beta_vals = gmfw_t[0].keys()
            for beta in beta_vals:
                if beta in ['0', '1/2', '1/4'] and gmfw_t[0][beta]:
                    plot_data_t.append(t * np.mean(
                        [item[beta][exp_idx]['offline_values'][-1] - item[beta][exp_idx]['online_values'][-1] for item
                         in gmfw_t]))
                    err_data_t.append(t * np.std(
                        [item[beta][exp_idx]['offline_values'][-1] - item[beta][exp_idx]['online_values'][-1] for item
                         in gmfw_t]))
                    if idx == 0:
                        labels.append(fr'GMFW($\beta={beta}$)')
                        colors.append(color_choices[beta])
                        linestyles.append('-')






            plot_data.append(plot_data_t)
            err_data.append(err_data_t)

        plot_data = np.array(plot_data)
        err_data = np.array(err_data)


        assert plot_data.shape == err_data.shape
        for idx, label in enumerate(labels):
            plt.errorbar(T, plot_data[:, idx], err_data[:, idx], label=label, color=colors[idx],
                         linestyle=linestyles[idx],
                         lw=3, marker='D', markersize=8, capsize=5)

        plt.yscale('log')
        plt.xscale('log')

        plt.legend(prop={'size': 10}, loc="upper left", mode="expand", ncol=3)
        plt.title(fr'$n={n_m_pairs[exp_idx][0]}, m={n_m_pairs[exp_idx][1]}$')

        plt.xlabel(fr'Horizon $T$')
        if exp_idx == 0:
            plt.ylabel(r'Cumulative $\frac{1}{e}$-regret')

        plot_data_max, plot_data_min = plot_data[plot_data > 0].max(), plot_data[plot_data > 0].min()

        plt.ylim(top=plot_data_max + 100, bottom=plot_data_min - 500)

        # # Regret bound guides
        x = np.linspace(T[0], T[-1])

        ys = [i / 10 * x ** (1 / 2) for i in
              np.exp(np.linspace(np.log(plot_data_min), np.log(plot_data_max) + 0.75, 7))]

        xticks = T
        plt.xticks(xticks, xticks)
        for y in ys:
            plt.plot(x, y, '-', color='gray', lw=2, alpha=0.25)

        plt.savefig(f'{plots_dir}/fig_total_regret_n_{n_m_pairs[exp_idx][0]}__m_{n_m_pairs[exp_idx][1]}.pdf', dpi=500,
                    bbox_inches='tight')
        plt.savefig(
            f'{plots_dir}/fig_total_regret_n_{n_m_pairs[exp_idx][0]}__m_{n_m_pairs[exp_idx][1]}.png',
            dpi=500,
            bbox_inches='tight')
        plt.figure()



    # Calculating running times
    meta_t, mono_t, sbfw_t, gmfw_t, odc_t = load_averaged_data(
        output_dir=f'{root_dir}/output',
        T=T_instant,
        seeds=seeds,
        grad_noise=grad_noise, Nms=n_m_pairs)

    for exp_idx in range(len(n_m_pairs)):
        plot_data = []
        err_data = []

        labels = []
        colors = []
        linestyles = []

        plot_data_t = []
        err_data_t = []






        # SBFW
        plot_data_t.append(np.mean([item[exp_idx]['total_time'] for item in sbfw_t]))
        err_data_t.append(np.std([item[exp_idx]['total_time'] for item in sbfw_t]))
        labels.append('SBFW')

        # GMFW
        beta_vals = gmfw_t[0].keys()
        for beta in beta_vals:

            if beta in ['0', '1/4', '1/2'] and gmfw_t[0][beta]:
                plot_data_t.append(np.mean([item[beta][exp_idx]['total_time'] for item in gmfw_t]))
                err_data_t.append(np.std([item[beta][exp_idx]['total_time'] for item in gmfw_t]))
                labels.append(fr'GMFW($\beta={beta}$)')

        plot_data.append(plot_data_t)
        err_data.append(err_data_t)

        plot_data = np.array(plot_data)
        err_data = np.array(err_data)

        assert plot_data.shape == err_data.shape

        print(f'Exp {exp_idx}')
        for i, l in enumerate(labels):
            print(l, '%.2f' % plot_data[0][i], '%.2f' % err_data[0][i])