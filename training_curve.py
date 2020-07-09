#%%
import os
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors as colors
from matplotlib.pyplot import figure
mpl.style.use('default')
import seaborn
seaborn.set(style='whitegrid')
seaborn.set_context('paper')
import re

import utils
import colors
smooth_weight = 0.9

#%%
log_path = '/home/xtwang/remote/190/home/xtwang/Projects/2019/EDVR/tb_logger/001_EDVRwoTSA_scratch_lr4e-4_600k_REDS_LrCAR4S/events.out.tfevents.1560357929.cdc-190.13885.0'
tag = 'l_pix'
steps, values = utils.read_data_from_tensorboard(log_path, tag)

values = [v if v < 30000 else 30000 for v in values]
values_sml = utils.smooth_data(values, 0.9)
values_sm = utils.smooth_data(values, 0.999)

#%%
plt.figure(1)
plt.subplot(111)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('(Zoom in) Training curve after smoothing', fontsize=16, color='k')

plt.plot(steps, values_sml, color=colors.C1, alpha=0.2, linewidth=1.5, label='FWG')
plt.plot(steps, values_sm, color=colors.C1, linewidth=1.5, label='FWG')

# legend = plt.legend(loc='lower right', shadow=False)
ax = plt.gca()
ax.set_ylim([18400, 19400])
# ax.set_xlim([0, 150])
labels = ax.get_xticks().tolist()

ax.set_ylabel('Loss pixel')
ax.set_xlabel('Iteration')
fig = plt.gcf()
fig.set_size_inches(14, 5)
# fig.savefig('training_curve.png', dpi=800)
plt.show()

#%%
# comparison group 1 - official DnCNN and its variants
data_phase1 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_2172_190523-032247.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase2 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_2214_190523-032529.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase3 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_2227_190523-040315.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase4 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_2238_190523-033400.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase5 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_2258_190523-041356.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')

l1 = []
for k in data_phase1[1]:
    l1.append(k + 0.302)  #0.362
l2 = []
for k in data_phase2[1]:
    l2.append(k + 0.201)  # 0.241
l3 = []
for k in data_phase3[1]:
    l3.append(k + 0.011)
all_data = []
all_data.extend(l1)
all_data.extend(l2)
all_data.extend(l3)
all_data.extend(data_phase4[1])
all_data.extend(data_phase5[1])
# comparison group
data_phase1 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_1067_190523-034505.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase2 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_1080_190523-081454.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase3 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_1083_190523-081538.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase4 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_1092_190523-081622.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')
data_phase5 = utils.read_data_from_txt_2p(
    '/mnt/3TB/Projects/plot_figures/universal/logs2/val_1095_190523-034730.log',
    r'.+?Iter\s*(\d+)\s*- Total Average PSNR:\s*(\d+\.\d+)\s*dB.+?')

all_data_dcn = []
all_data_dcn.extend(data_phase1[1])
all_data_dcn.extend(data_phase2[1])
all_data_dcn.extend(data_phase3[1])
all_data_dcn.extend(data_phase4[1])
all_data_dcn.extend(data_phase5[1])

plt.figure(1)
plt.subplot(111)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('Title', fontsize=16, color='k')

tmp = list(range(len(all_data)))
l = []
for k in tmp:
    l.append(k * 0.75)
plt.plot(l, all_data, color=mycolors.C1, linewidth=1.5, label='FWG')
plt.plot(l, all_data_dcn, color=mycolors.C0, linewidth=1.5, label='DCN')

# legend = plt.legend(loc='lower right', shadow=False)
ax = plt.gca()
ax.set_ylim([29.6, 30.55])
ax.set_xlim([0, 150])
labels = ax.get_xticks().tolist()

ax.set_ylabel('PSNR')
ax.set_xlabel('Iter')
fig = plt.gcf()
fig.set_size_inches(6, 5)
fig.savefig('training_curve.png', dpi=800)
plt.show()
'''
#%%
#########################################
# comparison group 2 - official DnCNN and noBN variants
DnCNN_data = utils.read_data_from_txt_1p('./logs/901_DnCNN_sigma25.txt', r'PSNR = (\d+\.\d+)')
noBN_data = utils.read_data_from_txt_1p('./logs/905_noBN.txt', r'PSNR = (\d+\.\d+)')
noBN2e4_data = utils.read_data_from_txt_1p('./logs/906_noBN_2e-4.txt', r'PSNR = (\d+\.\d+)')
noBN5e4_data = utils.read_data_from_txt_1p('./logs/907_noBN5e4.txt', r'PSNR = (\d+\.\d+)')

plt.figure(1)
plt.subplot(111)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('Title', fontsize=16, color='k')
plt.plot(
    list(range(len(DnCNN_data))),
    DnCNN_data,
    color=mycolors.C0,
    linewidth=1.5,
    label='901_DnCNN_sigma25')
plt.plot(list(range(len(noBN_data))), noBN_data, color=mycolors.C1, linewidth=1.5, label='905_noBN')
plt.plot(
    list(range(len(noBN2e4_data))),
    noBN2e4_data,
    color=mycolors.C2,
    linewidth=1.5,
    label='906_noBN_2e-4')
plt.plot(
    list(range(len(noBN5e4_data))),
    noBN5e4_data,
    color=mycolors.C3,
    linewidth=1.5,
    label='907_noBN5e4')

legend = plt.legend(loc='lower right', shadow=False)
ax = plt.gca()
ax.set_ylim([29., 29.25])
labels = ax.get_xticks().tolist()

ax.set_ylabel('PSNR')
ax.set_xlabel('Epoch')
fig = plt.gcf()
# fig.set_size_inches(9, 5)
# fig.savefig('deeper_net_coarse.pdf', dpi=800)
plt.show()

#%%
#########################################
# comparison group 3 - my implementation of DnCNN and its
D026_epoch, D026_data = utils.read_data_from_txt_2p(
    './logs/026_DnCNNv2_T400_BNini_loss_1e-3.txt',
    r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)',
    step_one=True)
D030_epoch, D030_data = utils.read_data_from_txt_2p(
    './logs/030_DnCNNv2_T400_BNini_loss_1e-3_share_N25.txt',
    r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)',
    step_one=True)
D026_N35_epoch, D026_N35_data = utils.read_data_from_txt_2p(
    './logs/026_DnCNNv2_T400_BNini_loss_1e-3_N35.txt',
    r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)',
    step_one=True)
D030_N35_epoch, D030_N35_data = utils.read_data_from_txt_2p(
    './logs/030_DnCNNv2_T400_BNini_loss_1e-3_share_N35.txt',
    r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)',
    step_one=True)
D028_epoch, D028_data = utils.read_data_from_txt_2p(
    './logs/028_DnCNNv2_blind_mGPU4.txt', r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)', step_one=True)
D029_epoch, D029_data = utils.read_data_from_txt_2p(
    './logs/029_DnCNNv2_blind_mGPU4_lr1e-3.txt',
    r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)',
    step_one=True)

plt.figure(1)
plt.subplot(111)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('Title', fontsize=16, color='k')

# plt.plot(
#     D026_epoch,
#     D026_data,
#     color=mycolors.C3,
#     linewidth=1.5,
#     label='026_DnCNNv2_T400_BNini_loss_1e-3')
# plt.plot(
#     D030_epoch,
#     D030_data,
#     color=mycolors.C0,
#     linewidth=1.5,
#     label='030_DnCNNv2_T400_BNini_loss_1e-3_share')
# plt.plot(
#     D026_N35_epoch,
#     D026_N35_data,
#     color=mycolors.C3,
#     linewidth=1.5,
#     label='026_DnCNNv2_T400_BNini_loss_1e-3_N35')
# plt.plot(
#     D030_N35_epoch,
#     D030_N35_data,
#     color=mycolors.C0,
#     linewidth=1.5,
#     label='030_DnCNNv2_T400_BNini_loss_1e-3_share_N35')
plt.plot(D028_epoch, D028_data, color=mycolors.C0, linewidth=1.5, label='028_DnCNNv2_blind_mGPU4')
plt.plot(
    D029_epoch, D029_data, color=mycolors.C1, linewidth=1.5, label='029_DnCNNv2_blind_mGPU4_lr1e-3')

legend = plt.legend(loc='lower right', shadow=False)
ax = plt.gca()
# ax.set_ylim([28., 29.5])
labels = ax.get_xticks().tolist()

ax.set_ylabel('PSNR')
ax.set_xlabel('Epoch')
fig = plt.gcf()
# fig.set_size_inches(9, 5)
fig.savefig('test.png', dpi=800)
plt.show()


#%%
#########################################
D0171_epoch, D0171_data = utils.read_data_from_txt_2p(
    './logs/017_log_noclamp_noround.txt', r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)', step_one=True)
D0172_epoch, D0172_data = utils.read_data_from_txt_2p(
    './logs/017_log_clamp_noround.txt', r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)', step_one=True)
D0173_epoch, D0173_data = utils.read_data_from_txt_2p(
    './logs/017_log_clamp_round.txt', r'Epoch: (\d+).+?PSNR:\s*(\d+\.\d+)', step_one=True)

plt.figure(1)
plt.subplot(111)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.title('Title', fontsize=16, color='k')

plt.plot(D0171_epoch, D0171_data, color=mycolors.C0, linewidth=1.5, label='017_log_noclamp_noround')
plt.plot(D0172_epoch, D0172_data, color=mycolors.C1, linewidth=1.5, label='017_log_clamp_noround')
# plt.plot(D0173_epoch, D0173_data, color=mycolors.C2, linewidth=1.5, label='017_log_clamp_round')

legend = plt.legend(loc='lower right', shadow=False)
ax = plt.gca()
ax.set_ylim([28., 29.25])
labels = ax.get_xticks().tolist()

ax.set_ylabel('PSNR')
ax.set_xlabel('Epoch')
fig = plt.gcf()
# fig.set_size_inches(9, 5)
fig.savefig('test.png', dpi=800)
plt.show()

'''