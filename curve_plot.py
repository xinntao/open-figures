#%%
#########################################
import utils
import mycolors
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
smooth_weight = 0.9

#%%
#########################################
# comparison group 1 - official DnCNN and its variants
DnCNN_data = utils.read_data_from_txt_1p('./logs/901_DnCNN_sigma25.txt', r'PSNR = (\d+\.\d+)')
MSE_data = utils.read_data_from_txt_1p('./logs/902_MSEloss.txt', r'PSNR = (\d+\.\d+)')
MSE05_data = utils.read_data_from_txt_1p('./logs/903_MSEloss_05.txt', r'PSNR = (\d+\.\d+)')
BNmomentum_data = utils.read_data_from_txt_1p('./logs/904_BN_momentadefault.txt',
                                              r'PSNR = (\d+\.\d+)')
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
plt.plot(
    list(range(len(MSE_data))), MSE_data, color=mycolors.C1, linewidth=1.5, label='902_MSEloss')
plt.plot(
    list(range(len(MSE05_data))),
    MSE05_data,
    color=mycolors.C2,
    linewidth=1.5,
    label='903_MSEloss_05')
plt.plot(
    list(range(len(BNmomentum_data))),
    BNmomentum_data,
    color=mycolors.C3,
    linewidth=1.5,
    label='904_BN_momentadefault')

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