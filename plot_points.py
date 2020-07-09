#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
    os.chdir(os.path.join(os.getcwd(), '../../../../../home/xtwang'))
    print(os.getcwd())
except:
    pass

#%%
# first version
import matplotlib.pyplot as plt
interp_1_rmse = [
    1,2,3,4
]
interp_1_ps = [
    30.51,
    30.505,  #30153 TODO wait for 450k
    30.55,  # 30162; 300k
    30.57,  # 3017; 450k
]

interp_2_ps = [
    30.347,  # 30184_300k
    30.403,  # 30192 600k 30.403
    30.460,  # 30202 300k 30.445; 450k
    30.495,  # 30213 150k TODO aim for 300k 30.49
]

fig, ax = plt.subplots()
plt.title('test different interpolation', fontsize=16, color='r')
ax.plot(interp_1_rmse, interp_1_ps, 'o-', color='#ff7f0e', label='test_79_130k_RRDB')
ax.plot(interp_1_rmse, interp_2_ps, 'o-', color='#1f77b4', label='test_79_130k_RRDB')

# legend = ax.legend(loc='upper right', shadow=False)
ax.set_ylabel('PSNR')
ax.set_xlabel('Offset Diversity')
plt.axis([0.5, 4.5, 30.3, 30.65])
# plt.axvline(x=11.5, color='k', ls='--', linewidth=1.0)
# plt.axvline(x=12.5, color='k', ls='--', linewidth=1.0)
# plt.axvline(x=16, color='k', ls='--', linewidth=1.0)
# # plt.axvline(x=12.5, color='k', ls='--')

# # Hide the right and top spines
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# # Only show ticks on the left and bottom spines
# ax.yaxis.set_ticks_position('left')
# ax.xaxis.set_ticks_position('bottom')
# plt.yticks([2, 3, 4, 5, 5.5])

fig = plt.gcf()
# fig.set_size_inches(18.5, 10.5)
fig.savefig('gate.png', dpi=300)

plt.show()
