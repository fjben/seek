


import os

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

 
# # making a simple plot
# x =[1, 2, 3, 4, 5]
# y =[1, 1, 1, 1, 1]

# # creating error
# x_error = [0.2, 0.4, 0.5, 0.2]
 
# # plotting graph
# # plt.plot(x, y)
# plt.errorbar(x, y,
#              xerr = x_error, 
#              fmt ='o',
#              orientation='horizontal')
 
  


# # creating the dataset
# data = {'C':20, 'C++':15, 'Java':30, 
#         'Python':35}
# courses = list(data.keys())
# values = list(data.values())
  
# fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot
# plt.barh(courses, values, color ='maroon')
 
# plt.xlabel("Courses offered")
# plt.ylabel("No. of students enrolled")
# plt.title("Students enrolled in different courses")
# # plt.show()




# # Enter raw data
# aluminum = np.array([6.4e-5 , 3.01e-5 , 2.36e-5, 3.0e-5, 7.0e-5, 4.5e-5, 3.8e-5, 4.2e-5, 2.62e-5, 3.6e-5])
# copper = np.array([4.5e-5 , 1.97e-5 , 1.6e-5, 1.97e-5, 4.0e-5, 2.4e-5, 1.9e-5, 2.41e-5 , 1.85e-5, 3.3e-5 ])
# steel = np.array([3.3e-5 , 1.2e-5 , 0.9e-5, 1.2e-5, 1.3e-5, 1.6e-5, 1.4e-5, 1.58e-5, 1.32e-5 , 2.1e-5])

# # Calculate the average
# aluminum_mean = np.mean(aluminum)
# copper_mean = np.mean(copper)
# steel_mean = np.mean(steel)

# # Calculate the standard deviation
# aluminum_std = np.std(aluminum)
# copper_std = np.std(copper)
# steel_std = np.std(steel)

# # Create lists for the plot
# materials = ['Aluminum', 'Copper', 'Steel']
# x_pos = np.arange(len(materials))
# CTEs = [aluminum_mean, copper_mean, steel_mean]
# error = [aluminum_std, copper_std, steel_std]

# datasets = ['AIFB', 'MUTAG', 'AM', 'MDGENRE']
# means = [2.79, 2.771, 2.289, 3.933]
# stds = [1.708, 1.614, 1.568, 1.445]

# # # Build the plot
# # fig, ax = plt.subplots()
# # ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# # ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
# # ax.set_xticks(x_pos)
# # ax.set_xticklabels(materials)
# # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# # ax.yaxis.grid(True)

# # Build the plot
# fig, ax = plt.subplots()
# # ax.barh(x_pos, CTEs, xerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.errorbar(means, datasets, xerr=stds, fmt='o')
# ax.set_xlabel('size of explanation')
# ax.set_yticks(datasets)
# # ax.set_yticklabels(datasets)
# # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.xaxis.grid(True)
# ax.set_ylabel('dataset')
# ax.set_xlim(0, 6)

# # Save the figure and show
# plt.tight_layout()
# plt.savefig('bar_plot_with_error_bars.png')
# plt.show()

# plt.savefig('test_len_explanation.png')


# # plt.cm.get_cmap('Paired')
_Paired_data = (
    (0.65098039215686276, 0.80784313725490198, 0.8901960784313725 ),
    (0.12156862745098039, 0.47058823529411764, 0.70588235294117652),
    (0.69803921568627447, 0.87450980392156863, 0.54117647058823526),
    (0.2,                 0.62745098039215685, 0.17254901960784313),
    (0.98431372549019602, 0.60392156862745094, 0.6                ),
    (0.8901960784313725,  0.10196078431372549, 0.10980392156862745),
    (0.99215686274509807, 0.74901960784313726, 0.43529411764705883),
    (1.0,                 0.49803921568627452, 0.0                ),
    (0.792156862745098,   0.69803921568627447, 0.83921568627450982),
    (0.41568627450980394, 0.23921568627450981, 0.60392156862745094),
    (1.0,                 1.0,                 0.6                ),
    (0.69411764705882351, 0.34901960784313724, 0.15686274509803921),
    )

main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_rf_local_final'

explanation_limit = 'class_change'
# explanation_limit = 'threshold'

explan_type = 'necessary'
# explan_type = 'sufficient'

results_file = f'global_explain_stats_{explanation_limit}.csv'

datasets = ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']

kge_models = ['RDF2Vec', 'ComplEx', 'distMult', 'TransH', 'TransE']

mean_list = []
std_list = []
for kge_model in kge_models:
    for dataset in datasets:
    # for kge_model in kge_models:
        df = pd.read_csv(os.path.join(main_path, f'{dataset}_{kge_model}', results_file), sep='\t')
        mean_list.append(float(df[f'{explan_type}_len5_facts_size_mean_(std)'].values[0].split(' ')[0]))
        std_list.append(float(df[f'{explan_type}_len5_facts_size_mean_(std)'].values[0].split(' ')[1][1:-1]))


# print(mean_list)
# print(std_list)
# raise




datasets = ['AIFB', 'MUTAG', 'AM', 'MDGENRE']
means = [2.79, 2.771, 2.289, 3.933]
stds = [1.708, 1.614, 1.568, 1.445]

# # Build the plot
# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# ax.set_ylabel('Coefficient of Thermal Expansion ($\degree C^{-1}$)')
# ax.set_xticks(x_pos)
# ax.set_xticklabels(materials)
# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# ax.yaxis.grid(True)

x = np.arange(1, 5)
tick_labels_1 = ['RDF2Vec'] * len(x)
tick_labels_2 = ['ComplEx'] * len(x)
tick_labels_3 = ['distMult'] * len(x)
tick_labels_4 = ['TransH'] * len(x)
tick_labels_5 = ['TransE'] * len(x)
all_x = np.concatenate([x - 0.4, x - 0.25, x -0.1, x + 0.05, x + 0.2])

# Build the plot
if explan_type == 'necessary':
    fig, ax = plt.subplots(figsize=(6.3, 6))
elif explan_type == 'sufficient':
    fig, ax = plt.subplots(figsize=(2, 6))
if explan_type == 'necessary':
    ax.errorbar(y=all_x, x=mean_list, xerr=std_list, fmt='o', color=_Paired_data[1], label='necessary')
    ax.errorbar(y=all_x, x=mean_list, xerr=std_list, fmt='o', color=_Paired_data[7], label='sufficient')
elif explan_type == 'sufficient':
    ax.errorbar(y=all_x, x=mean_list, xerr=std_list, fmt='o', color=_Paired_data[7])
if explan_type == 'necessary':
    ax.set_xlabel('size of explanation', fontsize=15)
elif explan_type == 'sufficient':
    ax.set_xlabel(' ', fontsize=15)
ax.set_yticks(all_x)
if explan_type == 'necessary':
    ax.set_yticklabels(tick_labels_1 + tick_labels_2 + tick_labels_3 + tick_labels_4 + tick_labels_5, fontsize=15)
elif explan_type == 'sufficient':
    ax.set_yticklabels([' ' for i in range(20)], fontsize=15)
ax.xaxis.grid(True)
# ax.set_ylabel('dataset')
if explan_type == 'necessary':
    ax.set_xlim(0, 6)
elif explan_type == 'sufficient':
    ax.set_xlim(0, 2)

ax.legend(bbox_to_anchor=(0, 1), loc='upper left', fontsize=15)

# ax.set_yticks(x - 0.1001, minor=True)
ax.set_yticks([1.04, 2.14, 2.99, 4.23], minor=True)
if explan_type == 'necessary':
    ax.set_yticklabels(['AIFB', 'MUTAG', 'AM', 'MDGENRE'], minor=True)
ax.tick_params(axis='y', labelrotation=90, which='minor', length=0, pad=82, labelsize=15)
ax.tick_params(axis='x', labelsize=15)

# Save the figure and show
plt.tight_layout()
plt.savefig('bar_plot_with_error_bars.png')
plt.show()

plt.savefig(f'len_explanation_{explan_type}_v2.png')





# # # plt.cm.get_cmap('Paired')
# _Paired_data = (
#     (0.65098039215686276, 0.80784313725490198, 0.8901960784313725 ),
#     (0.12156862745098039, 0.47058823529411764, 0.70588235294117652),
#     (0.69803921568627447, 0.87450980392156863, 0.54117647058823526),
#     (0.2,                 0.62745098039215685, 0.17254901960784313),
#     (0.98431372549019602, 0.60392156862745094, 0.6                ),
#     (0.8901960784313725,  0.10196078431372549, 0.10980392156862745),
#     (0.99215686274509807, 0.74901960784313726, 0.43529411764705883),
#     (1.0,                 0.49803921568627452, 0.0                ),
#     (0.792156862745098,   0.69803921568627447, 0.83921568627450982),
#     (0.41568627450980394, 0.23921568627450981, 0.60392156862745094),
#     (1.0,                 1.0,                 0.6                ),
#     (0.69411764705882351, 0.34901960784313724, 0.15686274509803921),
#     )

# import matplotlib.pyplot as plt
# import numpy as np

# # import matplotlib.transforms

# width = 0.25
# x = np.arange(1, 5)

# fig, ax = plt.subplots(figsize=(10, 6))

# # tick_labels_1 = ['1'] * len(x)
# # tick_labels_2 = ['2'] * len(x)
# # tick_labels_3 = ['3'] * len(x)
# # tick_labels_4 = ['4'] * len(x)
# # tick_labels_5 = ['5'] * len(x)
# tick_labels_1 = ['RDF2Vec'] * len(x)
# tick_labels_2 = ['ComplEx'] * len(x)
# tick_labels_3 = ['distMult'] * len(x)
# tick_labels_4 = ['TransH'] * len(x)
# tick_labels_5 = ['TransE'] * len(x)
# shift1_rbc = np.random.uniform(1100, 1200, 4) # rdf2vec in 4 datasets
# shift2_rbc = np.random.uniform(900, 1000, 4) # complex in 4 datasets
# shift3_rbc = np.random.uniform(1000, 1100, 4)
# shift4_rbc = np.random.uniform(900, 1000, 4)
# shift5_rbc = np.random.uniform(1000, 1100, 4)
# shift1_plt = np.random.uniform(600, 700, 4)
# shift2_plt = np.random.uniform(400, 500, 4)
# shift3_plt = np.random.uniform(500, 600, 4)
# shift4_plt = np.random.uniform(400, 500, 4)
# shift5_plt = np.random.uniform(500, 600, 4)
# # print(shift1_rbc, shift2_rbc)
# # print('\n')
# # print(np.concatenate([shift1_rbc, shift2_rbc]))
# # raise
# # print(shift1_rbc)
# # raise
# # shift1_ffp = np.random.uniform(250, 300, 6)
# # shift2_ffp = np.random.uniform(150, 200, 6)
# # shift3_ffp = np.random.uniform(200, 250, 6)
# # all_x = np.concatenate([x - 0.4, x - 0.1, x + 0.2])
# all_x = np.concatenate([x - 0.4, x - 0.25, x -0.1, x + 0.05, x + 0.2])
# # ax.bar(all_x, np.concatenate([shift1_rbc, shift2_rbc, shift3_rbc, shift4_rbc, shift5_rbc]), width * .55,
# #        tick_label=tick_labels_1 + tick_labels_2 + tick_labels_3 + tick_labels_4 + tick_labels_5,
# #        color=_Paired_data[1], label='compound')
# # print(compound_list)
# ax.bar(all_x, mean_list, yerr=std_list, width=width * .55,
#        tick_label=tick_labels_1 + tick_labels_2 + tick_labels_3 + tick_labels_4 + tick_labels_5,
#        color=_Paired_data[1], label='compound')
# # ax.errorbar(all_x, mean_list, yerr=std_list, fmt='o')
# # ax.bar(all_x, np.concatenate([shift1_plt, shift2_plt, shift3_plt, shift4_plt, shift5_plt]),
# #        width * .45, color=_Paired_data[0], label='single')
# # ax.bar(all_x, std_list,
# #        width * .45, color=_Paired_data[0], label='single')
# ax.tick_params(axis='x', labelrotation=90)
# # ax.bar(all_x, np.concatenate([shift1_ffp, shift2_ffp, shift3_ffp]),
# #        width * .5, color='limegreen', label='green')
# ax.margins(x=0.02)
# # ax.legend(title='Data', bbox_to_anchor=(0.99, 1), loc='upper left')
# ax.legend(bbox_to_anchor=(0.83, 1), loc='upper left')
# for spine in ['top', 'right']:
#     ax.spines[spine].set_visible(False)

# ax.set_xticks(x - 0.1001, minor=True)
# # ax.set_xticks(x, minor=True)
# ax.set_xticklabels(['AIFB', 'MUTAG', 'AM', 'MDGENRE'], minor=True)
# # ax.tick_params(axis='x', which='minor', length=0, pad=18)
# ax.tick_params(axis='x', which='minor', length=0, pad=65)
# ax.set_xlabel('dataset - kge model')
# ax.set_ylabel('size of explanations')


# # # Build the plot
# # fig, ax = plt.subplots()
# # # ax.barh(x_pos, CTEs, xerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
# # ax.errorbar(means, datasets, xerr=stds, fmt='o')
# # ax.set_xlabel('size of explanation')
# # ax.set_yticks(datasets)
# # # ax.set_yticklabels(datasets)
# # # ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
# # ax.xaxis.grid(True)
# # ax.set_ylabel('dataset')
# # ax.set_xlim(0, 6)

# # # Save the figure and show
# # plt.tight_layout()
# # plt.savefig('bar_plot_with_error_bars.png')
# # plt.show()

# # plt.savefig('test_len_explanation.png')


# # plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45) 
# # # Create offset transform by 5 points in x direction
# # dx = 5/72.; dy = 0/72. 
# # offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
# # # apply offset transform to all x ticklabels.
# # for label in ax.xaxis.get_majorticklabels():
# #     label.set_transform(label.get_transform() + offset)


# plt.tight_layout()
# plt.savefig('test_len_explanation_v2.png')