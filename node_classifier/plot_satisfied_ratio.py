


import os

import numpy as np

# import necessary libraries
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt


# plt.cm.get_cmap('Paired')


# ## working plot
# single = [0.318, 0.321, 0.519, 0.125]
# compound = [0.534, 0.759, 0.799, 0.354]
# dif_compound = list(np.array(compound) - np.array(single))
 
# # create DataFrame
# df = pd.DataFrame({'single': single,
#                    'compound': dif_compound},
#                   index=['AIFB', 'MUTAG', 'AM', 'MDGENRE'])
 
 
# # create stacked bar chart for monthly temperatures
# df.plot(kind='bar', stacked=True, color=['darkorange', 'darkturquoise'], rot=0)

# plt.ylim(0, 1)
 
# # labels for x & y axis
# plt.xlabel('dataset')
# plt.ylabel('ratio of satisfied explanation condition')

# # plt.xticks(rotation=90)
 
# # title of plot
# # plt.title('Monthly Temperatures in a year')

# plt.savefig('test_satisfied_ratio.png')


# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.DataFrame({'subsidiary': ['EU','EU','EU','EU','EU','EU','EU','EU','EU','US','US','US','US','US','US','US','US','US'],'date': ['2019-03','2019-04', '2019-05','2019-03','2019-04', '2019-05','2019-03','2019-04', '2019-05','2019-03','2019-04', '2019-05','2019-03','2019-04', '2019-05','2019-03','2019-04', '2019-05'],'business': ['RETAIL','RETAIL','RETAIL','CORP','CORP','CORP','PUBLIC','PUBLIC','PUBLIC','RETAIL','RETAIL','RETAIL','CORP','CORP','CORP','PUBLIC','PUBLIC','PUBLIC'],'value': [500.36,600.45,700.55,750.66,950.89,1300.13,100.05,120.00,150.01,800.79,900.55,1000,3500.79,5000.36,4500.25,50.17,75.25,90.33]})
# # print(df)

# import seaborn as sns

# plot = sns.catplot(kind='bar', data=df, col='subsidiary', x='date', y='value', hue='business')

# for ax in plot.axes.ravel():
#     for c in ax.containers:
#         ax.bar_label(c, label_type='edge')


# plot.savefig('test_satisfied_ratio_v2.png')




main_path = '/home/fpaulino/SEEK/seek/node_classifier/cv_model_rf_local_final'

explanation_limit = 'class_change'
# explanation_limit = 'threshold'

# explan_type = 'necessary'
explan_type = 'sufficient'

results_file = f'global_explain_stats_{explanation_limit}.csv'

datasets = ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']

kge_models = ['RDF2Vec', 'ComplEx', 'distMult', 'TransH', 'TransE']

single_list = []
compound_list = []
for kge_model in kge_models:
    for dataset in datasets:
    # for kge_model in kge_models:
        df = pd.read_csv(os.path.join(main_path, f'{dataset}_{kge_model}', results_file), sep='\t')
        single_list.append(float(df[f'{explan_type}_len1_satisfied_class_change_racio'].values[0]))
        compound_list.append(float(df[f'{explan_type}_len5_satisfied_class_change_racio'].values[0]))


# print(compound_list)
# print(len(compound_list))
# raise


plt.cm.get_cmap('Paired')
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

import matplotlib.pyplot as plt
import numpy as np

# import matplotlib.transforms

width = 0.25
x = np.arange(1, 5)

fig, ax = plt.subplots(figsize=(10, 6))

# tick_labels_1 = ['1'] * len(x)
# tick_labels_2 = ['2'] * len(x)
# tick_labels_3 = ['3'] * len(x)
# tick_labels_4 = ['4'] * len(x)
# tick_labels_5 = ['5'] * len(x)
tick_labels_1 = ['RDF2Vec'] * len(x)
tick_labels_2 = ['ComplEx'] * len(x)
tick_labels_3 = ['distMult'] * len(x)
tick_labels_4 = ['TransH'] * len(x)
tick_labels_5 = ['TransE'] * len(x)
shift1_rbc = np.random.uniform(1100, 1200, 4) # rdf2vec in 4 datasets
shift2_rbc = np.random.uniform(900, 1000, 4) # complex in 4 datasets
shift3_rbc = np.random.uniform(1000, 1100, 4)
shift4_rbc = np.random.uniform(900, 1000, 4)
shift5_rbc = np.random.uniform(1000, 1100, 4)
shift1_plt = np.random.uniform(600, 700, 4)
shift2_plt = np.random.uniform(400, 500, 4)
shift3_plt = np.random.uniform(500, 600, 4)
shift4_plt = np.random.uniform(400, 500, 4)
shift5_plt = np.random.uniform(500, 600, 4)
# print(shift1_rbc, shift2_rbc)
# print('\n')
# print(np.concatenate([shift1_rbc, shift2_rbc]))
# raise
# print(shift1_rbc)
# raise
# shift1_ffp = np.random.uniform(250, 300, 6)
# shift2_ffp = np.random.uniform(150, 200, 6)
# shift3_ffp = np.random.uniform(200, 250, 6)
# all_x = np.concatenate([x - 0.4, x - 0.1, x + 0.2])
all_x = np.concatenate([x - 0.4, x - 0.25, x -0.1, x + 0.05, x + 0.2])
# ax.bar(all_x, np.concatenate([shift1_rbc, shift2_rbc, shift3_rbc, shift4_rbc, shift5_rbc]), width * .55,
#        tick_label=tick_labels_1 + tick_labels_2 + tick_labels_3 + tick_labels_4 + tick_labels_5,
#        color=_Paired_data[1], label='compound')
print(compound_list)
ax.bar(all_x, compound_list, width * .55,
       tick_label=tick_labels_1 + tick_labels_2 + tick_labels_3 + tick_labels_4 + tick_labels_5,
       color=_Paired_data[1], label='compound')
# ax.bar(all_x, np.concatenate([shift1_plt, shift2_plt, shift3_plt, shift4_plt, shift5_plt]),
#        width * .45, color=_Paired_data[0], label='single')
ax.bar(all_x, single_list,
       width * .45, color=_Paired_data[0], label='single')
ax.tick_params(axis='x', labelrotation=90, labelsize=15)
# ax.bar(all_x, np.concatenate([shift1_ffp, shift2_ffp, shift3_ffp]),
#        width * .5, color='limegreen', label='green')
ax.margins(x=0.02)
# ax.legend(title='Data', bbox_to_anchor=(0.99, 1), loc='upper left')
if explan_type == 'necessary':
    ax.legend(bbox_to_anchor=(0.74, 1), loc='upper left', fontsize=15)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
else:
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

ax.set_xticks(x - 0.1001, minor=True)
# ax.set_xticks(x, minor=True)
ax.set_xticklabels(['AIFB', 'MUTAG', 'AM', 'MDGENRE'], minor=True)
# ax.tick_params(axis='x', which='minor', length=0, pad=18)
ax.tick_params(axis='x', which='minor', length=0, pad=90, labelsize=15)
# ax.set_xlabel('dataset - kge model')
ax.tick_params(axis='y', labelsize=15)
ax.set_ylabel(f'{explan_type} explanations', fontsize=15)


# plt.setp(ax.xaxis.get_majorticklabels(), rotation=-45) 
# # Create offset transform by 5 points in x direction
# dx = 5/72.; dy = 0/72. 
# offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
# # apply offset transform to all x ticklabels.
# for label in ax.xaxis.get_majorticklabels():
#     label.set_transform(label.get_transform() + offset)


plt.tight_layout()
plt.savefig(f'ratio_satisfied_explanations_{explan_type}.png')