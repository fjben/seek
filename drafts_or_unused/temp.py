


from joblib import load

datasets = ['AIFB', 'MUTAG', 'AM_FROM_DGL', 'MDGENRE']
kge_models = ['RDF2Vec', 'ComplEx', 'distMult', 'TransE', 'TransH']
models = list(range(10))

for ds in datasets:
    for kge_m in kge_models:
        for num in models:
            clf = load(f'/home/fpaulino/SEEK/seek/node_classifier/cv_model_mlp_local_final/{ds}_{kge_m}/{ds}_model_{num}_RAN/models/classifier_{ds}')
            print(clf.estimator.hidden_layer_sizes)