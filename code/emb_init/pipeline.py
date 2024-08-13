'''
======================================
Full Pykg2vec pipeline (kgpipeline.py)
======================================
kgpipeline.py demonstrates the full pipeline of training KGE methods with pykg2vec.
This pipeline first discover the best set of hyperparameters using training and validation set.
Then it uses the discovered hyperparameters to evaluate the KGE algorithm on the testing set. ::

    python kgpipeline.py

====

We also attached the source code of kgpipeline.py below for your reference.
You can adjust to fit your usage.

'''
# Author: Sujit Rokka Chhetri and Shih Yuan Yu
# License: MIT

from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.data.kgcontroller import KnowledgeGraph
from pykg2vec.utils.bayesian_optimizer import BaysOptimizer
from pykg2vec.utils.trainer import Trainer


def main():
    model_name = "transr"
    dataset_name = "ml-1m"
    dataset_path = "../datasets/ml-1m_my/kgdata/"#"path_to_dataset"
    yaml_path = "custom_hp.yaml"

    # 1. Tune the hyper-parameters for the selected model and dataset.
    # p.s. this is using training and validation set.
    args = KGEArgParser().get_args(['-mn', model_name, '-ds', dataset_name, '-dsp', dataset_path, '-hpf', yaml_path])

    # initializing bayesian optimizer and prepare data.
    bays_opt = BaysOptimizer(args=args)

    # perform the golden hyperparameter tuning.
    bays_opt.optimize()
    best = bays_opt.return_best()

    # 2. Evaluate final model using the found best hyperparameters on testing set.
    args = KGEArgParser().get_args(['-mn', model_name, '-ds', dataset_name, '-dsp', dataset_path, '-hpf', yaml_path])

    # Preparing data and cache the data for later usage
    knowledge_graph = KnowledgeGraph(dataset=args.dataset_name, custom_dataset_path=args.dataset_path)
    knowledge_graph.prepare_data()

    # Extracting the corresponding model config and definition from Importer().
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args)

    # Update the config params with the golden hyperparameter
    for k, v in best.items():
        config.__dict__[k] = v
    model = model_def(**config.__dict__)

    # Create, Compile and Train the model.
    trainer = Trainer(model, config)
    trainer.build_model()
    trainer.train_model()


if __name__ == "__main__":
    main()
