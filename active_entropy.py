# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detectron2 training script with a plain training loop.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer features, and also
includes fewer abstraction.
"""
import math
import shutil
import logging
import os,csv,random
from collections import OrderedDict
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from contextlib import contextmanager
from detectron2.data import DatasetCatalog, MetadataCatalog
import detectron2.utils.comm as comm
from detectron2.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_train_loader,
    build_detection_test_loader,
    build_batch_data_loader
)
from detectron2.data.samplers import InferenceSampler
from torch.utils.data.sampler import Sampler
from detectron2.data.build import DatasetMapper,get_detection_dataset_dicts,DatasetFromList,MapDataset,trivial_batch_collator
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    COCOEvaluator,
    DatasetEvaluators
    #RotatedCOCOEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from relation_data_tool_old import register_pathway_dataset, PathwayDatasetMapper, register_Kfold_pathway_dataset
from pathway_evaluation import PathwayEvaluator

logger = logging.getLogger("pathway_parser")


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg= cfg,dataset_name= dataset_name, mapper= PathwayDatasetMapper(cfg, False))
        evaluator = PathwayEvaluator(
             dataset_name=dataset_name, cfg= cfg, distributed= False, output_dir= os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_train(cfg, model, output_dir, resume=False):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    model.to(device)


    model.train()
    

    print(model)

    #do not load checkpointer's optimizer and scheduler
    checkpointer = DetectionCheckpointer(
        model, os.path.join(output_dir,"models"))
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )

    #model.load_state_dict(optimizer)
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    train_data_loader = build_detection_train_loader(cfg, mapper= DatasetMapper(cfg, True))

    val_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=output_dir+"/inference")
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])


    # epoch_num has # of iterations per epoch
    if cfg.DATALOADER.ASPECT_RATIO_GROUPING:
        epoch_num = (train_data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH) + 1
    else:
        epoch_num = train_data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH


    # max_iter is # of iterations for set # of epochs
    max_iter = (train_data_loader.dataset.sampler._size // cfg.SOLVER.IMS_PER_BATCH) * 2

    
    cfg.defrost()
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.freeze()
    

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
        if comm.is_main_process()
        else []
    )

    print(epoch_num)
    print(max_iter)
    max_val = -1000000
    logger.info("Starting training from iteration {}".format(start_iter))
    loss_weights = {'loss_cls': 1, 'loss_box_reg': 1}
    with EventStorage(start_iter) as storage:
        loss_per_epoch = 0.0
        best_loss = 99999.0
        best_val_loss = 99999.0
        better_train = False
        better_val = False
        for data, iteration in zip(train_data_loader, range(0, max_iter)):
            iteration = iteration + 1
            storage.step()

            loss_dict,_ = model(data)

            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() * loss_weights[k]  for k, v in comm.reduce_dict(loss_dict).items()}

            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            #prevent gredient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            #if comm.is_main_process():
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            loss_per_epoch += losses_reduced
            # if iteration % epoch_num == 0 or iteration == max_iter:
            if iteration % epoch_num == 0:

                #do validation
                outputs = inference_on_dataset(model, val_loader, val_evaluator)

                if not math.isnan(outputs['bbox']['AP50']) and outputs['bbox']['AP50'] > max_val:

                    max_val = outputs['bbox']['AP50']

                    for i in os.listdir(os.path.join(output_dir,"models")):
                        os.remove(os.path.join(output_dir,"models",i))

                    checkpointer.save("model_{:07d}".format(iteration), **{"iteration": iteration})

                for writer in writers:
                    writer.write()

                #reset loss_per_epoch
                loss_per_epoch = 0.0
             
            del loss_dict,losses,losses_reduced,loss_dict_reduced
            torch.cuda.empty_cache()

def evaluate_all_checkpoints(args, checkpoint_folder, output_csv_file):
    cfg = setup(args)
    csv_results=[]
    header = []
    for file in os.listdir(checkpoint_folder):

        file_name, file_ext = os.path.splitext(file)
        if file_ext != ".pth" :
            continue

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(checkpoint_folder, file), resume=False)
        results=do_test(cfg, model)
        results['bbox'].update(checkpoint=file)
        header = results['bbox'].keys()
        csv_results.append(results['bbox'])
        print('main_results:')
        print(results)
        del results
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = header)
        writer.writeheader()
        writer.writerows(csv_results)
    csvfile.close()
    del csv_results, header

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # customize reszied parameters
    # cfg['INPUT']['MIN_SIZE_TRAIN'] = (20,)
    # cfg['INPUT']['MAX_SIZE_TRAIN'] = 50
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def clear_train_namespace(output_dir):

    if "pathway_val_0_regular_element" in MetadataCatalog.list():
        MetadataCatalog.remove('pathway_val_0_regular_element')
        
    if "pathway_train_0_regular_element" in MetadataCatalog.list():
        MetadataCatalog.remove('pathway_train_0_regular_element')
        
    if "pathway_train_0_regular_element" in DatasetCatalog.list():
        DatasetCatalog.remove('pathway_train_0_regular_element')
        
    if "pathway_val_0_regular_element" in DatasetCatalog.list():
        DatasetCatalog.remove('pathway_val_0_regular_element')

    for i in os.listdir(os.path.join(output_dir,"inference")):
        os.remove(os.path.join(output_dir,"inference",i))


def eval_points(cfg, model, output_dir):

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.nn.DataParallel(model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    model.to(device)

    model.eval()

    model_checkpoint = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
    for file in os.listdir(os.path.join(output_dir,"models")):
        if "model" in file:
            model_checkpoint = os.path.join(output_dir,"models",file)

    cfg.defrost()
    cfg.MODEL.WEIGHTS = model_checkpoint
    cfg.freeze()
    
    DetectionCheckpointer(model, output_dir).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )


    category_list = ['activate','inhibit','indirect_activate','indirect_inhibit']
    # category_list = ['gene','cluster']
    img_path = r'dataset_margin2/img/'
    json_path = r'dataset_margin2/json/'
    register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)


    val_evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], output_dir=output_dir+"/inference")
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])

    print(len(val_loader))

    filenames = []
    avg_scores = []
    for data, iteration in zip(val_loader, range(0, 15)):

        predictions = model(data,learn_active=True)
        pred_scores = predictions[0]._fields['scores']
        avg_score = torch.mean(pred_scores).cpu().detach().numpy()
        avg_scores.append(avg_score)
        filenames.append(data[0]['file_name'])

        # print(predictions)
    print(avg_scores)
    print(filenames)

    avg_scores = np.array(avg_scores)

    sort_indices = np.argsort(avg_scores)[:10]
    # top_scores = avg_scores[sort_indices]
    # filenames = filenames[sort_indices]

    source_dir = "dataset_margin2"
    target_dir = "dataset_margin1"
    for sorted_idx in sort_indices:
        filepath = filenames[sorted_idx]
        file = os.path.basename(filepath)
        shutil.move(os.path.join(source_dir,"img",file), os.path.join(target_dir,"img",file))
        shutil.move(os.path.join(source_dir,"json","val_0",file[:-3]+"json"), os.path.join(target_dir,"json","train_0",file[:-3]+"json"))



def main(args):
    cfg = setup(args)

    # import the relation_retinanet as meta_arch, so they will be registered
    from relation_retinanet import RelationRetinaNet

    # if args.eval_only:
    #         DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #             cfg.MODEL.WEIGHTS, resume=args.resume
    #         )
    #         return do_test(cfg, model)

    output_dir = "output_active"

    for train_idx in range(10):

        clear_train_namespace(output_dir)

        category_list = ['activate','inhibit','indirect_activate','indirect_inhibit']
        # category_list = ['gene','cluster']
        img_path = r'dataset_margin1/img/'
        json_path = r'dataset_margin1/json/'
        register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        
        do_train(cfg, model, output_dir)

        clear_train_namespace(output_dir)

        eval_points(cfg, model, output_dir)



if __name__ == "__main__":
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    # category_list = ['gene','activate','inhibit']
    # img_path = r'/home/fei/Desktop/debug_data/image_0109/'
    # json_path = r'/home/fei/Desktop/debug_data/json_0109/'
    # register_Kfold_pathway_dataset(json_path, img_path, category_list, K=1)
    #register_pathway_dataset(json_path, img_path, category_list)


    parser = default_argument_parser()
    # parser.add_argument("--task", choices=["train", "eval", "data"], required=True)
    args = parser.parse_args()
    assert not args.eval_only
    #args.eval_only = True
    #args.num_gpus = 2
    args.config_file = r'Base-RetinaNet.yaml'
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
