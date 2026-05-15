from utils.utils import pipeline_wrapper, merge
from utils import config

import argparse
import os
import time
from multiprocessing import Process


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MRELight for traffic signal control."
    )

    # Experiment name used under model/, records/, summary/, and errors/.
    parser.add_argument(
        "--memo",
        "-memo",
        type=str,
        default="benchmark_1001",
        help="Experiment memo name. Default: benchmark_1001."
    )

    parser.add_argument(
        "--model",
        "-mod",
        type=str,
        choices=["MRELight"],
        default="MRELight",
        help="Model name. This release only supports MRELight."
    )

    # Dataset selection.
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hangzhou", "jinan", "newyork"],
        default="hangzhou",
        help="Dataset to run. Default: hangzhou."
    )

    # Default paper-style training rounds.
    parser.add_argument(
        "--rounds",
        type=int,
        default=120,
        help="Number of training rounds. Default: 120."
    )

    # Number of data generators per training round.
    # In each round, the pipeline runs NUM_GENERATORS simulations to collect training samples.
    parser.add_argument(
        "--gen",
        "-gen",
        type=int,
        default=1,
        help="Number of generators per training round. Default: 1."
    )

    parser.add_argument(
        "--eightphase",
        "-eightphase",
        action="store_true",
        default=False,
        help="Use eight-phase signal setting. Default: False."
    )

    parser.add_argument(
        "--multi_process",
        "-multi_process",
        action="store_true",
        default=False,
        help="Run different traffic files in parallel. Default: False."
    )

    parser.add_argument(
        "--workers",
        "-workers",
        type=int,
        default=3,
        help="Maximum number of parallel traffic processes. Default: 3."
    )

    # -1 means running all traffic files in the selected dataset.
    # For quick debugging, use --traffic_index 0.
    parser.add_argument(
        "--traffic_index",
        type=int,
        default=-1,
        help="Traffic file index to run. Use -1 to run all traffic files. Default: -1."
    )

    return parser.parse_args()


def get_dataset_config(dataset):
    if dataset == "hangzhou":
        count = 3600
        road_net = "4_4"
        traffic_file_list = [
            "anon_4_4_hangzhou_real_5816.json",
            "anon_4_4_hangzhou_real.json",
        ]
        template = "Hangzhou"

    elif dataset == "jinan":
        count = 3600
        road_net = "3_4"
        traffic_file_list = [
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json",
            "anon_3_4_jinan_real.json",
        ]
        template = "Jinan"

    elif dataset == "newyork":
        count = 3600
        road_net = "28_7"
        traffic_file_list = [
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json",
        ]
        template = "newyork_28_7"

    else:
        raise ValueError("Unsupported dataset: {}".format(dataset))

    return count, road_net, traffic_file_list, template


def main(in_args=None):
    if in_args is None:
        in_args = parse_args()

    if in_args.model != "MRELight":
        raise ValueError("This release only supports MRELight.")

    count, road_net, traffic_file_list, template = get_dataset_config(in_args.dataset)

    if in_args.traffic_index >= 0:
        if in_args.traffic_index >= len(traffic_file_list):
            raise ValueError(
                "traffic_index {} is out of range for dataset {}. Available range: 0 to {}.".format(
                    in_args.traffic_index,
                    in_args.dataset,
                    len(traffic_file_list) - 1
                )
            )
        traffic_file_list = [traffic_file_list[in_args.traffic_index]]

    num_col = int(road_net.split("_")[1])
    num_row = int(road_net.split("_")[0])
    num_intersections = num_row * num_col

    print("dataset:", in_args.dataset)
    print("road_net:", road_net)
    print("num_intersections:", num_intersections)
    print("traffic_file_list:", traffic_file_list)
    print("num_rounds:", in_args.rounds)
    print("num_generators:", in_args.gen)

    process_list = []

    for traffic_file in traffic_file_list:
        dic_agent_conf_extra = {
            "CNN_layers": [[32, 32]],
        }

        deploy_dic_agent_conf = merge(
            config.DIC_BASE_AGENT_CONF,
            dic_agent_conf_extra
        )

        dic_traffic_env_conf_extra = {
            "NUM_ROUNDS": in_args.rounds,
            "NUM_GENERATORS": in_args.gen,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "MODEL_NAME": in_args.model,

            "NUM_ROW": num_row,
            "NUM_COL": num_col,

            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "LIST_STATE_FEATURE": [
                "cur_phase",
                "lane_num_vehicle",
                "average_vehicle_speed",
                "adjacency_matrix",
            ],

            "DIC_REWARD_INFO": {
                "queue_length": -0.25,
                "average_speed": 0.25,
            },
        }

        if in_args.eightphase:
            dic_traffic_env_conf_extra["NUM_PHASES"] = 8
            dic_traffic_env_conf_extra["PHASE"] = {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0],
            }
            dic_traffic_env_conf_extra["PHASE_LIST"] = [
                "WT_ET",
                "NT_ST",
                "WL_EL",
                "NL_SL",
                "WL_WT",
                "EL_ET",
                "SL_ST",
                "NL_NT",
            ]

        time_stamp = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
        experiment_name = "{0}_{1}_{2}".format(
            in_args.model,
            traffic_file,
            time_stamp
        )

        dic_path_extra = {
            "PATH_TO_MODEL": os.path.join(
                "model",
                in_args.memo,
                experiment_name
            ),
            "PATH_TO_WORK_DIRECTORY": os.path.join(
                "records",
                in_args.memo,
                experiment_name
            ),
            "PATH_TO_DATA": os.path.join(
                "data",
                template,
                str(road_net)
            ),
            "PATH_TO_ERROR": os.path.join(
                "errors",
                in_args.memo
            ),
        }

        deploy_dic_traffic_env_conf = merge(
            config.dic_traffic_env_conf,
            dic_traffic_env_conf_extra
        )

        deploy_dic_path = merge(
            config.DIC_PATH,
            dic_path_extra
        )

        print("========================================")
        print("Start experiment:", experiment_name)
        print("PATH_TO_DATA:", deploy_dic_path["PATH_TO_DATA"])
        print("PATH_TO_WORK_DIRECTORY:", deploy_dic_path["PATH_TO_WORK_DIRECTORY"])
        print("PATH_TO_MODEL:", deploy_dic_path["PATH_TO_MODEL"])

        if in_args.multi_process:
            process = Process(
                target=pipeline_wrapper,
                args=(
                    deploy_dic_agent_conf,
                    deploy_dic_traffic_env_conf,
                    deploy_dic_path,
                )
            )
            process_list.append(process)
        else:
            pipeline_wrapper(
                dic_agent_conf=deploy_dic_agent_conf,
                dic_traffic_env_conf=deploy_dic_traffic_env_conf,
                dic_path=deploy_dic_path
            )

    if in_args.multi_process:
        for i in range(0, len(process_list), in_args.workers):
            i_max = min(len(process_list), i + in_args.workers)

            for j in range(i, i_max):
                print("start traffic process:", j)
                process_list[j].start()

            for k in range(i, i_max):
                print("traffic process to join:", k)
                process_list[k].join()
                print("traffic process finished:", k)

    return in_args.memo


if __name__ == "__main__":
    args = parse_args()
    main(args)