# MRELight

Official implementation of **MRELight**, a multi-reward reinforcement learning method for traffic signal control.

MRELight uses multi-objective Q-values for traffic signal control. The two reward objectives are queue length and efficient average speed. Action selection is performed through Tchebycheff scalarization.

## Repository Structure

```text
MRELight/
├── data/                  # Traffic datasets
│   ├── Hangzhou/
│   ├── Jinan/
│   └── newyork_28_7/
├── models/                # MRELight agent implementation
│   ├── agent.py
│   └── mrelight_agent.py
├── utils/                 # Environment, training pipeline, and utility modules
│   ├── cityflow_env.py
│   ├── config.py
│   ├── construct_sample.py
│   ├── generator.py
│   ├── model_test.py
│   ├── pipeline.py
│   ├── scalarization.py
│   ├── updater.py
│   └── utils.py
├── run_mrelight.py         # Main running script
├── summary.py              # Result summarization script
├── requirements.txt        # Python dependencies
└── README.md
```

## Requirements

The code has been tested with the following environment:

```text
Python: 3.6.15
TensorFlow: 2.4.1
pandas: 1.1.5
NumPy: 1.19.5
pymoo: 0.5.0
CityFlow: 0.1
```

CityFlow requires a Linux environment. We tested the code on a Linux server.

We recommend creating a new conda environment:

```bash
conda create -n mrelight python=3.6
conda activate mrelight
pip install -r requirements.txt
```

Please also make sure CityFlow is properly installed. You can verify the installation by running:

```bash
python -c "import cityflow; print('CityFlow installed')"
```

## Dataset Structure

The expected dataset structure is:

```text
data/
├── Hangzhou/
│   └── 4_4/
│       ├── roadnet_4_4.json
│       ├── anon_4_4_hangzhou_real_5816.json
│       └── anon_4_4_hangzhou_real.json
├── Jinan/
│   └── 3_4/
│       ├── roadnet_3_4.json
│       ├── anon_3_4_jinan_real_2000.json
│       ├── anon_3_4_jinan_real_2500.json
│       └── anon_3_4_jinan_real.json
└── newyork_28_7/
    └── 28_7/
        ├── roadnet_28_7.json
        ├── anon_28_7_newyork_real_double.json
        └── anon_28_7_newyork_real_triple.json
```

## Default Run

The default command is:

```bash
python run_mrelight.py
```

This is equivalent to:

```bash
python run_mrelight.py --dataset hangzhou --rounds 120 --gen 1 --traffic_index -1
```

By default, the code runs MRELight on the Hangzhou dataset for 120 training rounds.

## Command Line Arguments

| Argument | Description | Default |
|---|---|---|
| `--memo` | Experiment name used for saving records, models, summaries, and errors | `benchmark_1001` |
| `--model` | Model name. This release only supports `MRELight` | `MRELight` |
| `--dataset` | Dataset name: `hangzhou`, `jinan`, or `newyork` | `hangzhou` |
| `--rounds` | Number of training rounds | `120` |
| `--gen` | Number of data generators per training round | `1` |
| `--traffic_index` | Index of the traffic file to run. Use `-1` to run all traffic files | `-1` |
| `--eightphase` | Use eight-phase signal setting | `False` |
| `--multi_process` | Run different traffic files in parallel | `False` |
| `--workers` | Maximum number of parallel traffic processes | `3` |

## Outputs

The code will generate the following output folders:

```text
records/<memo>/
model/<memo>/
summary/<memo>/
errors/<memo>/
```

The `records/` folder stores training and testing logs.

The `model/` folder stores trained model files.

The `summary/` folder stores summarized test results.

The `errors/` folder stores error logs when exceptions occur.

## Summarizing Results

After running experiments, use:

```bash
python summary.py --memo benchmark_1001
```

The summarized results will be saved to:

```text
summary/<memo>/total_test_results.csv
```

## Method Configuration

The main MRELight configuration is in:

```text
utils/config.py
```

The default objective weights are:

```text
OBJECTIVE_WEIGHTS = [0.7, 0.3]
```

The default Tchebycheff parameter is:

```text
TCHEBICHEFF_TAU = 4
```

The reward coefficients are:

```text
queue_length: -0.25
average_speed: 0.25
```

The reward normalization factor is:

```text
NORMAL_FACTOR = 20
```

## Citation

The citation information will be updated after the paper is published.

## License

This project is released under the Apache-2.0 License.