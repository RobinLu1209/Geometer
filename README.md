# GEOMETER
Graph Few-Shot Class-Incremental Learning via Prototype Representation

![Graph Few-Shot Class-Incremental Learning via Prototype Representation](system_model.png "Model Architecture")

## Requirements
- pytorch >= 1.8.1
- numpy >= 1.21.3
- scikit-learn >= 0.24.2
- pytorch geometric >= 2.0.2
- pyaml
- tensorboardX
- tqdm

## How to run
```bash
python main.py --config_filename='config/config_cora_stream.yaml' --iteration 10 
```

## Citation
If you find this repository, e.g., the paper, code and the datasets, useful in your research, please cite the following paper:
```
@inproceedings{DBLP:conf/kdd/Geometer,
  author    = {Bin Lu and
               Xiaoying Gan and
               Lina Yang and
               Weinan Zhang and
               Luoyi Fu and
               Xinbing Wang},
  title     = {Geometer: Graph Few-Shot Class-Incremental Learning via Prototype Representation},
  booktitle = {{KDD} '22: The 28th {ACM} SIGKDD Conference on Knowledge Discovery and Data Mining,
              Washington, DC, USA, August 14--18, 2022},
  publisher = {{ACM}},
  year      = {2022}
}
```
