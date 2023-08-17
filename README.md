## Getting Started

Followint these steps to local running.

### Prerequisites

1. Check environments.yaml for list of needed packages.
2. Download ```_database``` directory from the [pnet project](https://github.com/marakeby/pnet_prostate_paper) if you need train model and test model. You can store it somewhere else by setting ```DATA_PATH``` variable and change directory name by setting ```data``` in ```config.py``` accordingly.
3. if you want run experiment 3, please train and explain model first to genarate required file.

### Installation

1. Clone the repository
    ```sh
    git clone https://github.com/yisnsiy/pnet_based_on_pytorch.git
    ```

2. Create environment
    ```sh
    conda env create --name pnet_pytorch --file=environment.yaml
    ```

## Usage

1. Activate conda environment
    ```sh
    conda activate pnet_pytorch
    ```

2. Train model
    ```sh
    python run_me.py
    ```

3. Change the config file ```config.py``` based your use and re-train model.


## License

Distributed under the GPL-2.0 License License. See `LICENSE` for more information.

## References
* Elmarakeby, Haitham A., et al. "Biologically informed deep neural network for prostate cancer discovery." Nature 598.7880 (2021): 348-352.
* Bevilacqua, Michele, and Roberto Navigli. "Breaking through the 80% glass ceiling: Raising the state of the art in word sense disambiguation by incorporating knowledge graph information." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2020.