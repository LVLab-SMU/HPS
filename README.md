# Hard Preference Sampling for Human Preference Alignment
<p align="center">
  <a href="https://github.com/Yqcca/HPS"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
  <a href="https://arxiv.org/abs/2502.14400"><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
</p>

## 📜 Description
TL;DL: We propose Hard Preference Sampling (HPS), a novel framework for robust and efficient human preference alignment.

## 🚀 Getting Started
### Setting Up the Environment
To begin, set up your environment with the necessary packages. It is recommended to have two separate environments for **inference** and **training**, respectively. 

**Inference Environment**

```sh
conda create -n vllm python=3.10
conda activate vllm

# The following code is tested for CUDA11.8 and CUDA12.6
pip3 install torch==2.4.0 torchvision torchaudio
pip install datasets==3.1.0
pip install vllm==0.5.4
pip install accelerate==1.2.1
pip install deepspeed==0.14.5
pip install huggingface-hub==0.26.2
pip install transformers==4.47.1
pip install numpy==1.26.4

pip install xformers
pip install trl
pip install flash-attn
pip install einops
pip install ninja
pip install nltk
pip install peft
```

**Training Environment**

```sh
conda create -n rlhflow python=3.10
conda activate rlhflow

# The following code is tested for CUDA11.8 and CUDA12.6
git clone https://github.com/huggingface/alignment-handbook.git
cd ./alignment-handbook/
git checkout 27f7dbf00663dab66ad7334afb7a1311fa251f41
pip3 install torch==2.1.2 torchvision torchaudio
python -m pip install .
pip install flash-attn==2.6.3
pip install accelerate==0.34.0
pip install huggingface-hub==0.24.7
pip install transformers==4.46.2
pip install trl
```

You can also install the wandb to record the training and login with your huggingface account.

```sh
pip install wandb==0.17.7

wandb login
```

### Preference Data Generation and Annotation
We provide two cleaned and curated prompt sets on Hugging Face: 
* [HH-RLHF](https://huggingface.co/datasets/yqcca/HH-RLHF)
* [PKU-Safety](https://huggingface.co/datasets/yqcca/PKU-Safety)

You can download them using the Hugging Face CLI:
```sh
brew install huggingface-cli

huggingface-cli login

huggingface-cli download yqcca/HH-RLHF online_hh.json --local-dir HPS/data --repo-type dataset
huggingface-cli download yqcca/PKU-Safety pkusafe.json --local-dir HPS/data --repo-type dataset
```

Our workflow involves expanding the response candidates using a strong instruction LLM, followed by ranking these responses using a trained reward model or human evaluation. An example is provided using a [Llama3-Instruct](https://huggingface.co/RLHFlow/Llama3-v2-iterative-DPO-iter3) to expand responses for the HH-RLHF dataset. The responses are then scored using the [Skywork Reward Model](https://huggingface.co/Skywork/Skywork-Reward-Llama-3.1-8B).

To conduct the data generation and annotation pipeline:

```sh
bash gene.sh
```

### LLM Fine-tuning based on HPS
We integrate our proposed HPS into several implicit reward parameterization frameworks to fine-tune supervised LLM baselines. For example, HPS can be incorporated into Direct Preference Optimization(DPO) to fine-tune a [Llama3 checkpoint](https://huggingface.co/RLHFlow/Llama3-SFT-v2.0-epoch3).

To run the fine-tuning pipeline:

```sh
bash run.sh
```

We refer the interested readers to this [repo](https://github.com/RLHFlow/Online-RLHF) for a detailed recipe to train the state-of-the-art open-source SFT models, reward models and RLHF models.

## Acknowledgments
* This code is built on the [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), and [Online-RLHF](https://github.com/RLHFlow/Online-RLHF). The authors would like to thank the open-source community for sharing the models, codebases, and training datasets. 

## Citation

If you find this work useful, please kindly cite our paper:

```bibtex
@article{zou2025hps,
  title={HPS: Hard Preference Sampling for Human Preference Alignment},
  author={Zou, Xiandong and Lin, Wanyu and Li, Yuchen and Zhou, Pan},
  journal={arXiv preprint arXiv:2502.14400},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.