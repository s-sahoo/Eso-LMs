# Esoteric Language Models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12dWwEsPWaKq2vaTJoRy7ZEClTLGRFmeE?usp=sharing)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://s-sahoo.com/Eso-LMs/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01928-red.svg)](https://arxiv.org/abs/2506.01928)
[![deploy](https://img.shields.io/badge/HuggingFace%20-Eso--LMs%20-blue)](https://huggingface.co/collections/sahoo-diffusion/eso-lms-6838e86cb2c49f45302f0092)

<a href="https://s-sahoo.com/"><b>Subham Sekhar Sahoo</b></a><sup>\*</sup><sup>1</sup>, <a href="https://zhihanyang2022.github.io"><b>Zhihan Yang</b></a><sup>\*</sup><sup>2</sup>, <a href="https://akhauriyash.github.io/">Yash Akhauri</a><sup>†1</sup>, <a href="https://johnnajliu.github.io/">Johnna Liu</a><sup>†1</sup>, <a href="http://www.deepansha.com/">Deepansha Singh</a><sup>†1</sup>, <a href="https://blankcheng.github.io/">Zhoujun Cheng</a><sup>†3</sup>, <a href="https://hunterhector.github.io/">Zhengzhong Liu</a><sup>3</sup>, <a href="https://www.cs.cmu.edu/~epxing/">Eric Xing</a><sup>3</sup>, <a href="https://johnthickstun.com/">John Thickstun</a><sup>2</sup>, <a href="http://latentspace.cc/">Arash Vahdat</a><sup>4</sup>  

<sup>1</sup>Cornell Tech <sup>2</sup>Cornell University <sup>3</sup>MBZUAI <sup>4</sup>NVIDIA  
<sup>\*</sup>Joint first authors <sup>†</sup>Joint second authors

**Pre-print 2025**

<p align="center">
  <img src="https://github.com/user-attachments/assets/51506974-12f5-4379-ab2e-62fc0c5bc408" alt="Eso-LM (B)" style="width:100%;">
</p>

We propose **Eso**teric **L**anguage **M**odels (Eso-LMs), a new framework for language modeling that fuses AR and MDM paradigms and outperforms the previous hybrid approach, BD3-LMs. Our model uses a revised attention mechanism to support both paradigms, and is trained with a hybrid loss—a combination of AR and MDM objectives—which allows it to interpolate smoothly between the two paradigms in terms of perplexity and sample quality. Further, ours is the first approach to enable KV caching for MDMs while preserving parallel generation, achieving up to **65× faster inference** than standard MDMs and **4× faster inference** than KV-cached semi-autoregressive baselines. 

In this repository, we release both variants of Eso-LMs: Eso-LM (A) and Eso-LM (B). 

<a name="code-organization"></a>
## Code Organization
1. ```main.py```: Routines for training, evaluation, and generating samples
2. ```trainer_base.py```: Base classes for AR and all kinds of discrete diffusion in ```algo.py```
3. ```algo.py```: Classes for AR, MDLM, EsoLM, D3PMAbsorb, SEDDAbsorb and DUO
4. ```dataloader.py```: Dataloaders
5. ```utils.py```: LR scheduler, logging, `fsspec` handling, etc.
6. ```models/```: Denoising network architectures. Supports [DiT](https://arxiv.org/abs/2212.09748), EsoLM-DiT, and AR transformer
7. ```configs/```: Config files for algorithms/datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training, evaluation, and generating samples

<a name="getting_started"></a>

## Getting started in this repository

To get started, create a conda environment containing the required dependencies.

```bash
conda create -n esolm python=3.9
conda activate esolm
pip install -r requirements.txt
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

## Training

Run training as a batch job:

```bash
sbatch scripts/esolm/train_owt_esolmb_alpha0_1.sh
```

Modify `DATA_DIR` and `CHECKPOINT_DIR` accordingly within the bash script.

Logging is done with `wandb`. Configure entity and project in `configs/config.yaml` to your own. 

## Evaluating a checkpoint

Download our Eso-LM (B) checkpoints trained on OpenWebText from this [Google Drive folder](https://drive.google.com/drive/folders/1P14wRCXIjpjjVA26zLf0wXOCfuHwrCCA?usp=sharing).

Run evaluation as a batch job:

```bash
sbatch scripts/esolm/eval_owt_esolmb.sh \
  --alpha_0 1 \
  --batch_split 1 \
  --ckpt_path folder/esolmb-alpha0-1-250k.ckpt
```

By default, this bash script occupies 4 GPUs.

The values of `alpha_0` and `batch_split` used for evaluation must be the same as the ones used for training.

## Sampling from a checkpoint

Download our Eso-LM (B) checkpoints trained on OpenWebText from this [Google Drive folder](https://drive.google.com/drive/folders/1P14wRCXIjpjjVA26zLf0wXOCfuHwrCCA?usp=sharing).

Run sampling as a batch job (generate 8 samples):

```bash
sbatch scripts/esolm/gen_ppl_owt_esolmb.sh \
  --alpha_0 1 \
  --T 1024 \
  --batch_size 8 \
  --num_batches 1 \
  --ckpt_path folder/esolmb-alpha0-1-250k.ckpt
```

By default, this bash script occupies a single GPU.

The value of `alpha_0` used for sampling can be different from the one used for training.

Adjust `batch_size` (must fit on your GPU) and `num_batches` to generate the desired total number of samples.

## Acknowledgements

This repository was built off of [DUO](https://github.com/s-sahoo/duo).

## Citation
```
@misc{sahoo2025esotericlanguagemodels,
      title={Esoteric Language Models}, 
      author={Subham Sekhar Sahoo and Zhihan Yang and Yash Akhauri and Johnna Liu and Deepansha Singh and Zhoujun Cheng and Zhengzhong Liu and Eric Xing and John Thickstun and Arash Vahdat},
      year={2025},
      eprint={2506.01928},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.01928}, 
}
```
