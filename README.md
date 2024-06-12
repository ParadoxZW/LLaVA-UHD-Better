# LLaVA-UHD-Better

An all-bugs-fixed (I hope so), cleaner and improved implementation of LLaVA-UHD, based on the code from the official repo, [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD).

## Why this repo exists?

The study of [LLaVA-UHD](https://arxiv.org/pdf/2403.11703) is a great work for the community of LMMs.
However, the official-released repo is full of bugs, wrong logic, uncleared/uncommented debugging codes, and missing but necessary procedures. For some examples, it wrongly deal with the patch numbers of the slices and the overall image in a batch, wrongly calculate `pos_embed` and `attention_mask` in `resampler`, misses the separate tokens between sub-images, and misses the procedures of normalizing the image according CLIP's preprocessing, etc. And many people have raised problems related to [this issue](https://github.com/thunlp/LLaVA-UHD/issues/5). It seems the official team will not fix these issues in the near future.

Thus, this repo aims to address these issues and improve the overall quality of the implementation, following the spirit of the paper [LLaVA-UHD](https://arxiv.org/pdf/2403.11703).

## Works I have done

- cleaned up the file structure and the code
- fixed the bugs
- supplemented the missing procedures
- improved the implementation of some important functions
- a little modification to the architecture definition of the `Resampler`

You can find details about the major modifications by searching the comments like `[Edited by zhenwei - 2024-xx-xx xx:xx]` in the code. I have left detailed comments there to explain what I have changed and why (mostly in Chinese though 😊).

## How to use this repo?

The preparation of the environment and datasets follows [LLaVA](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#install). Run the bash script to start the training:
```BASH
bash scripts/train.sh
```

**Note:** dependency packages of higher versions may lead problems. For example, you may meet `token mismatch` warning at stage-2. Commenting out the line `use_fast=False` of the tokenizer may help.

## Checkpoints and evaluation results

It is ongoing... I will release checkpoints as soon as I reproduce the results of a satisfactory performance.

## TODO

- [] add codes for model evaluation on common benchmarks
- [] release reproducible checkpoint
- [] add logic for simultaneously fine-tuning the vision encoder when stage-2

## Acknowledgements

This repo is based on the official repo [LLaVA-UHD](https://github.com/thunlp/LLaVA-UHD). I would like to express my profound gratitude and respect for their valuable work.

## Citation

If you find this repo useful for your research and applications, please cite:
```bibtex
@misc{llava-uhd-better,
  author = {Shao Zhenwei},
  title = {LLaVA-UHD-Better},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/ParadoxZW/LLaVA-UHD-Better},
  note = {GitHub repository},
}

@article{xu2024llava-uhd,
  title={{LLaVA-UHD}: an LMM Perceiving Any Aspect Ratio and High-Resolution Images},
  author={Xu, Ruyi and Yao, Yuan and Guo, Zonghao and Cui, Junbo and Ni, Zanlin and Ge, Chunjiang and Chua, Tat-Seng and Liu, Zhiyuan and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11703},
  year={2024}
}
```