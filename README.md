# TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models

<div align="center">    
    <p style="font-size: 45px;"> by 
        <a href="https://pooyanrg.me">Pooyan Rahmanzadehgervi</a><sup>1</sup>, 
        <a>Hung Huy Nguyen</a><sup>1</sup>, 
        <a href="https://rosanneliu.com/">Rosanne Liu</a><sup>2,3</sup>, 
        <a href="https://mai-t-long.com/">Long Mai</a><sup>4</sup>, 
        <a href="https://anhnguyen.me/research/">Anh Totti Nguyen</a><sup>1</sup>
    </p>
    <p>
        <sup>1</sup>Auburn University, 
        <sup>2,3</sup>Google DeepMind, ML Collective,
        <sup>4</sup>Adobe Research
    </p>

    
<!-- [![Website](http://img.shields.io/badge/Website-4b44ce.svg)](https://vlmsareblind.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-2407.06581-b31b1b.svg)](https://arxiv.org/abs/2407.06581)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-red)](https://huggingface.co/datasets/XAI/vlmsareblind) -->
    
</div>

<!-- This repository contains the code and data for the paper `Vision Language Models Are Blind`.

    @article{vlms2024blind,
      title={Vision language models are blind},
      author={Rahmanzadehgervi, Pooyan and Bolton, Logan and Taesiri, Mohammad Reza and Nguyen, Anh Totti},
      journal={arXiv preprint arXiv:2407.06581},
      year={2024}
    } -->





# TAB

Official implementation of paper **TAB: Transformer Attention Bottlenecks enable User Intervention and
Debugging in Vision-Language Models**

## Requirements

```sh
conda env create -f env.yml
conda activate cab
```


## Data Preparing

**For CLEVR-Change**

The official data can be found here: [google drive link](https://drive.google.com/file/d/1HJ3gWjaUJykEckyb2M0MB4HnrJSihjVe/view) provided by [Robust Change Captioning (ICCV19)](https://github.com/Seth-Park/RobustChangeCaptioning). 

Extracting this file will create data directory.

```sh
tar -xzvf clevr_change.tar.gz
```

For the convenience, you can also download the three json files from [link](https://drive.google.com/drive/folders/1g8QD6Y3La8cIamE7jeSSlXTw8G3r5Q8o?usp=sharing).

You would get

```
your_data_path
|–– clevr_change/
|   |–– data/
|   |   |–– images/
|   |   |–– nsc_images/
|   |   |–– sc_images/
|   |   |–– sc_images/
|   |   |–– change_captions.json
|   |   |–– no_change_captions.json
|   |   |–– splits.json
|   |   |–– type_mapping.json
```

Download CLIP (ViT-B/32 and ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

