# em,很多问题,伙伴们可以帮我改改,改完直接扔ipynb就好(

# 一个项目体验stable-diffusion、waifu、NovelAI三大模型！  

### 原作者:chz061 [点击访问原文](https://aistudio.baidu.com/aistudio/projectdetail/4704437)  

还是那句话,垃圾飞桨! (垃圾百度!)  
  

Stable Diffusion、waifu、NovelAI三大模型及其权重 ，可以使用最基本模型权重组合，也同时可载入自主训练的模型（），模型无限创作无限  
  

**有问题的话, 提Issues,实在不行去原文的评论区留言(**  
  

已知问题  
  

- NovelAI 模型中CFG暂时无效果  
-   自训练模型中negative_prompt暂时无效果  
  
  

# 一个项目体验stable-diffusion、waifu、NovelAI三大模型！

### 原作者:chz061 [点击访问原文](https://aistudio.baidu.com/aistudio/projectdetail/4704437)

还是那句话,垃圾飞桨! (垃圾百度!)

Stable Diffusion、waifu、NovelAI三大模型及其权重 ，可以使用最基本模型权重组合，也同时可载入自主训练的模型（），模型无限创作无限

**有问题的话, 提Issues,实在不行去原文的评论区留言(**

已知问题

-   NovelAI 模型中CFG暂时无效果
-   自训练模型中negative_prompt暂时无效果

那就开始罢(喜)

## 第一步,解压必要文件+安装文件

总共3个部分

- 1.1:拉取项目
- 1.2:解压文件
- 1.3:安装文件

```
#@title 1.1 拉取项目

#从Github上拉取项目,需要连接到云端硬盘

from google.colab import drive

drive.mount('/content/drive')

#连接到云端硬盘

  

!mkdir ./drive/MyDrive/sdwn_files #建个目录

!git clone https://github.com/Wangs-offical/Stable-diffusion-Waifu-NovelAI-Colab.git ./drive/MyDrive/sdwn_files #克隆存储库

```

```
#@title 1.2 解压

  

!unzip /content/drive/MyDrive/sdwn_files/diffusers_paddle.zip -d /content/drive/MyDrive/sdwn_files

!unzip /content/drive/MyDrive/sdwn_files/sd-concepts-library.zip -d /content/drive/MyDrive/sdwn_files

  

#解压两个文件
```

```
#@title 1.3 安装文件

  

print('正在安装库')

!pip install --upgrade pip -i https://mirror.baidu.com/pypi/simple

!pip install -U fastcore paddlenlp ftfy regex --user

!pip install --upgrade paddlenlp

!pip install paddlehub==1.6.0

!pip install common

!pip install dual

!pip install tight

!pip install data

!pip install prox

!pip install paddle

!pip install paddlepaddle
```

## 第二步,重新启动代码执行程序

![](https://raw.githubusercontent.com/Wangs-offical/PictureBed-Wangs/master/2022/10/22/viRbCj1NoUL1s1SI.png)
## 第三步:下载及解压SD预训练模型及waifu模型！

总共一个部分:
- 3.1:下载及解压SD预训练模型及waifu模型

>可能会出现错误,不用管

```
#@title 3.1 下载及解压SD预训练模型及waifu模型

  

import os

if  not os.path.exists("CompVis-stable-diffusion-v1-4"):

!wget https://paddlenlp.bj.bcebos.com/models/community/CompVis/stable-diffusion-v1-4/CompVis-stable-diffusion-v1-4.tar.gz

!tar -zxvf CompVis-stable-diffusion-v1-4.tar.gz -C /content/drive/MyDrive/sdwn_files

!rm -rf CompVis-stable-diffusion-v1-4.tar.gz

if  not os.path.exists("waifu-diffusion"):

!wget http://bj.bcebos.com/paddlenlp/models/community/hakurei/waifu-diffusion/waifu-diffusion.tar.gz #waifu-diffusion-v1-2

!tar -zxvf waifu-diffusion.tar.gz -C /content/drive/MyDrive/sdwn_files

!rm -rf waifu-diffusion.tar.gz

#!wget https://paddlenlp.bj.bcebos.com/models/community/hakurei/waifu-diffusion/waifu-diffusion-v1-3.tar.gz #waifu-diffusion-v1-3 注意下面移动和解压

#!tar -zxvf waifu-diffusion-v1-3.tar.gz -C /content/drive/MyDrive/sdwn_files

#!rm -rf waifu-diffusion-v1-3.tar.gz

# if not os.path.exists("trinart_stable_diffusion_v2_95k"):

# !wget http://bj.bcebos.com/paddlenlp/models/community/naclbit/trinart_stable_diffusion_v2/trinart_stable_diffusion_v2_95k.tar.gz

# !tar -zxvf trinart_stable_diffusion_v2_95k.tar.gz -C /content/drive/MyDrive/sdwn_files

# !rm -rf trinart_stable_diffusion_v2_95k.tar.gz
```
需要10分钟

## 第四步 开始!

下次从这里直接开始!

![](https://raw.githubusercontent.com/Wangs-offical/PictureBed-Wangs/master/2022/10/22/22lVAUK6PSEhQTCA.png)

```
# 模型和权重加载都在这里,pipe 加载的权重,vae_path 加载的本体,如果第一次运行下面报错, 就重启内核重新运行一次

from diffusers_paddle import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

import paddle

from utils import save_image_info

import os

pipe = StableDiffusionPipeline.from_pretrained("./CompVis-stable-diffusion-v1-4") # 加载SD默认权重

#pipe = StableDiffusionPipeline.from_pretrained("./waifu-diffusion") # 加载waifu权重

#pipe = StableDiffusionPipeline.from_pretrained("./NovelAI_latest_ab21ba3c_paddle") # 加载NovelAI权重

# pipe = StableDiffusionPipeline.from_pretrained("./model_pruned_paddle")

  

vae_path = 'stable-diffusion-v1-4/model_state.pdparams'  # 加载 SD默认 模型

#vae_path = 'waifu-diffusion/unet/model_state.pdparams' # 加载 waifu 模型

#vae_path = './NovelAI_latest_ab21ba3c_paddle/vae/animevae.pdparams' # 加载 NovelAI vae 模型

pipe.vae.load_state_dict(paddle.load(vae_path)) # 换用更好的 vae (有效果!)

  

# 图生图

pipe_i2i = StableDiffusionImg2ImgPipeline(vae=pipe.vae,text_encoder=pipe.text_encoder,tokenizer=pipe.tokenizer,

unet=pipe.unet,scheduler=pipe.scheduler,safety_checker=pipe.safety_checker,feature_extractor=pipe.feature_extractor)

print('加载完毕')

```

	
