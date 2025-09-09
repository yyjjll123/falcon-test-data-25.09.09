# falcon-test-data-25.09.09

本代码一共有三个重要文件及一个文件夹，prepare_eval_model.py（配置加速镜像并下载权重文件），run_evaluation.py（复现数据的主要代码），falcon-download_model.py(下载权重文件)和Falcon-project文件夹
首先需要创建Falcon-project文件夹，然后下载相关文件到根目录的Falcon-project文件夹里：

需要下载的文件包括：
从 https://github.com/huggingface/transformers/blob/main/src/transformers/models/falcon/ 下载falcon文件,
从 https://github.com/tiiuae/onebitllms 下载整个文件

在terminal输入以下载模型Falcon-E-1B-Base和Falcon-E-3B-Base：
```sh
python3 /root/lanyun-tmp/save_model_local.py
```
Falcon-project文件夹配置完成

然后需要下载库：
```sh
pip install lm-eval==0.4.2 && pip install evaluate && pip install trl && pip install /root/lanyun-tmp/Falcon-project/onebitllms/onebitllms/

mkdir -p /root/lanyun-tmp/falcon_eval_results   #自定义存放地址
```

随后运行prepare_eval_model.py文件：
```sh
#准备第一个模型权重Falcon-E-1B-BitNet
python /root/lanyun-tmp/prepare_eval_model.py --model_path /root/lanyun-tmp/Falcon-project/Falcon-E-1B-Base --output_dir /root/lanyun-tmp/prepared_models/Falcon-E-1B-BitNet --bitnet
#注意代码地址
```
```sh
#准备第二个模型权重Falcon-E-1B-bf16
python /root/lanyun-tmp/prepare_eval_model.py --model_id tiiuae/Falcon-E-1B-Base --revision bfloat16 --output_dir /root/lanyun-tmp/prepared_models/Falcon-E-1B-bf16
```
```sh
#准备第三个模型权重Falcon-E-3B-BitNet
python /root/lanyun-tmp/prepare_eval_model.py --model_path /root/lanyun-tmp/Falcon-project/Falcon-E-3B-Base --output_dir /root/lanyun-tmp/prepared_models/Falcon-E-3B-BitNet --bitnet
```
```sh
#准备第四个模型权重Falcon-E-3B-bf16
python /root/lanyun-tmp/prepare_eval_model.py --model_id tiiuae/Falcon-E-3B-Base --revision bfloat16 --output_dir /root/lanyun-tmp/prepared_models/Falcon-E-3B-bf16
```
随后运行run_evaluation.py文件：
```sh
python /root/lanyun-tmp/run_evaluation.py
```
