
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 设置Hugging Face的国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 3B模型的Hugging Face Hub ID
#model_id = "tiiuae/Falcon-E-1B-Base"
model_id = "tiiuae/Falcon-E-3B-Base"

print(f"使用镜像: {os.environ.get('HF_ENDPOINT')}")
print(f"开始下载模型: {model_id}")
print("这可能需要一些时间，具体取决于您的网络速度和模型大小。")

# from_pretrained 方法会自动下载并缓存模型
# Falcon模型需要设置 trust_remote_code=True 参数
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, # 如果可能，使用bfloat16以节省内存
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("\n" + "="*50)
    print(f"成功下载并加载模型: {model_id}")
    print(f"模型已缓存至您的Hugging Face缓存目录中。")
    print("您现在可以在脚本中直接使用此模型ID。")
    print("重要提示：为了让之后的所有操作（包括 lm-evaluation-harness）都通过镜像下载，")
    print("请在运行其他脚本前，在终端执行: export HF_ENDPOINT=https://hf-mirror.com")
    print("="*50)

except Exception as e:
    print(f"\n发生错误: {e}")
    print("请检查您的网络连接，并确认已安装必要的库 (如 transformers, torch)。")
    print("如果问题仍然存在，可能是镜像服务器暂时不可用，或模型不在该镜像上。")