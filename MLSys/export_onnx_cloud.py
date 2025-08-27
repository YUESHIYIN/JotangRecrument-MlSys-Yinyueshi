import os
import shutil
from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer
from modelscope.hub.snapshot_download import snapshot_download

# --- 1. 使用 ModelScope 获取模型的安全本地路径 ---
model_id = "Qwen/Qwen1.5-1.8B-Chat"
print(f"====================================================================")
print(f"步骤 1/3: 正在确认模型 '{model_id}' 的本地路径...")
print(f"====================================================================")

try:
    local_model_path = snapshot_download(model_id, cache_dir='./model_cache')
    print(f"✅ 模型已就绪，本地路径为: {local_model_path}")
except Exception as e:
    print(f"❌ 获取模型路径失败！错误: {e}")
    exit()

# --- 2. 定义你指定的 ONNX 输出路径 ---
onnx_output_path = "./llama-onnx"  # <--- 修改在这里！

# 在导出前，先清理一下目标文件夹，防止旧文件干扰
if os.path.exists(onnx_output_path):
    print(f"\n警告：发现已存在的输出目录 '{onnx_output_path}'，将进行清理。")
    shutil.rmtree(onnx_output_path)
print(f"\nONNX 模型将保存到你指定的目录: {onnx_output_path}")


# --- 3. 从本地路径导出模型 ---
print(f"\n====================================================================")
print(f"步骤 2/3: 开始从本地路径将模型导出为 ONNX...")
print(f"====================================================================")
try:
    main_export(
        model_name_or_path=local_model_path,
        output=onnx_output_path,
        task="text-generation-with-past",
        do_constant_folding=True,
        trust_remote_code=True,
    )
    print(f"\n✅ 模型成功导出为 ONNX 格式到: {onnx_output_path}")

except Exception as e:
    print(f"\n❌ 模型导出失败！错误: {e}")
    exit()

# --- 4. 保存分词器 ---
print(f"\n====================================================================")
print(f"步骤 3/3: 正在保存分词器...")
print(f"====================================================================")
try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
    tokenizer.save_pretrained(onnx_output_path)
    print(f"✅ 分词器成功保存到: {onnx_output_path}")
    print(f"\n🎉🎉🎉 恭喜！所有步骤已成功完成！🎉🎉🎉")

except Exception as e:
    print(f"\n❌ 分词器保存失败！错误: {e}")