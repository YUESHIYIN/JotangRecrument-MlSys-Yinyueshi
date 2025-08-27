import torch
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM # <--- 关键！导入官方的“ONNX驾驶舱”

# --- 1. 定义模型路径 ---
# 这个路径指向包含了 model.onnx 和所有配置文件的文件夹
onnx_model_directory = "./llama-onnx"

print("正在使用 Optimum 加载 ONNX 模型和分词器...")

# --- 2. 使用 Optimum 的“智能驾驶舱”加载模型 ---
try:
    tokenizer = AutoTokenizer.from_pretrained(onnx_model_directory)
    # ORTModelForCausalLM 会自动加载 onnx 文件，并配置好 session
    # use_cache=True 表示我们要使用 past_key_values 来加速
    # provider='CUDAExecutionProvider' 明确指定使用 GPU
    model = ORTModelForCausalLM.from_pretrained(
        onnx_model_directory, 
        use_cache=True, 
        provider='CUDAExecutionProvider'
    )
    print("✅ 模型和分词器加载成功！")
except Exception as e:
    print(f"❌ 加载失败！错误信息: {e}")
    exit()


# --- 3. 编写一个更简洁、更强大的推理函数 ---
def generate_text(prompt, max_length=50):
    print(f"\n输入 Prompt: '{prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful chatbot."},
        {"role": "user", "content": prompt},
    ]
    
    # 分词器处理输入
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda") # <-- 关键！把输入数据移动到 GPU 上

    print("正在 GPU 上使用 Optimum .generate() 方法生成文本...")

    # --- 核心改变！我们现在可以使用和原生 Transformers 完全一样的 .generate() 方法！---
    # “驾驶舱”会自动处理所有 attention_mask, past_key_values 等细节
    output_ids = model.generate(input_ids, max_length=max_length)
    
    # 解码并打印
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    print("\n--- 模型输出 ---")
    print(generated_text)
    print("----------------")


# --- 4. 运行示例 ---
generate_text("What is the capital of France?")
generate_text("Write a short story about a robot who discovers music.")