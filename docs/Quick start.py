import torch
from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor

# 定义模型路径
model_name = "/hsiam02/huotao/huggingface/models/vidore/colqwen2-v0.1-merged"
# 加载模型
model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # or "mps" if on Apple Silicon
    ).eval()     # .eval() 将模型设置为评估模式
# 加载处理器
processor = ColQwen2Processor.from_pretrained(model_name)

# Your inputs

images = [
    Image.new("RGB", (32, 32), color="white"),
    Image.new("RGB", (16, 16), color="black"),
]
'''
images = [
    Image.open("/hsiam02/huotao/Data/capture_20221115120127257.bmp"),
    Image.open("/hsiam02/huotao/Data/capture_20240408202457252.bmp"),
]
'''
queries = [
    "What does the picture show?",
    "What does the picture show?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)  # 对图像数据批次运行正向传递，获取每个图像的嵌入向量表示形式
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)
# 计算相似度得分






