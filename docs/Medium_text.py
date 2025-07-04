from byaldi import RAGMultiModalModel # 用于多模态检索和语言生成的模型
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info # 在 Qwen 框架中处理图像或视频的实用程序
import torch
from pdf2image import convert_from_path  # 用于将 PDF 转换为图像的实用程序
import groq
import os
# 使用指定的预训练模型加载 RAGMultiModalModel，从而启用检索增强生成
RAG = RAGMultiModalModel.from_pretrained("/huggingface/models/vidore/colpali-v1.2-merged")
# 加载LLM
model_name = "/huggingface/models/Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
# 进行检索
RAG.index(input_path="/Data/PDF/2023tj.pdf",
          index_name="multimodal_rag",
          store_collection_with_index=False,
          overwrite=True,)
# text_query = "What is the type of star hosting thge kepler-51 planetary system?"

text_query = "天津市2023年全市参加职工基本医疗保险人数是多少?"
# text_query="Colpali模型相比于其他所有模型的最大优势体验在哪个数据集上？提升幅度是多少？"

# text_query = "Colpali模型如何利用视觉模型（VLMs）来进行文档改进，它相比传统方法有哪些优势？"

# text_query = "Is the Pairwise CE loss best?"

# text_query  = "What is the age of the star hosting the kepler-51 planetary system?"
# 在已编入索引的 PDF 中搜索与 text_query相关的前 k=3 个结果
results = RAG.search(text_query,k=3)
print(results)

# 加载 Qwen2-VL 模型的 AutoProcessor 以处理多模态输入
processor = AutoProcessor.from_pretrained("/huggingface/models/Qwen/Qwen2-VL-7B-Instruct")
# processor = AutoProcessor.from_pretrained("/huggingface/models/Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

# images = convert_from_path("/Data/input.pdf")
images = convert_from_path("/Data/PDF/2023tj.pdf")

image_index = results[0]["page_num"] -1
# from IPython.display import Image,display
# display(images[image_index])
# display(images[1])

# images[image_index].save('/Data/image/jpg/image3.jpg')


# messages = [
#     {"role":"user",
#      "content":[{"type":"image",
#                  "image":images[image_index]
#                  },
#                 {"type":"text","text":text_query}
#               ]
#     }
#             ]

# #
# text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# #
# image_inputs,video_inputs = process_vision_info(messages)
# #
# inputs = processor(text=[text],
#                    images=image_inputs,
#                    videos=video_inputs,
#                    padding=True,
#                    return_tensors="pt")
# inputs = inputs.to("cuda")
# #
# generate_ids = model.generate(**inputs, 
#                               max_new_tokens=256)
# #
# generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
# #
# output_text = processor.batch_decode(generated_ids_trimmed, 
#                                      skip_special_tokens=True,
#                                      clean_up_tokenization_spaces=False)
# #
# print(output_text[0])








# 准备聊天消息
messages = [
    {"role":"system",
     "content":"你是一位专精于多模态的大模型，你主要是根据查询对PDF进行分析，由文本和图片给出理想的结果。"    
    },
    {"role":"user",
     "content":[{"type":"image",
                 "image":images[image_index]
                 },
                {"type":"text","text":text_query}
              ],
    }
            ]
 
# 为生成设置文本格式
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# 处理视觉输入
image_inputs,video_inputs = process_vision_info(messages)
# 创建输入张量
inputs = processor(text=[text],
                   images=image_inputs,
                   videos=video_inputs,
                   padding=True,
                   return_tensors="pt")
inputs = inputs.to("cuda")
# 生成 ID
generate_ids = model.generate(**inputs, 
                              max_new_tokens=256)

# 输出
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generate_ids)]
# 解码输出文本
output_text = processor.batch_decode(generated_ids_trimmed, 
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
# 打印输出
# print(output_text[0])
print(output_text)
