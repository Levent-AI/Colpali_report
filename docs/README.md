### 概览
&emsp;&emsp;在这里，我将运行的部分的代码与我的 PDF 进行了上传，大家酌情选择。


### First
&emsp;&emsp;对于[Medium_text.py](/VScode/self_test/Medium_text.py),代码的参考链接如下：
[Implement Multimodal RAG with ColPali and Vision Language Model Groq(Llava) and Qwen2-VL](
https://medium.com/the-ai-forum/implement-multimodal-rag-with-colpali-and-vision-language-model-groq-llava-and-qwen2-vl-5c113b8c08fd)

里面只有一点需要强调，当初在安装 `flash_attention` 时出现了一些问题，总是安装失败(可能是安装依赖的问题，时间有点长，忘记了)，但是后来在一个项目里面必须要安装 `flash_attention`，解决后，我再返回来安装它，发现没有问题。安装过程如下：


> 1. 首先是去 flash_attention 的 git 网站 ：https://github.com/Dao-AILab/flash-attention/releases/，找到你想安装的版本。
> 2. 检查你当前环境的 CUDA、torch、python 版本，根据显示出来的版本，找到对应的安装包。
![图片](./images/x20.png) 
![图片](./images/x21.png) 
比如说，我现在的版本是 CUDA 11.8 、torch 2.4.0 、python 3.11.6,我安装的是 FALSE ,我当时看大多数博主都安装的 FALSE。
> 3. 下载好后，直接输入 `pip install 下载这个文件的目录地址`，就会自动的安装，这样就成功了。

### Second
&emsp;&emsp;对于代码`gen_colpali_similarity_maps`、`gen_colqwen2_similarity_maps`、`rag_colqwen2_with`基于这个链接里面的内容：https://github.com/tonywu71/colpali-cookbooks
前两个是文献中起到的，看到应该都懂。最后一个比较有意思，对于之前我们是让 `colpali`+`Qwen-VL` 实现 PDF/图片 对于查询的检索并回答，后来发现官方搞了个 `colqwen` 用单个模型来处理整个 RAG 流程,从而节省了需要多个占用大量 VRAM 的模型的资源！😍


注意:源代码是对图片进行测试，但是现实生活中，我们不可能只对一张图片进行检索，所以我用的是 PDF,并且写出了后续的我的想法。
对 PDF 进行问题测试时，发现回答的正确率并不高，所以调整了分辨率进行测试，发现效果不错；后续不满足于一个问题，所以进行了多 Query 多 answer 的实现；后续又实现了多 PDF 多 Query 进行检索。(最后一个的效果还并不是很智能，纯属于好奇想要改进而进行，大家看个乐)
***对于模型 colqwen2-v1.0，这个模型直接下载是没有办法使用的，需要再下载colqwen2-base，将adapter_config.json 里面的路径换成你下载好的路径，这样才能成功的记载模型***

### Third
&emsp;&emsp;我会将我做出来的实验的过程以及结果放入一个 PDF 中，供大家参考，最后还希望大家多多支持，如果有帮助到你的地方，那么这就让我很高兴了！
