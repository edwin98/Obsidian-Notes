# [[RAG项目|无线豆包]]
- [x] 将无线豆包的流程全部写进去
	- [ ] 补充框架图
- [ ] 让gpt帮忙匹配合适的技术栈
- [ ] 针对每个技术栈，详细了解具体的细节
- [ ] 具体的细节，可以与[[技术八股]]中结合
- [ ] 技术八股涉及到的东西，最基本在本地实操一下，有个印象
# [[leetcode]]
根据gpt整理的内容，把题针对性刷一遍，不用花太多时间

# agent
搜索一下应该和什么套用
# 大模型基础知识
- [ ] PPO\GRPO的区别
- [ ] 读deepseek的技术报告
- [ ] vllm的核心原理：PageAttention
- [ ] Slide Window attention
- [x] RoPE[[杂七杂八技术#RoPE编码]]
- [x] CoT ToT GoT[[杂七杂八技术#CoT、ToT、GoT]]
- [x] 多模态具体是怎么实现的[[杂七杂八技术#多模态的实现]]
- [ ] agentic RL的范围、实现、效果
- [ ] LangChain、LammaIndex、LangGraph框架学习
- [ ] SFT\RLHF\RL
- [x] RAG中，为什么使用RSF而非RRF
- [ ] 实现模型评价的方式-LangSmith？DeepEval？
- [ ] 整个流程中，各部门或各人员是怎么分工的
- [ ] agent的评估中，是否应该跑pass@3来判断任务是否实现
- [ ] RAG系统为了提高效率做了哪些工作
- [ ] RAG中负责：embedding的微调数据集构建、微调；reranker的数据集构建、微调；chunk设计、召回链路设计；评估系统的设计、评估数据集的构建；召回系统的设计、chunk算法的设计
- [ ] agent中负责：

# 需要精通的地方
- 无线豆包的评估、召回、切分的设计、embedding和reranker的微调
- agent的安全机制设计、评估设计、rag库的设计、审计、整个流程的设计
- 经典的论文讲解：rag、模型评估、deepseek R1、agent安全

面试的问题
chunk的时候，表格的处理方式，不同版本文档的处理方式
RAG系统为了提高效率做了哪些工作
整个流程中，各部门或各人员是怎么分工的
infoNCEloss的原理
embedding微调和reranker微调是用的什么框架
reranker的工作方式，内部是什么结构？和embedding模型的区别？
agent的正常工作流程是什么
文档的量级
agent调用工具的时候，工具的设计和输入输出参数