# Pydantic 数据验证

Pydantic 是本项目逻辑层的“护城河”，负责将凌乱的非结构化入参校验为严谨的内部对象模型。

## 0. 技术原理与背景信息
*   **技术原理**：Pydantic 底层利用 Python 类型提示（Type Hints）机制，在运行时对数据进行强制校验与转换。在 V2 版本中，它的核心逻辑使用 Rust 语言重写，通过构建底层的 Schema 校验树（`pydantic-core`）来执行验证，因此校验速度极高。当遇到类型不匹配的数据时，Pydantic 会尝试进行安全的强制类型转换（如将字符串 `"123"` 转为整型 `123`），如果转换失败则抛出包含精确定位信息的错误。
*   **背景信息（为什么需要它）**：在传统的 Python 后端开发中，前端传来的 JSON 数据通常会被解析成了原生的字典（`dict`）。开发者需要在业务代码中编写大量繁琐且容易遗漏的 `if-else` 来判断字段是否存在、类型对不对。在 RAG 系统中，我们需要处理用户极其灵活的输入，以及从大语言模型（LLM）返回的半结构化 JSON 数据。如果没有一套严谨的数据清洗与验证护城河，这些“脏数据”会立刻弄乱系统的内部状态并导致服务崩溃。

## 1. 核心作用
*   **请求参数校验**：前端传来的 JSON 格式是否正确、类型是否合法。
*   **配置中心管理**：将 `.env` 或配置文件映射为具备类型提示的对象。
*   **大模型 Schema 约束**：定义大模型结构化输出的 JSON 骨架（如：提取文档中的实体列表）。

## 2. 核心能力
- **类型强制转换**：如果传了字符串 `"10"` 但模型要求 `int`，Pydantic 会自动转换。
- **自定义校验器 (Validators)**：编写业务规则（如：`top_k` 不能为负数且最大不超过 200）。
- **极高性能**：其核心基于 Rust (V2版本)，是 Python 领域最快的数据校验库。
- **与 IDE 深度集成**：提供完美的代码补全体验，大幅减少“拼写错误”导致的低级 Bug。

## 3. 常见用法
- **BaseModel 子类化**：定义业务实体。
- **Field 语义化定义**：通过 `Field` 的 `description` 描述参数含义，这些描述会直接渲染在 Swagger 开发文档上。
- **Config 配置**：使用 `frozen=True` 创建不可变对象，确保在复杂的 RAG 调度链路中参数不被中途篡改。

## 4. 技术实现示例
```python
from pydantic import BaseModel, Field, field_validator

class RAGQueryRequest(BaseModel):
    """
    RAG 检索请求规范模型
    """
    user_id: str = Field(..., max_length=50, description="唯一用户标识")
    query: str = Field(..., min_length=2, max_length=1000, description="原始提问文本")
    top_k: int = Field(default=80, ge=10, le=200, description="粗筛召回数量")
    
    # 业务逻辑校验：严禁包含敏感词
    @field_validator('query')
    @classmethod
    def query_must_be_safe(cls, v: str) -> str:
        if "敏感词库" in v:
            raise ValueError('Query contains sensitive terms')
        return v
    
    class Config:
        frozen = True # 设置为不可变对象，保障业务链横向传递安全
        extra = "forbid" # 禁止传入未定义的额外参数，防止参数注入攻击
```

## 5. 注意事项与坑点
- **V1 与 V2 的差异**：Pydantic V2 进行了重构，很多装饰器从 `@validator` 变成了 `@field_validator`。本项目由于追求极致性能，建议统一使用 **V2.x**。
- **Lazy Evaluation 坑**：在复杂的嵌套模型中，有时会出现递归引用的问题，需使用 `Model.update_forward_refs()` 显式解决。
- **性能过度损耗**：在极其频繁的内循环中（如处理千万行文本行）使用 Pydantic 会有一定 overhead。对于纯原始数据处理，应优先使用原生 `dict`。

## 6. Pydantic 常见面试问题及回答

### Q1: Pydantic 的验证原理是什么？在 V2 版本中为什么性能得到了极大提升？
**回答**：
- **验证原理**：利用 Python 3 引入的类型提示（Type Hints），在对象实例化（即运行时）对传入的数据执行类型推断、强制类型转换与自定义约束验证（Validation）。
- **V2 性能提升原因**：在 Pydantic V1 中，验证逻辑是使用纯 Python 编写的，存在一定的性能损耗。在 V2 中，其底层用于进行核心 Schema 校验的引擎（`pydantic-core`）彻底使用 **Rust 语言重写**。由于 Rust 作为底层编译语言的极速性能，Pydantic V2 较 V1 获得了 5 倍至 50 倍的性能突破，成为了数据处理的工业级标杆。

### Q2: 如果对方传入了多余的或不在定义中的 JSON 字段，Pydantic 会怎么处理？如何防范参数注入？
**回答**：
- **默认行为**：Pydantic 默认采取极具包容性的 `ignore` 策略。它会将不认识的额外参数直接忽略丢弃，不会报错，也不会纳入最终的实例属性中。
- **防范注入（严格模式）**：在企业级开发中，为了防止恶意用户构造多余参数进行试探或污染，我们必须在模型的内部配置类中使用 `Config` 开启严格模式：`extra = "forbid"`。这样一旦前端传入了未在 schema 上的字段，Pydantic 会立刻阻断请求并抛出 422 Unprocessable Entity 的数据验证异常。

### Q3: FastAPI 和 Pydantic 是如何梦幻联动的？
**回答**：
Pydantic 是 FastAPI 的核心数据引擎，两者的结合主要体现在：
1. **自动反序列化与验证**：FastAPI 的路由函数只要标注 Request Body 的类型为 Pydantic 所定义的 `BaseModel` 单例，FastAPI 会自动接管底层的 JSON 解析和 Pydantic 验证工作，失败时统一返回 422 错误格式。
2. **自动生成 OpenAPI (Swagger) 文档**：Pydantic 类中的每一个参数名称、类型约束、以及 `Field(description="...")` 中的中文注释与限制（如 `max_length`、`ge`），都会被 FastAPI 自动抽取并渲染成美观可交互的网页文档，真正实现了“代码即文档”的工程梦魇终结方案。
