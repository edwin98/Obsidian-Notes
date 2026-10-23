# Pydantic 数据验证

Pydantic 用于校验输入数据，并将其转换为内部使用的数据模型。

## 0. 技术原理与背景信息
*   **技术原理**：Pydantic 底层利用 Python 类型提示（Type Hints）机制，在运行时对数据进行强制校验与转换。在 V2 版本中，它的核心逻辑使用 Rust 语言重写，通过构建底层的 Schema 校验树（`pydantic-core`）来执行验证，因此校验速度极高。当遇到类型不匹配的数据时，Pydantic 会尝试进行安全的强制类型转换（如将字符串 `"123"` 转为整型 `123`），如果转换失败则抛出包含精确定位信息的错误。
*   **背景信息（为什么需要它）**：前端 JSON 和模型返回的数据可能存在字段缺失、类型错误或结构不符。使用 Pydantic 定义字段、类型和校验规则，可以减少重复的 `if-else` 检查，并在数据进入后续流程前报告错误。业务正确性和权限仍需另外校验。

## 1. 核心作用
*   **请求参数校验**：前端传来的 JSON 格式是否正确、类型是否合法。
*   **配置中心管理**：将 `.env` 或配置文件映射为具备类型提示的对象。
*   **大模型 Schema 约束**：定义大模型结构化输出的 JSON 骨架（如：提取文档中的实体列表）。

## 2. 核心能力
- **类型强制转换**：如果传了字符串 `"10"` 但模型要求 `int`，非严格模式通常可转换，strict 与字段配置会改变行为。
- **自定义校验器 (Validators)**：编写业务规则（如：`top_k` 不能为负数且最大不超过 200）。
- **校验性能**：V2 的核心校验引擎使用 Rust，实际性能取决于数据结构和校验逻辑，不能仅凭实现语言判断所有场景都更快。
- **IDE 支持**：类型提示有助于代码补全和静态检查，减少字段名拼写与类型使用错误。

### V2 的版本与校验边界

官方 2026-10-08 发布 2.14.0，调整 Python 支持范围，并扩展动态模型、部分类型约束和序列化/Schema 配置。这些属于 V2 内的软件演进。[官方变更记录](https://pydantic.dev/docs/validation/latest/get-started/changelog/)

需要区分类型转换、Schema 合法与业务正确。一个对象通过模型校验，不代表来源可信、权限正确或字段之间满足所有业务关系；是否允许强制转换、多余字段和默认值，也取决于模型配置。

序列化与 Schema 变化还会影响工具参数和 API 描述。因此需要注明具体版本与约束，而不能仅凭“使用 Pydantic”推断所有接口一致。历史项目依赖保持原样。关联：[[01_大模型/推理与接口/结构化输出与校验]]。

## 3. 常见用法
- **BaseModel 子类化**：定义业务实体。
- **Field 语义化定义**：通过 `Field` 的 `description` 描述参数含义，这些描述会直接渲染在 Swagger 开发文档上。
- **ConfigDict 配置**：V2 使用 `model_config`；`frozen=True` 禁止字段重新赋值，但不会递归冻结字段内的 list/dict。

## 4. 技术实现示例
```python
from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGQueryRequest(BaseModel):
    """
    RAG 检索请求规范模型
    """

    user_id: str = Field(..., max_length=50, description="唯一用户标识")
    query: str = Field(..., min_length=2, max_length=1000, description="原始提问文本")
    top_k: int = Field(default=80, ge=10, le=200, description="粗筛召回数量")

    # 业务逻辑校验：严禁包含敏感词
    @field_validator("query")
    @classmethod
    def query_must_be_safe(cls, v: str) -> str:
        if "敏感词库" in v:
            raise ValueError("Query contains sensitive terms")
        return v

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
```

## 5. 注意事项与坑点
- **V1 与 V2 的差异**：V2 调整了部分 API，例如从 `@validator` 迁移到 `@field_validator`。项目应统一依赖版本，并根据迁移指南检查兼容性。
- **前向引用**：V2 在需要手动解析未完成的类型引用时使用 `Model.model_rebuild()`；`update_forward_refs()` 是 V1 API。
- **高频处理的额外开销**：在逐行处理大量文本的循环中，重复创建 Pydantic 模型会增加校验开销。可以在输入边界完成校验，再根据内部处理需求选择原生 `dict` 等结构。

## 6. Pydantic 常见面试问题及回答

### Q1: Pydantic 如何验证数据？V2 为什么能改善校验性能？
**回答**：
- **验证原理**：利用 Python 3 引入的类型提示（Type Hints），在对象实例化（即运行时）对传入的数据执行类型推断、强制类型转换与自定义约束验证（Validation）。
- **V2 性能提升原因**：核心校验引擎 `pydantic-core` 使用 Rust，并根据 Schema 执行校验，减少了部分 Python 层处理开销。性能提升幅度需要结合具体数据、校验逻辑和基准测试判断。

### Q2: 如果对方传入了多余的或不在定义中的 JSON 字段，Pydantic 会怎么处理？如何防范参数注入？
**回答**：
- **默认行为**：Pydantic 默认采取极具包容性的 `ignore` 策略。它会将不认识的额外参数直接忽略丢弃，不会报错，也不会纳入最终的实例属性中。
- **额外字段与严格类型分开配置**：`ConfigDict(extra="forbid")` 拒绝未声明字段；`strict=True` 才控制严格类型校验。Pydantic 抛出 `ValidationError`，FastAPI 默认将请求校验失败转换为 HTTP 422，二者不是同一种异常。

### Q3: FastAPI 和 Pydantic 是如何梦幻联动的？
**回答**：
Pydantic 是 FastAPI 的核心数据引擎，两者的结合主要体现在：
1. **自动反序列化与验证**：FastAPI 的路由函数只要标注 Request Body 的类型为 Pydantic 所定义的 `BaseModel` 单例，FastAPI 会自动接管底层的 JSON 解析和 Pydantic 验证工作，失败时统一返回 422 错误格式。
2. **自动生成 OpenAPI (Swagger) 文档**：Pydantic 类中的每一个参数名称、类型约束、以及 `Field(description="...")` 中的中文注释与限制（如 `max_length`、`ge`），都会被 FastAPI 自动抽取并渲染成美观可交互的网页文档，真正实现了“代码即文档”的工程梦魇终结方案。
