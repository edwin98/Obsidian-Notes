# Pydantic 数据验证

Pydantic 是本项目逻辑层的“护城河”，负责将凌乱的非结构化入参校验为严谨的内部对象模型。

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
