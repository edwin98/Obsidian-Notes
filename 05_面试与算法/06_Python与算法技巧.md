---
title: "Python 与算法技巧"
updated: 2026-10-23
---

# Python 与算法技巧

从刷题记录中提取可复用的 Python 写法、堆与链表、二分边界。完成状态统一维护在 [[05_面试与算法/05_LeetCode刷题清单]]。

## 算法面试核心技巧与笔记

### 1. 数组初始化 (The Foundation)

> [!CAUTION] 致命陷阱：浅拷贝
> **绝对不要**使用 `[[0] * n] * m` 进行二维数组的初始化。这会创建 $m$ 个指向同一个列表对象的引用，导致在[[05_面试与算法/05_LeetCode刷题清单#七、回溯]]或[[05_面试与算法/05_LeetCode刷题清单#八、动态规划]]更新状态时，修改一处会连带改变多行（例如 `visited` 数组会发生异常连带变化）。
> 
> ✅ **推荐做法：**
> ```python
> # 初始 m 行 n 列的二维数组
> dp = [[0 for _ in range(n)] for _ in range(m)]
> ```

---

### 2. 哈希表与集合 (Hash & Set)

> [!TIP] 技巧：使用 defaultdict 简化逻辑
> 避免冗长的 `if key not in dict` 初始化操作，减少重复代码。
> ```python
> from collections import defaultdict
> 
> # 自动创建默认值为列表的字典
> ans = defaultdict(list) 
> ans[key].append(value)
> 
> # 自动创建默认值为整数（即 0）的计数器字典
> count = defaultdict(int)
> count[key] += 1
> ```

> [!WARNING] 哈希键 (Key) 的要求
> 字典的 **Key** 必须是**不可变类型 (Hashable)**。
> - `List` ❌ -> `tuple(my_list)` ✅
> - `Dict` ❌ -> `tuple(sorted(my_dict.items()))` ✅
> - `Set` ❌ -> `frozenset(my_set)` ✅

---

### 3. 字符串处理 (Strings)

> [!SUCCESS] 性能优化
> 1. **字符索引转换**：在处理小写英文字母分类问题时，使用 `idx = ord(char) - ord('a')` 是必备操作。
> 2. **高效拼接**：在 Python 中，字符串是**不可变**的，因此使用 `s += "a"` 进行循环拼接会导致时间复杂度达到 $O(n^2)$。
>    - 推荐使用 `"".join(list)` 进行替代（时间复杂度为 $O(n)$）。拼接字符串时，可先将需要拼接的片段存入一个 `list` 中，最后统一调用 `join()` 进行组合。

---

### 4. 搜索与递归 (DFS & BFS)

> [!IMPORTANT] 递归深度与队列
> 1. **高效队列 (BFS)**：实现层序遍历等广度优先搜索场景，建议使用 `collections.deque` 来显著提升性能表现。
>    ```python
>    from collections import deque
>    
>    queue = deque([root])
>    node = queue.popleft() # O(1) 操作，而 list.pop(0) 是 O(n) 操作
>    ```

---

### 5. 常用技巧 Cheat Sheet

| 场景 | 推荐语法 |
| :--- | :--- |
| **同时获取索引和值** | `for idx, val in enumerate(nums):` |
| **同时遍历两个列表** | `for a, b in zip(list1, list2):` |
| **反向遍历** | `for i in range(len(nums) - 1, -1, -1):` |
| **极大/极小整数** | `float('inf')` / `float('-inf')` |
| **向下取整除法** | `5 // 2` (结果为 2) |

---

### 6. LeetCode 提交必备

> [!NOTE] 结果转换
> 即使你在实现逻辑时使用了带有特定方法的 `defaultdict` 或 `deque`，在最终返回结果给 LeetCode 时，往往需要安全地将其转换回标准类型（尤其是当函数签名指定类型为 `List` 的时候）：
> - 返回对应字典的特定集合列表：`return list(ans.values())`
> - 将特定的双端队列实体转化为列表：`return list(my_deque)`

---

## heapq 与多路归并 (Merge K Lists) 核心经验总结

### 1. heapq 核心操作

Python 中的 `heapq` 模块默认实现的是**最小堆 (Min-Heap)**，其堆顶元素始终为 `heap[0]`。

**常用时间复杂度操作：**
```python
import heapq

heapq.heappush(heap, x)      # O(log n)
heapq.heappop(heap)          # O(log n)
heapq.heapify(nums)          # O(n)
```

**高频技巧与写法：**
- **最大堆 (Max-Heap) 的实现**：在入堆和出堆时对需要比较的数值添加负号即可。
  ```python
  heapq.heappush(heap, -val)
  ```
- **TopK 问题快捷方法**：利用 `heapq.nlargest(k, nums)` 或是 `heapq.nsmallest(k, nums)` 快速实现。
- **高效替换与优化限制**：使用 `heapq.heappushpop()` 或者 `heapq.heapreplace()` 来维持堆在固定大小。

> [!WARNING] 堆元素本身可比较的陷阱
> 堆元素需要支持 `<` 比较。若使用 `(priority, node)`，当 `priority` 相同时，Python 会继续比较 `node`；自定义节点未实现比较方法时会报错。
> - **错误写法**：`heappush(heap, (priority, node))`（若 `priority` 相同，则会直接尝试进一步去比较无法进行数值判定操作的 `node`）。
> - **正确写法**：加入唯一的递增 `index`，使优先级相同时比较索引，避免比较节点对象：
>   ```python
>   heappush(heap, (priority, index, node))
>   ```

---

### 2. 链表的核心原则

**拼接链表的关键是修改 `next` 指针。**

> [!info] 区分移动变量与连接节点
> 合并或重排链表时，通常通过修改 `next` 复用已有节点。给局部变量重新赋值，只会改变该变量指向，不会自动建立节点之间的连接。

- **错误示范**：只执行 `cur = node`，并没有将 `node` 接到当前链表后面。
- **正确示范**：
  ```python
  cur.next = node
  cur = cur.next
  ```

**Dummy 虚拟头节点**：统一处理头节点与普通节点，减少空链表等特殊分支：
```python
dummy = ListNode(-1)
cur = dummy

# ... 遍历并连接链表节点 ...
cur.next = node

return dummy.next
```

> [!CAUTION] 高频踩坑操作记录
> - 堆元素包含不可比较的节点时，没有加入唯一索引，可能在优先级相同时报错。
> - 用 `cur = node` 代替 `cur.next = node`，没有真正建立链表连接。
> - 连接节点后忘记执行 `cur = cur.next`，后续操作可能反复覆盖同一位置。

---

### 3. 堆与链表的使用要点
- **堆（`heapq`）**：用于在数据变化时维护最小值、最大值或 TopK，不只是另一种排序方法。
- **链表（`LinkedList`）**：修改链表时，重点检查 `next` 指向是否正确，并保留后续节点的引用。

> [!info] TopK 问题思维导向
> **找最大用最小堆，找最小用最大堆**
> 求最大的 K 个元素时，可以维护一个大小为 K 的**最小堆**。遍历数据，将新元素与堆顶比较：若新元素更大，就替换堆顶；遍历结束后，堆中保留的就是最大的 K 个元素。

---

## 二分法的设置经验整理

### 第一步：根据搜索目标确定循环条件

- **查找指定值**：闭区间模板通常使用 **`while left <= right:`**；命中时可直接 `return mid`，否则排除已经检查的位置。

- **查找边界或收敛到一个位置**：一种常用模板是 **`while left < right:`**，每轮缩小区间，最后检查剩余位置。循环条件需要与区间定义和更新规则配套。

### 第二步：判断 mid 是否仍可能是答案

> [!TIP] 更新边界的关键：
> **只有确认 `mid` 不可能是答案时，才能用 $+1$ 或 $-1$ 将它排除；否则要保留它。**

- `left = mid + 1`：已确认 `mid` 及其左侧不可能是答案。
- `right = mid`：`mid` 仍可能是答案，因此保留它，继续缩小右边界。

### 第三步：检查区间定义与数组访问边界
- 如果使用 `nums[mid + 1]`，必须保证 `mid + 1` 不越界。例如，在 `left < right` 的闭区间模板中，将 `right` 初始化为 **`len(nums) - 1`**，并向下取整计算 `mid`。
- 如果只访问 `nums[mid]`，也要保持区间定义一致：闭区间通常以 `right = len(nums) - 1` 开始，左闭右开区间通常以 `right = len(nums)` 开始，二者的循环和更新规则不能混用。
