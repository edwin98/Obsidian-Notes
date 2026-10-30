"""【学习定位：SQLite与消息幂等】
tasks保存聚合JSON，messages保存去重与回应。同一消息重传不能再次调度。锁和连接只在一个进程内有效。
对应05、14；调用方管理事务，数据库记录才是持久事实。

SQLite 保存新版任务与消息去重记录；连接只在进程内共享。
"""

import json
import sqlite3
from threading import RLock

from task_models import TaskRecord


class ConflictError(ValueError):
    """消息内容、目标或任务版本冲突。"""


class TaskStore:
    def __init__(self, path: str = ":memory:"):
        """同一连接由服务锁串行使用；RLock 允许服务内部嵌套读取。

        check_same_thread=False 只解除线程限制，不自动提供并发安全。
        busy_timeout 让短暂数据库占用可以等待，不是分布式任务锁。
        """
        self.lock = RLock()
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.executescript(
            "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, payload TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS messages (conversation TEXT, id TEXT, "
            "request TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY(conversation,id));"
        )
        self.connection.commit()

    def save(self, task: TaskRecord) -> None:
        """在调用方事务中保存聚合对象；写入本身不擅自commit。"""
        self.connection.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?, ?)",
            (task.task_id, task.model_dump_json()),
        )

    def get(self, task_id: str) -> TaskRecord:
        """反序列化得到独立对象，避免只改内存却误以为已持久化。"""
        row = self.connection.execute(
            "SELECT payload FROM tasks WHERE id=?", (task_id,)
        ).fetchone()
        if row is None:
            raise KeyError(task_id)
        return TaskRecord.model_validate_json(row[0])

    def list(self, conversation: str | None = None) -> list[TaskRecord]:
        """按登记顺序读取副本，再筛选对话；教学数据量小，暂不额外建索引列。"""
        tasks = [
            TaskRecord.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT payload FROM tasks ORDER BY rowid"
            )
        ]
        return [
            t
            for t in tasks
            if conversation is None or t.conversation_id == conversation
        ]

    def replay(self, request: dict) -> dict | None:
        """消息重传可重放回应，但run_requested必须为False，防止重复调度。"""
        row = self.connection.execute(
            "SELECT request,response FROM messages WHERE conversation=? AND id=?",
            (request["conversation_id"], request["message_id"]),
        ).fetchone()
        if not row:
            return None
        if json.loads(row[0]) != request:
            raise ConflictError("同一消息编号不能用于不同内容")
        return {**json.loads(row[1]), "duplicate": True, "run_requested": False}

    def remember(self, request: dict, response: dict) -> None:
        """与任务修改共用事务，避免消息记已处理而任务尚未保存。"""
        self.connection.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?)",
            (
                request["conversation_id"],
                request["message_id"],
                json.dumps(request, ensure_ascii=False),
                json.dumps(response, ensure_ascii=False),
            ),
        )

    def commit(self) -> None:
        """显式结束手动事务；常规服务写入使用 connection 上下文自动提交。"""
        self.connection.commit()

    def close(self) -> None:
        """释放连接；评测每个 episode 都关闭自己的独立存储。"""
        self.connection.close()
