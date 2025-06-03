import torch
import multiprocessing
import queue
import time
from multiprocessing import Process, Queue, Lock

# Prefetch and Offload Task Submission
class ProcessManager:
    def __init__(self):
        self.task_queue = Queue()  # 使用 multiprocessing.Queue
        self.worker_process = Process(target=self._worker)  # 创建一个进程来执行任务
        self.worker_process.start()  # 启动进程
        self.task_lock = Lock()  # 锁，用于跟踪任务状态

    def _worker(self):
        while True:
            task, args, kwargs = self.task_queue.get()  # 从队列中取出任务
            if task is None:  # 退出信号
                break
            # 执行任务前加锁
            self.task_lock.acquire()
            try:
                task(*args, **kwargs)  # 执行任务
            except Exception as e:
                print(f"Error in task: {e}")
            self.task_lock.release()
            # 任务完成后，锁自动释放

    def submit_task(self, task, *args, **kwargs):
        """
        提交任务到线程。
        """
        self.task_queue.put((task, args, kwargs))

    def shutdown(self):
        """
        停止线程。
        """
        self.task_queue.put((None, None, None))  # 提交退出任务
        self.worker_process.join()

    def wait_for_task_completion(self):
        self.task_lock.acquire()
        self.task_lock.release()