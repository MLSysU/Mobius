import threading
import queue
import time

class ThreadManager:
    def __init__(self):
        self.task_queue = queue.Queue()  # 任务队列
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()  # 启动线程
        self.task_lock = threading.Lock()  # 锁，用于跟踪任务状态

    def _worker(self):
        while True:
            task, args, kwargs = self.task_queue.get()  # 从队列中取出任务
            if task is None:  # 退出信号
                break
            # 执行任务前加锁
            with self.task_lock:
                task(*args, **kwargs)  # 执行任务
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
        self.worker_thread.join()

    def wait_for_task_completion(self):
        """
        等待当前任务执行完成。
        """
        time.sleep(0.1)
        # 如果锁被占用，说明有任务在执行
        while self.task_lock.locked():
            time.sleep(0.1)  # 主线程稍作休眠，避免高 CPU 占用


# 示例
'''
def example_task(name, duration):
    print(f"Task {name} started")
    time.sleep(duration)
    print(f"Task {name} completed")

thread_manager = ThreadManager()

try:
    thread_manager.submit_task(example_task, "A", 2)
    thread_manager.submit_task(example_task, "B", 1)
    thread_manager.submit_task(example_task, "C", 3)

    time.sleep(5)  # 主线程做其他事情
finally:
    thread_manager.shutdown()
'''
