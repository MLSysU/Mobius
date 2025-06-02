// wrapper.cpp (C++ 侧实现)
#include <torch/script.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11; 
class OffloadThread {
private:
    std::queue<std::pair<torch::jit::script::Module, torch::jit::script::Module>> task_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    std::thread worker;
    std::atomic<bool> running{false};
    c10::cuda::CUDAStream offload_stream;

    void worker_loop() {
        c10::cuda::CUDAGuard device_guard(0);  // 确保在正确的 GPU 设备
        c10::cuda::CUDAStreamGuard stream_guard(offload_stream);
        
        while (running || !task_queue.empty()) {
            std::pair<torch::jit::script::Module, torch::jit::script::Module> task;
            
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [&]{ return !task_queue.empty() || !running; });
                
                if (task_queue.empty()) continue;
                
                task = std::move(task_queue.front());
                task_queue.pop();
            }

            // 执行实际卸载逻辑
            auto& model = task.first;
            auto& cpu_model = task.second;
            
            {
                torch::AutoGradMode enable_grad(false);  // 禁用梯度计算
                for (const auto& param : model.named_parameters()) {
                    auto& tensor = param.value;
                    if (tensor.is_cuda()) {
                        // 异步释放 GPU 内存
                        auto& mutable_tensor = const_cast<torch::Tensor&>(param.value);
                        mutable_tensor.reset();
                        // auto new_tensor = tensor.cpu().detach();
                        // tensor.set_data(new_tensor);
                        
                        // 梯度拷贝到 CPU
                        if (tensor.grad().defined()) {
                            auto cpu_param = cpu_model.attr(param.name).toTensor();
                            cpu_param.grad().copy_(tensor.grad().cpu(), true);
                            tensor.mutable_grad().reset();
                        }
                    }
                }
            }
            
            offload_stream.synchronize();
        }
    }

public:
    OffloadThread() : offload_stream(c10::cuda::getStreamFromPool()) {
        running = true;
        worker = std::thread(&OffloadThread::worker_loop, this);
    }

    ~OffloadThread() {
        running = false;
        cv.notify_all();
        if (worker.joinable()) worker.join();
    }

    void submit(const torch::jit::Module& model, const torch::jit::Module& cpu_model) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_queue.emplace(std::move(model), std::move(cpu_model));
        }
        cv.notify_one();
    }
};

// 全局单例（线程安全初始化）
OffloadThread& get_offloader() {
    static OffloadThread offloader;
    return offloader;
}

PYBIND11_MODULE(parallel, m) {
    py::class_<OffloadThread>(m, "OffloadThread")
        .def(py::init<>())
        // .def("submit", &OffloadThread::submit);
        .def("submit", [](OffloadThread& self, const torch::jit::Module& model, const torch::jit::Module& cpu_model) {
            self.submit(model, cpu_model);
        });
    
    m.def("get_offloader", &get_offloader, py::return_value_policy::reference);
}