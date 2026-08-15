// Timing harness, C++ / LibTorch side.
//
// Mirrors python/harness.py invariant-for-invariant:
//   * CUDA events for timing, never wall-clock around async calls.
//   * Warmup epochs / steps excluded from every statistic.
//   * Median + IQR for throughput; full percentile distribution for latency.
//   * Synchronise only at measurement boundaries.
//
// Emits JSON to stdout with the same schema as the Python side so a single
// aggregator (scripts/aggregate.py) consumes both. Config is read from the
// shared config/experiment.json.
//
// Build via CMake (see cpp/CMakeLists.txt). Requires the same libtorch that
// ships with the Python wheel used on the other side -- see README.

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "resnet.h"
#include "json_mini.h"   // tiny header-only JSON reader/writer (see include/)

namespace {

struct Config {
    long seed = 42;
    std::string model = "resnet18";
    int num_classes = 10;
    bool cifar_stem = true;
    int batch_size = 128;
    double lr = 0.1, momentum = 0.9, weight_decay = 5e-4;
    std::string precision = "fp32";
    bool cudnn_benchmark = true;

    int warmup_steps = 50;
    int measured_steps = 200;
    int lat_warmup = 500;
    int lat_iters = 5000;
    int lat_batch = 1;
    int thr_batch = 512;
};

Config load_config(const std::string& path) {
    Config c;
    JsonValue j = json_parse_file(path);
    c.seed = j["seed"].as_int();
    c.model = j["model"].as_string();
    c.num_classes = j["num_classes"].as_int();
    c.cifar_stem = j["cifar_stem"].as_bool();
    c.batch_size = j["train"]["batch_size"].as_int();
    c.lr = j["train"]["lr"].as_double();
    c.momentum = j["train"]["momentum"].as_double();
    c.weight_decay = j["train"]["weight_decay"].as_double();
    c.precision = j["precision"].as_string();
    c.cudnn_benchmark = j["cudnn_benchmark"].as_bool();
    c.warmup_steps = j["harness"]["warmup_steps_per_epoch"].as_int();
    if (j["harness"].has("measured_steps"))
        c.measured_steps = j["harness"]["measured_steps"].as_int();
    c.lat_warmup = j["harness"]["inference_warmup_iters"].as_int();
    c.lat_iters = j["harness"]["inference_latency_iters"].as_int();
    c.lat_batch = j["harness"]["latency_batch_size"].as_int();
    c.thr_batch = j["harness"]["throughput_batch_size"].as_int();
    return c;
}

// CUDA-event timer; wall-clock fallback on CPU for local logic tests.
struct GpuTimer {
    bool cuda;
    at::cuda::CUDAEvent start_ev, end_ev;  // only used when cuda
    std::chrono::high_resolution_clock::time_point t0;

    explicit GpuTimer(bool is_cuda) : cuda(is_cuda) {}

    void start() {
        if (cuda) start_ev.record();
        else t0 = std::chrono::high_resolution_clock::now();
    }
    double stop_ms() {
        if (cuda) {
            end_ev.record();
            end_ev.synchronize();
            return start_ev.elapsed_time(end_ev);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n == 1) return v[0];
    double rank = p / 100.0 * (n - 1);
    size_t lo = static_cast<size_t>(rank);
    double frac = rank - lo;
    size_t hi = std::min(lo + 1, n - 1);
    return v[lo] * (1 - frac) + v[hi] * frac;
}

void apply_precision(const Config& c) {
    at::globalContext().setBenchmarkCuDNN(c.cudnn_benchmark);
    if (c.precision == "fp32") {
        at::globalContext().setAllowTF32CuBLAS(false);
        at::globalContext().setAllowTF32CuDNN(false);
    } else if (c.precision == "tf32") {
        at::globalContext().setAllowTF32CuBLAS(true);
        at::globalContext().setAllowTF32CuDNN(true);
    }
    // bf16_amp handled at the autocast call site.
}

}  // namespace

int main(int argc, char** argv) {
    std::string config_path = "config/experiment.json";
    std::string arm, out_path;
    int repeat_index = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a == "--arm" && i + 1 < argc) arm = argv[++i];
        else if (a == "--out" && i + 1 < argc) out_path = argv[++i];
        else if (a == "--repeat-index" && i + 1 < argc)
            repeat_index = std::stoi(argv[++i]);
    }

    Config c = load_config(config_path);
    torch::manual_seed(c.seed + repeat_index);
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);
    bool is_cuda = device.is_cuda();
    apply_precision(c);
    bool amp = (c.precision == "bf16_amp");

    std::ostringstream js;
    js << "{\n";
    js << "  \"language\": \"cpp\",\n";
    js << "  \"arm\": \"" << arm << "\",\n";
    js << "  \"precision\": \"" << c.precision << "\",\n";
    js << "  \"model\": \"" << c.model << "\",\n";
    js << "  \"device\": \"" << (is_cuda ? "cuda" : "cpu") << "\",\n";
    js << "  \"repeat_index\": " << repeat_index << ",\n";

    auto autocast_guard = [&]() {
        // Return an optional autocast scope for bf16.
        return amp && is_cuda;
    };

    if (arm == "train_throughput") {
        ResNet model = build_model(c.model, c.num_classes, c.cifar_stem);
        model->to(device);
        model->train();
        torch::optim::SGD opt(model->parameters(),
            torch::optim::SGDOptions(c.lr).momentum(c.momentum)
                .weight_decay(c.weight_decay));

        int total = c.warmup_steps + c.measured_steps;
        std::vector<std::pair<torch::Tensor, torch::Tensor>> data;
        data.reserve(total);
        for (int i = 0; i < total; ++i) {
            auto x = torch::randn({c.batch_size, 3, 32, 32}).to(device);
            auto y = torch::randint(0, c.num_classes, {c.batch_size},
                                    torch::kLong).to(device);
            data.emplace_back(x, y);
        }

        std::vector<double> step_ms;
        GpuTimer timer(is_cuda);
        for (int i = 0; i < total; ++i) {
            timer.start();
            opt.zero_grad();
            torch::Tensor loss;
            if (autocast_guard()) {
                at::autocast::set_autocast_enabled(at::kCUDA, true);
                at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
                auto out = model->forward(data[i].first);
                loss = torch::nn::functional::cross_entropy(out, data[i].second);
                at::autocast::clear_cache();
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            } else {
                auto out = model->forward(data[i].first);
                loss = torch::nn::functional::cross_entropy(out, data[i].second);
            }
            loss.backward();
            opt.step();
            double ms = timer.stop_ms();
            if (i >= c.warmup_steps) step_ms.push_back(ms);
        }
        double med = median(step_ms);
        double q1 = percentile(step_ms, 25), q3 = percentile(step_ms, 75);
        double ips = c.batch_size / (med / 1e3);
        js << "  \"batch_size\": " << c.batch_size << ",\n";
        js << "  \"step_ms\": {\n";
        js << "    \"n\": " << step_ms.size() << ",\n";
        js << "    \"median\": " << med << ",\n";
        js << "    \"iqr\": " << (q3 - q1) << ",\n";
        js << "    \"min\": " << *std::min_element(step_ms.begin(), step_ms.end()) << ",\n";
        js << "    \"max\": " << *std::max_element(step_ms.begin(), step_ms.end()) << ",\n";
        js << "    \"images_per_sec_median\": " << ips << "\n";
        js << "  }\n";

    } else if (arm == "inference_latency") {
        ResNet model = build_model(c.model, c.num_classes, c.cifar_stem);
        model->to(device);
        model->eval();
        torch::NoGradGuard ng;
        auto x = torch::randn({c.lat_batch, 3, 32, 32}).to(device);

        std::vector<double> lat_ms;
        GpuTimer timer(is_cuda);
        int total = c.lat_warmup + c.lat_iters;
        for (int i = 0; i < total; ++i) {
            timer.start();
            if (autocast_guard()) {
                at::autocast::set_autocast_enabled(at::kCUDA, true);
                at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
                auto out = model->forward(x);
                at::autocast::clear_cache();
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            } else {
                auto out = model->forward(x);
            }
            double ms = timer.stop_ms();
            if (i >= c.lat_warmup) lat_ms.push_back(ms);
        }
        js << "  \"batch_size\": " << c.lat_batch << ",\n";
        js << "  \"iters\": " << lat_ms.size() << ",\n";
        js << "  \"latency_ms_median\": " << median(lat_ms) << ",\n";
        js << "  \"latency_ms_percentiles\": {\n";
        js << "    \"50\": " << percentile(lat_ms, 50) << ",\n";
        js << "    \"90\": " << percentile(lat_ms, 90) << ",\n";
        js << "    \"95\": " << percentile(lat_ms, 95) << ",\n";
        js << "    \"99\": " << percentile(lat_ms, 99) << ",\n";
        js << "    \"99.9\": " << percentile(lat_ms, 99.9) << "\n";
        js << "  }\n";

    } else if (arm == "inference_throughput") {
        int warm = 50;
        ResNet model = build_model(c.model, c.num_classes, c.cifar_stem);
        model->to(device);
        model->eval();
        torch::NoGradGuard ng;
        auto x = torch::randn({c.thr_batch, 3, 32, 32}).to(device);

        std::vector<double> batch_ms;
        GpuTimer timer(is_cuda);
        int total = warm + c.measured_steps;
        for (int i = 0; i < total; ++i) {
            timer.start();
            if (autocast_guard()) {
                at::autocast::set_autocast_enabled(at::kCUDA, true);
                at::autocast::set_autocast_dtype(at::kCUDA, at::kBFloat16);
                auto out = model->forward(x);
                at::autocast::clear_cache();
                at::autocast::set_autocast_enabled(at::kCUDA, false);
            } else {
                auto out = model->forward(x);
            }
            double ms = timer.stop_ms();
            if (i >= warm) batch_ms.push_back(ms);
        }
        double med = median(batch_ms);
        double q1 = percentile(batch_ms, 25), q3 = percentile(batch_ms, 75);
        js << "  \"batch_size\": " << c.thr_batch << ",\n";
        js << "  \"batch_ms\": {\n";
        js << "    \"n\": " << batch_ms.size() << ",\n";
        js << "    \"median\": " << med << ",\n";
        js << "    \"iqr\": " << (q3 - q1) << ",\n";
        js << "    \"images_per_sec_median\": "
           << (c.thr_batch / (med / 1e3)) << "\n";
        js << "  }\n";
    } else {
        std::cerr << "unknown arm: " << arm << "\n";
        return 2;
    }

    js << "}\n";
    std::cout << js.str();
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        f << js.str();
    }
    return 0;
}
