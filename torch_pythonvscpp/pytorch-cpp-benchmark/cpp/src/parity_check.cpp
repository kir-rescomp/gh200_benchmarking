// Parity gate, C++ side.
//
// Loads the SAME artefacts Python exported:
//   - parity/init_resnet18.pt   (weights, keyed by state_dict name)
//   - parity/fixed_batch.pt     (input + label)
//   - parity/reference_step.pt  (Python's loss, logits, grads)
//
// Loads weights into the C++ model by explicit name, runs one forward +
// backward, and asserts loss / logits / every gradient match the reference
// within tolerance. Exits non-zero on any mismatch so it can gate CI / a
// Slurm dependency chain.
//
// Weights, batch, and reference are exchanged as TorchScript archives
// (torch::pickle_load of a dict<str, Tensor>) written by the companion
// Python exporter using torch.jit.save on a container module. See
// parity/export_torchscript_bundle.py for the exact format this expects.

#include <torch/torch.h>
#include <torch/script.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>

#include "resnet.h"

namespace {

// Read config values we need. Kept minimal to avoid a JSON dependency in the
// parity binary; the harness binary uses a proper JSON lib. These four values
// must match config/experiment.json.
struct MiniConfig {
    std::string model = "resnet18";
    int num_classes = 10;
    bool cifar_stem = true;
    double atol = 1e-4;
    double rtol = 1e-4;
};

// Pull a dict<str,Tensor> out of a TorchScript module that stored tensors as
// named attributes / buffers.
std::map<std::string, torch::Tensor> load_named_tensors(
        const std::string& path) {
    torch::jit::script::Module m = torch::jit::load(path);
    std::map<std::string, torch::Tensor> out;
    for (const auto& p : m.named_buffers(/*recurse=*/true)) {
        out[p.name] = p.value;
    }
    return out;
}

bool close(const torch::Tensor& a, const torch::Tensor& b,
           double atol, double rtol, double& max_abs, double& max_rel) {
    auto af = a.to(torch::kCPU).to(torch::kFloat64);
    auto bf = b.to(torch::kCPU).to(torch::kFloat64);
    auto diff = (af - bf).abs();
    max_abs = diff.max().item<double>();
    auto denom = bf.abs().clamp_min(1e-12);
    max_rel = (diff / denom).max().item<double>();
    return torch::allclose(af, bf, rtol, atol);
}

}  // namespace

int main(int argc, char** argv) {
    MiniConfig cfg;
    torch::manual_seed(42);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA
                                                     : torch::kCPU);
    // Deterministic path for the gate.
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicCuDNN(true);
    at::globalContext().setAllowTF32CuBLAS(false);
    at::globalContext().setAllowTF32CuDNN(false);

    std::cout << "[parity-cpp] device="
              << (device.is_cuda() ? "cuda" : "cpu") << "\n";

    // --- build model ------------------------------------------------------
    ResNet model = build_model(cfg.model, cfg.num_classes, cfg.cifar_stem);
    model->to(device);
    model->to(torch::kFloat32);

    // --- load shared weights by name -------------------------------------
    auto weights = load_named_tensors("parity/init_bundle.pt");
    {
        torch::NoGradGuard ng;
        auto params = model->named_parameters(/*recurse=*/true);
        auto buffers = model->named_buffers(/*recurse=*/true);
        size_t matched = 0;
        for (auto& pr : params) {
            auto it = weights.find(pr.key());
            if (it == weights.end()) {
                std::cerr << "[parity-cpp] MISSING weight for param: "
                          << pr.key() << "\n";
                return 2;
            }
            if (pr.value().sizes() != it->second.sizes()) {
                std::cerr << "[parity-cpp] SHAPE MISMATCH " << pr.key()
                          << "\n";
                return 2;
            }
            pr.value().copy_(it->second.to(device));
            ++matched;
        }
        for (auto& br : buffers) {
            auto it = weights.find(br.key());
            if (it != weights.end() &&
                br.value().sizes() == it->second.sizes()) {
                br.value().copy_(it->second.to(device));
                ++matched;
            }
        }
        std::cout << "[parity-cpp] loaded " << matched
                  << " tensors by name\n";
    }

    // --- load fixed batch -------------------------------------------------
    auto batch = load_named_tensors("parity/fixed_batch_bundle.pt");
    auto x = batch.at("input").to(device);
    auto y = batch.at("label").to(device).to(torch::kLong);

    // --- one forward + backward ------------------------------------------
    model->train();
    model->zero_grad();
    auto logits = model->forward(x);
    auto loss = torch::nn::functional::cross_entropy(logits, y);
    loss.backward();

    // --- load reference ---------------------------------------------------
    auto ref = load_named_tensors("parity/reference_bundle.pt");
    double ref_loss = ref.at("loss").item<double>();
    std::cout << "[parity-cpp] cpp loss    = " << loss.item<double>() << "\n";
    std::cout << "[parity-cpp] python loss = " << ref_loss << "\n";

    int failures = 0;

    // loss
    {
        double a = loss.item<double>();
        double d = std::abs(a - ref_loss);
        if (d > cfg.atol + cfg.rtol * std::abs(ref_loss)) {
            std::cerr << "[parity-cpp] FAIL loss abs_diff=" << d << "\n";
            ++failures;
        } else {
            std::cout << "[parity-cpp] PASS loss abs_diff=" << d << "\n";
        }
    }

    // logits
    {
        double ma, mr;
        bool ok = close(logits, ref.at("logits").to(device),
                        cfg.atol, cfg.rtol, ma, mr);
        std::cout << "[parity-cpp] logits max_abs=" << ma
                  << " max_rel=" << mr
                  << (ok ? "  PASS\n" : "  FAIL\n");
        if (!ok) ++failures;
    }

    // gradients, matched by parameter name (grad.<name>)
    {
        auto params = model->named_parameters(/*recurse=*/true);
        for (auto& pr : params) {
            std::string gk = "grad." + pr.key();
            auto it = ref.find(gk);
            if (it == ref.end()) continue;  // buffers etc. have no grad
            if (!pr.value().grad().defined()) {
                std::cerr << "[parity-cpp] FAIL no grad for " << pr.key()
                          << "\n";
                ++failures;
                continue;
            }
            double ma, mr;
            bool ok = close(pr.value().grad(), it->second.to(device),
                            cfg.atol, cfg.rtol, ma, mr);
            if (!ok) {
                std::cerr << "[parity-cpp] FAIL grad " << pr.key()
                          << " max_abs=" << ma << " max_rel=" << mr << "\n";
                ++failures;
            }
        }
    }

    if (failures == 0) {
        std::cout << "[parity-cpp] ALL PARITY CHECKS PASSED\n";
        return 0;
    }
    std::cerr << "[parity-cpp] " << failures << " PARITY CHECK(S) FAILED\n";
    return 1;
}
