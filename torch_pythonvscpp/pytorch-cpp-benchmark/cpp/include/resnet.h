// ResNet for CIFAR, C++ / LibTorch.
//
// This MUST mirror python/model.py exactly: identical layer order, identical
// channel counts, identical stem treatment, identical submodule *registration
// names* so that the exported Python state_dict keys line up with the C++
// module's named_parameters(). If the parity gate fails on a name/shape
// mismatch, this file and model.py have diverged.
#pragma once

#include <torch/torch.h>
#include <vector>

struct BasicBlockImpl : torch::nn::Module {
    static const int expansion = 1;

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample{nullptr};

    BasicBlockImpl(int in_planes, int planes, int stride = 1) {
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 3)
                                  .stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(planes));
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(planes, planes, 3)
                                  .stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(planes));

        if (stride != 1 || in_planes != expansion * planes) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(
                    in_planes, expansion * planes, 1)
                        .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(expansion * planes)));
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor identity = x;
        auto out = torch::relu(bn1(conv1(x)));
        out = bn2(conv2(out));
        if (!downsample.is_empty()) {
            identity = downsample->forward(x);
        }
        out += identity;
        return torch::relu(out);
    }
};
TORCH_MODULE(BasicBlock);

struct ResNetImpl : torch::nn::Module {
    int in_planes = 64;
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::AnyModule maxpool;
    torch::nn::Sequential layer1, layer2, layer3, layer4;
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};

    ResNetImpl(std::vector<int> num_blocks, int num_classes = 10,
               bool cifar_stem = true) {
        if (cifar_stem) {
            conv1 = register_module("conv1",
                torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3)
                                      .stride(1).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
            // Identity maxpool; registered so the module tree matches.
            maxpool = torch::nn::AnyModule(
                register_module("maxpool", torch::nn::Identity()));
        } else {
            conv1 = register_module("conv1",
                torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7)
                                      .stride(2).padding(3).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
            maxpool = torch::nn::AnyModule(register_module("maxpool",
                torch::nn::MaxPool2d(
                    torch::nn::MaxPool2dOptions(3).stride(2).padding(1))));
        }

        layer1 = register_module("layer1", make_layer(64,  num_blocks[0], 1));
        layer2 = register_module("layer2", make_layer(128, num_blocks[1], 2));
        layer3 = register_module("layer3", make_layer(256, num_blocks[2], 2));
        layer4 = register_module("layer4", make_layer(512, num_blocks[3], 2));
        avgpool = register_module("avgpool",
            torch::nn::AdaptiveAvgPool2d(
                torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        fc = register_module("fc", torch::nn::Linear(
            512 * BasicBlockImpl::expansion, num_classes));
    }

    torch::nn::Sequential make_layer(int planes, int blocks, int stride) {
        torch::nn::Sequential seq;
        std::vector<int> strides;
        strides.push_back(stride);
        for (int i = 1; i < blocks; ++i) strides.push_back(1);
        for (int s : strides) {
            seq->push_back(BasicBlock(in_planes, planes, s));
            in_planes = planes * BasicBlockImpl::expansion;
        }
        return seq;
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));
        x = maxpool.forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = avgpool(x);
        x = torch::flatten(x, 1);
        return fc(x);
    }
};
TORCH_MODULE(ResNet);

inline ResNet build_model(const std::string& name, int num_classes,
                          bool cifar_stem) {
    if (name == "resnet18")
        return ResNet({2, 2, 2, 2}, num_classes, cifar_stem);
    if (name == "resnet34")
        return ResNet({3, 4, 6, 3}, num_classes, cifar_stem);
    throw std::runtime_error("unknown model: " + name);
}
