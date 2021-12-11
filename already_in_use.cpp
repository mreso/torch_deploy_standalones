// PyTorch
#include <torch/torch.h>
// #include <torch/types.h>
// #include <torch/script.h>
#include <torch/csrc/deploy/deploy.h>
#include <torch/csrc/deploy/path_environment.h>

// C++ STD
#include <map>

using namespace std;

int main(const int argc, const char* const argv[]) {
    
    at::set_num_interop_threads(1);
    at::set_num_threads(1); 

    // Torch script
    vector<torch::jit::Module> traced;

    //TorchDeploy
    shared_ptr<torch::deploy::InterpreterManager> manager;
    shared_ptr<torch::deploy::Package> package;
    shared_ptr<torch::deploy::Environment> env;
    torch::deploy::ReplicatedObj deployed;

    env = make_shared<torch::deploy::PathEnvironment>("../transformers_venv/lib/python3.8/site-packages/");
    manager.reset(new torch::deploy::InterpreterManager(4, env));
    package = make_shared<torch::deploy::Package>(manager->loadPackage("../models/bert_model_only.pt"));
    deployed = package->loadPickle("model", "model.pkl");
    
    std::unordered_map<std::string, c10::IValue> data;

    data["input_ids"] = torch::tensor(std::vector<int64_t>{101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107, 1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111, 1240,  2332,   102}).unsqueeze(0).to("cuda");
    data["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0).to("cuda");
    data["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}).unsqueeze(0).to("cuda");

    for(int i=0; i<1000; ++i)
    {
        auto ret = deployed.callKwargs({}, data).toIValue().toTuple()->elements()[0];
    }
}
