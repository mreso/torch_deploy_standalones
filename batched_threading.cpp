// PyTorch
#include <torch/torch.h>
#include <torch/types.h>
#include <torch/script.h>

// C++ STD
#include <chrono>
#include <cmath>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <thread>

using namespace std;

int WARM_UP = 100;
int TOTAL_BATCHES = 10000 + WARM_UP;

typedef  unordered_map<string, c10::IValue> KWARG;

struct BatchQueue {
   BatchQueue() {
   }
   
   ~BatchQueue() {
   }

   void enqueue(vector<KWARG> batch) {
      {
         lock_guard<mutex> lock(m_);
         batches_.push(std::move(batch));
      }
      cv_.notify_one();
   }

   vector<KWARG> dequeue()
   {
       {
           unique_lock<mutex> lock(m_);
           if(batches_.size()==0)
            return vector<KWARG>();
           cv_.wait(lock, [&]{return batches_.size();});
       }

       vector<KWARG> batch;
       bool terminate = true;
       {
        lock_guard<mutex> lock(m_);
        if(batches_.size() ) {
            
            batch = std::move(batches_.front());
            batches_.pop();
            if(batches_.size() % 100 == 0)
                cout << "Queue size: " << batches_.size() << endl;
            terminate = batches_.size() == 0;
        }
       }
       if(terminate)
        cv_.notify_all();

       return batch;
   }
   
   condition_variable cv_;
   mutex m_;
   queue<vector<KWARG>> batches_;
};


class TorchScriptWorker{
    public:
    TorchScriptWorker(shared_ptr<BatchQueue> queue)
    :queue_(queue){
        wait_waiting = 0;
        wait_counter = 0;
        for(int i=0; i<4;++i)
            threads_.emplace_back([this, i]{process(i);});
    }

    ~TorchScriptWorker(){
        for(auto &t : threads_)
            t.join();
    }

    void process(int idx){

        auto model = torch::jit::load("../models/bert_model_only_traced.pt");
        model.to(at::Device(torch::kCUDA, idx));


        for(int i=0; i<WARM_UP; ++i) {
            vector<KWARG> batch = queue_->dequeue();
            if(batch.size() == 0)
                break;

            do_work_on_batch(model, move(batch), idx);
        }

        cout << "Finished warmup: " << idx << endl;

        wait();
        cout << "Starting for real: " << idx << endl;

        while(true) {
            vector<KWARG> batch = queue_->dequeue();
            if(batch.size() == 0)
                break;

            do_work_on_batch(model, move(batch), idx);
        }
    }

    void do_work_on_batch(torch::jit::script::Module& model, vector<KWARG> batch, int idx){
         vector<torch::Tensor> input_ids, token_type_ids, attention_mask;

        for(auto &kw : batch) {
            input_ids.push_back(kw["input_ids"].toTensor());
            token_type_ids.push_back(kw["token_type_ids"].toTensor());
            attention_mask.push_back(kw["attention_mask"].toTensor());
        }

        KWARG input_data;
        
        input_data["input_ids"] = torch::stack(input_ids).pin_memory().to(at::Device(torch::kCUDA, idx));
        input_data["token_type_ids"] = torch::stack(token_type_ids).pin_memory().to(at::Device(torch::kCUDA, idx));
        input_data["attention_mask"] = torch::stack(attention_mask).pin_memory().to(at::Device(torch::kCUDA, idx));

        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        auto ret = model.forward({}, input_data).toIValue().toTuple()->elements()[0];
        auto res = torch::softmax(ret.toTensor(),1);
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        {
            lock_guard<mutex> lock(m_);
            if(chrono::duration_cast<chrono::milliseconds>(end - begin).count()>50)
                cout << "Processing time (ms): " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << endl;
        }

        vector<string> answers;
        for(int i=0; i<batch.size(); ++i) {
            float paraphrased_percent = 100.0 * res[i][1].item<float>();
            answers.push_back(to_string((int)round(paraphrased_percent)) + "% paraphrase");
        }
    }

    void wait() {
        std::unique_lock<std::mutex> lk(wait_m_);
        ++wait_counter;
        ++wait_waiting;
        wait_cv_.wait(lk, [&]{return wait_counter >= threads_.size();});
        wait_cv_.notify_one();
        --wait_waiting;
        if(wait_waiting == 0)
           wait_counter = 0;
        lk.unlock();
    }

    mutex m_;
    shared_ptr<BatchQueue> queue_;
    vector<thread> threads_;

    mutex wait_m_;
    condition_variable wait_cv_;
    int wait_counter;
    int wait_waiting;
};


int main(const int argc, const char* const argv[]) {
    shared_ptr<BatchQueue> batch_queue = make_shared<BatchQueue>();
    
    for(int i=0; i< TOTAL_BATCHES; ++i) { 
        vector<KWARG> batch;
        for(int j=0; j<8; ++j) 
        {
            KWARG kwargs;

            kwargs["input_ids"] = torch::tensor(std::vector<int64_t>{
                101,  1109,  1419, 20164, 10932,  2271,  7954,  1110,  1359,  1107,
                1203,  1365,  1392,   102,  7302,  1116,  1132,  2108,  2213,  1111,
                1240,  2332,   102});
            kwargs["token_type_ids"] = torch::tensor(std::vector<int64_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            kwargs["attention_mask"] = torch::tensor(std::vector<int64_t>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
            batch.push_back(kwargs);
        }

        batch_queue->enqueue(batch);
    }

    TorchScriptWorker  worker(batch_queue);
}
