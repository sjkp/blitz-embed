#include "bert.h"
#include "ggml.h"
#include "httplib.h"
#include <unistd.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <thread>
#include "json.hpp"

using json = nlohmann::json;
using string = std::string;

struct bert_options {
    const char* model = nullptr;
    int32_t n_max_tokens = 0;
    int32_t batch_size = 32;
    bool use_cpu = false;
    bool normalize = true;
    int32_t n_threads = 6;
};

class BertApp {
private:
    std::unique_ptr<bert_ctx> bctx;
    bert_options options;

public:
    
    BertApp(const char* model_path, 
            int32_t n_max_tokens = 512, 
            bool use_cpu = true, 
            int32_t n_threads = 6,
            int32_t batch_size = 1, 
            bool normalize = true) {

        options.model = model_path;
        options.use_cpu = use_cpu;
        options.n_threads = n_threads;
        options.n_max_tokens = n_max_tokens;
        options.batch_size = batch_size;
        options.normalize = normalize;
        unsigned int max_threads = std::thread::hardware_concurrency();
        printf("Max threads %d\n", max_threads);

        ggml_time_init();
        int64_t t_start_us = ggml_time_us();
        bctx.reset(bert_load_from_file(options.model, options.use_cpu));
        if (!bctx) {
            fprintf(stderr, "Failed to load model from '%s'\n", options.model);
            exit(1); 
        }

        if (options.n_max_tokens <= 0) {
            options.n_max_tokens = bert_n_max_tokens(bctx.get());
        }

        bert_allocate_buffers(bctx.get(), options.n_max_tokens, options.batch_size);
        int64_t t_end_us = ggml_time_us();
        printf("Model loaded in %0.2f ms\n", (t_end_us - t_start_us) / 1000.0);
    }


    ~BertApp() {
        // if (bctx != nullptr) {
        //     bert_free(bctx.get());
        // }
    }


        json run(const std::vector<std::string>& prompts) {

            const int n_embd = bert_n_embd(bctx.get());
            std::vector<float> embed(prompts.size() * n_embd);

            int64_t t_start_us = ggml_time_us();  
            // Use bert_encode_batch to process all prompts
            bert_encode_batch(bctx.get(), prompts, embed.data(), options.normalize, options.n_threads);
            int64_t t_infer_us = ggml_time_us() - t_start_us;
            printf("Tokenise and Infer time = %0.2f ms", t_infer_us / 1000.0); 


            // Format the embeddings into a list of lists
            std::stringstream ss;
            ss << "["; // Start of the overall list

            for (size_t promptIndex = 0; promptIndex < prompts.size(); ++promptIndex) {
                if (promptIndex > 0) ss << ", "; // Separate lists for different prompts

                ss << "["; // Start of the list for the current prompt's embedding
                for (size_t elemIndex = 0; elemIndex < n_embd; ++elemIndex) {
                    if (elemIndex > 0) ss << ", "; // Separate elements within the current embedding
                    ss << embed[promptIndex * n_embd + elemIndex]; // Access the correct element in the flat `embed` array
                }
                ss << "]"; // End of the list for the current prompt's embedding
            }

            ss << "]"; // End of the overall list

            // return ss.str();

            json body_json = {
                {"message", "Success"},
                {"itime", t_infer_us},
                {"embedding", ss.str()} 
            };

            return body_json;
        }


    
};


std::unique_ptr<BertApp> app; // Global app instance

void ensureAppInstance(const json& bodyJson) {
    // Default values
    std::string model_path = "/opt/bge-base-en-v1.5-q4_0.gguf";
    int32_t n_max_tokens = 512;
    bool use_cpu = true;
    int32_t n_threads = 6;
    int32_t batch_size = 4;
    bool normalize = true;

    // Update values based on payload
    if (bodyJson.find("model") != bodyJson.end() && !bodyJson["model"].empty()) {
        model_path = bodyJson["model"].get<std::string>();
    }
    if (bodyJson.find("max_len") != bodyJson.end() && bodyJson["max_len"].is_number_integer()) {
        n_max_tokens = bodyJson["max_len"].get<int32_t>();
    }
    if (bodyJson.find("batch_size") != bodyJson.end() && bodyJson["batch_size"].is_number_integer()) {
        batch_size = bodyJson["batch_size"].get<int32_t>();
    }
    if (bodyJson.find("normalize") != bodyJson.end()) {
        normalize = bodyJson["normalize"].get<bool>();
    }

    // Instantiate or re-instantiate app based on the model change
    if (!app || bodyJson.find("model") != bodyJson.end()) {
        app = std::make_unique<BertApp>(model_path.c_str(), n_max_tokens, use_cpu, n_threads, batch_size, normalize);
    }
}

int main() {
    using namespace httplib;

    Server svr;

    svr.Post("/", [&](const Request& req, Response& res) {
        json bodyJson = json::parse(req.body);

        // Ensure app instance is correct before handling the request
        ensureAppInstance(bodyJson);

        auto sentArray = bodyJson["sent"].get<std::vector<std::string>>();
        json body_json = app->run(sentArray);

        json result_json = {
            {"isBase64Encoded", false},
            {"statusCode", 200},
            {"body", body_json.dump()}
        };

        std::string result = result_json.dump();
        res.set_content(result, "application/json");
    });

    std::string port = std::getenv("PORT") ? std::getenv("PORT") : "8080";
    std::cout << "Listening on port " << port << std::endl;

    svr.listen("0.0.0.0", std::stoi(port));

    return 0;
}
