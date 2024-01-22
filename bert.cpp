#include "bert.h"
#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <regex>
#include <thread>
#include <algorithm>

// default hparams (all-MiniLM-L6-v2)
struct bert_hparams {
    int32_t n_vocab = 30522;
    int32_t n_max_tokens = 512;
    int32_t n_embd = 256;
    int32_t n_intermediate = 1536;
    int32_t n_head = 12;
    int32_t n_layer = 6;
    int32_t f16 = 1;
};

struct bert_layer {
    // normalization
    struct ggml_tensor *ln_att_w;
    struct ggml_tensor *ln_att_b;

    struct ggml_tensor *ln_out_w;
    struct ggml_tensor *ln_out_b;

    // attention
    struct ggml_tensor *q_w;
    struct ggml_tensor *q_b;
    struct ggml_tensor *k_w;
    struct ggml_tensor *k_b;
    struct ggml_tensor *v_w;
    struct ggml_tensor *v_b;

    struct ggml_tensor *o_w;
    struct ggml_tensor *o_b;

    // ff
    struct ggml_tensor *ff_i_w;
    struct ggml_tensor *ff_i_b;

    struct ggml_tensor *ff_o_w;
    struct ggml_tensor *ff_o_b;
};

struct bert_vocab {
    std::map<std::string, bert_vocab_id> token_to_id;
    std::map<std::string, bert_vocab_id> subword_token_to_id;

    std::map<bert_vocab_id, std::string> _id_to_token;
    std::map<bert_vocab_id, std::string> _id_to_subword_token;
};

struct bert_model {
    bert_hparams hparams;

    // embeddings weights
    struct ggml_tensor *word_embeddings;
    struct ggml_tensor *token_type_embeddings;
    struct ggml_tensor *position_embeddings;
    struct ggml_tensor *ln_e_w;
    struct ggml_tensor *ln_e_b;

    std::vector<bert_layer> layers;

    struct ggml_context *ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
};

// replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct bert_buffer {
    uint8_t * data = NULL;
    size_t size = 0;

    void resize(size_t size) {
        delete[] data;
        data = new uint8_t[size];
        this->size = size;
    }

    ~bert_buffer() {
        delete[] data;
    }
};

struct bert_ctx {
    bert_model model;
    bert_vocab vocab;

    size_t mem_per_token;
    int64_t mem_per_input;
    int32_t max_batch_n;
    bert_buffer buf_compute;
};

int32_t bert_n_embd(bert_ctx * ctx) {
    return ctx->model.hparams.n_embd;
}

int32_t bert_n_max_tokens(bert_ctx * ctx) {
    return ctx->model.hparams.n_max_tokens;
}

const char* bert_vocab_id_to_token(bert_ctx * ctx, bert_vocab_id id) {
    bert_vocab & vocab = ctx->vocab;
    auto it = vocab._id_to_token.find(id);
    if (it != vocab._id_to_token.end())
    {
        return it->second.c_str();
    }
    it = vocab._id_to_subword_token.find(id);
    if (it != vocab._id_to_subword_token.end())
    {
        return it->second.c_str();
    }
    return "[UNK TOKEN from bert_vocab]";
}

//
// command line interface
//

void bert_print_usage(char **argv, const bert_params &params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -s SEED, --seed SEED  RNG seed (default: -1)\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  --port p     port to bind in server mode (default: %d)\n", params.port);
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model);
    fprintf(stderr, "\n");
}

bool bert_params_parse(int argc, char **argv, bert_params &params) {
    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads")
        {
            params.n_threads = std::stoi(argv[++i]);
        }
        else if (arg == "-p" || arg == "--prompt")
        {
            params.prompt = argv[++i];
        }
        else if (arg == "--port")
        {
            params.port = std::stoi(argv[++i]);
        }
        else if (arg == "-m" || arg == "--model")
        {
            params.model = argv[++i];
        }
        else if (arg == "-h" || arg == "--help")
        {
            bert_print_usage(argv, params);
            exit(0);
        }
        else
        {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bert_print_usage(argv, params);
            exit(0);
        }
    }

    return true;
}

//
// Tokenizing
//

static size_t utf8_len(char src) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

std::string strip_accents(const std::string &inputString) {
    std::string resultString;
    std::map<std::string, char> accentMap = {{"À", 'A'},{"Á", 'A'},
        {"Â", 'A'},{"Ã", 'A'},{"Ä", 'A'},{"Å", 'A'},{"à", 'a'},{"á", 'a'},
        {"â", 'a'},{"ã", 'a'},{"ä", 'a'},{"å", 'a'},{"È", 'E'},{"É", 'E'},
        {"Ê", 'E'},{"Ë", 'E'},{"è", 'e'},{"é", 'e'},{"ê", 'e'},{"ë", 'e'},
        {"Ì", 'I'},{"Í", 'I'},{"Î", 'I'},{"Ï", 'I'},{"ì", 'i'},{"í", 'i'},
        {"î", 'i'},{"ï", 'i'},{"Ò", 'O'},{"Ó", 'O'},{"Ô", 'O'},{"Õ", 'O'},
        {"Ö", 'O'},{"ò", 'o'},{"ó", 'o'},{"ô", 'o'},{"õ", 'o'},{"ö", 'o'},
        {"Ù", 'U'},{"Ú", 'U'},{"Û", 'U'},{"Ü", 'U'},{"ù", 'u'},{"ú", 'u'},
        {"û", 'u'},{"ü", 'u'},{"Ý", 'Y'},{"ý", 'y'},{"Ç", 'C'},{"ç", 'c'},
        {"Ñ", 'N'},{"ñ", 'n'},
    };

    for (size_t i = 0; i < inputString.length();)
    {
        int len = utf8_len(inputString[i]);
        std::string curChar = inputString.substr(i, len);
        auto iter = accentMap.find(curChar);
        if (iter != accentMap.end())
        {
            resultString += iter->second;
        }
        else
        {
            resultString += curChar;
        }
        i += len;
    }

    return resultString;
}

std::string bert_normalize_prompt(const std::string &text)
{
    // TODO: handle chinese characters? https://github.com/huggingface/tokenizers/blob/ef5f50605ddf9f8caef1598c0e4853862b9707a7/tokenizers/src/normalizers/bert.rs#L98
    std::string text2 = strip_accents(text);
    for (size_t i = 0; i < text2.size(); i += utf8_len(text2[i]))
    {
        char c = text2[i];
        if (c >= 'A' && c <= 'Z')
            text2[i] = c - 'A' + 'a';
    }
    return text2;
}

bool is_chinese_char(const std::string& str) {
    int len = str.length();
    unsigned int codepoint = 0;
    int num_bytes = 0;
    int i = 0;
    unsigned char ch = static_cast<unsigned char>(str[i]);
    if (ch <= 0x7f) {
        codepoint = ch;
        num_bytes = 1;
    } else if ((ch >> 5) == 0x06) {
        codepoint = ch & 0x1f;
        num_bytes = 2;
    } else if ((ch >> 4) == 0x0e) {
        codepoint = ch & 0x0f;
        num_bytes = 3;
    } else if ((ch >> 3) == 0x1e) {
        codepoint = ch & 0x07;
        num_bytes = 4;
    }
    for (int j = 1; j < num_bytes; ++j) {
        if (i + j >= len) {
            return false; // incomplete UTF-8 character
        }
        unsigned char next_ch = static_cast<unsigned char>(str[i + j]);
        if ((next_ch >> 6) != 0x02) {
            return false; // invalid trailing byte
        }
        codepoint = (codepoint << 6) | (next_ch & 0x3f);
    }
    if ((codepoint >= 0x4E00 && codepoint <= 0x9FFF) ||
        (codepoint >= 0x3400 && codepoint <= 0x4DBF) ||
        (codepoint >= 0x20000 && codepoint <= 0x2A6DF) ||
        (codepoint >= 0x2A700 && codepoint <= 0x2B73F) ||
        (codepoint >= 0x2B740 && codepoint <= 0x2B81F) ||
        (codepoint >= 0x2B920 && codepoint <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
        (codepoint >= 0xF900 && codepoint <= 0xFAFF) ||
        (codepoint >= 0x2F800 && codepoint <= 0x2FA1F) ||
        (codepoint >= 0x3000 && codepoint <= 0x303F) ||
        (codepoint >= 0xFF00 && codepoint <= 0xFFEF)) {
        return true;
    }
    return false;
}

void bert_tokenize(
    struct bert_ctx * ctx,
         const char * text,
      bert_vocab_id * tokens,
            int32_t * n_tokens,
            int32_t   n_max_tokens) {

    int cls_tok_id = 101;
    int sep_tok_id = 102;
    int unk_tok_id = 100;
    const bert_vocab &vocab = ctx->vocab;

    std::string ori_str = text;
    ori_str = bert_normalize_prompt(ori_str);

    // single punct / single symbol / single digit
    // baseline: add whitespace on the left and right of punct and chinese characters
    std::vector<std::string> words;
    std::string new_str = "";
    int i = 0;
    while (i < ori_str.size()) {
        int utf_char_len = utf8_len(ori_str[i]);
        if ((utf_char_len == 1) && ispunct(ori_str[i])) {
            new_str += " ";
            new_str += ori_str[i];
            new_str += " ";
            i += 1;
        }
        else if ((utf_char_len == 3) && is_chinese_char(ori_str.substr(i, 3))) {
            new_str += " ";
            new_str += ori_str.substr(i, 3);
            new_str += " ";
            i += 3;
        }
        else {
            new_str += ori_str[i];
            i += 1;
        }
    }

    int l = 0;
    int r = 0;
    while (r < new_str.size()) {
        // if is whitespace
        if (isspace(new_str[r])) {
            if (r > l)
                words.push_back(new_str.substr(l, (r - l)));
            l = r + 1;
            r = l;
        }
        else {
            r += 1;
        }
    }
    if (r > l) {
        words.push_back(new_str.substr(l, (r - l)));
    }

    /*
    assert (words.size() == words.size());
    for (auto i = 0; i < words.size(); i++)
    {
        if (words[i] != words[i])
        {
            printf("words[%d] = %s, words[%d] = %s\n", i, words[i].c_str(), i, words[i].c_str());
        }
    }
    */

    int32_t t = 0;
    int32_t prev_t = 0;
    tokens[t++] = cls_tok_id;

    // find the longest tokens that form the words:
    for (const auto &word : words) {
        if (word.size() == 0) {
            continue;
        }
        prev_t = t;

        int i = 0;
        int n = word.size();
        auto *token_map = &vocab.token_to_id;
    loop:
        while (i < n) {
            if (t >= n_max_tokens - 1) {
                break;
            }
            int j = n;
            while (j > i) {
                auto it = token_map->find(word.substr(i, j - i));
                if (it != token_map->end()) {
                    tokens[t++] = it->second;
                    i = j;
                    token_map = &vocab.subword_token_to_id;
                    goto loop;
                }
                --j;
            }
            if (j == i) {
                // fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                token_map = &vocab.subword_token_to_id;
                ++i;
            }
        }
        if (prev_t == t) {
            // fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.data());
            tokens[t++] = unk_tok_id;
        }

    }
    tokens[t++] = sep_tok_id;
    *n_tokens = t;
}

//
// Loading and setup
//

struct bert_ctx * bert_load_from_file(const char *fname) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname);

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        return nullptr;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *)&magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname);
            return nullptr;
        }
    }

    bert_ctx * new_bert = new bert_ctx;
    bert_model & model = new_bert->model;
    bert_vocab & vocab = new_bert->vocab;

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *)&hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *)&hparams.n_max_tokens, sizeof(hparams.n_max_tokens));
        fin.read((char *)&hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *)&hparams.n_intermediate, sizeof(hparams.n_intermediate));
        fin.read((char *)&hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *)&hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *)&hparams.f16, sizeof(hparams.f16));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_max_tokens   = %d\n", __func__, hparams.n_max_tokens);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_intermediate  = %d\n", __func__, hparams.n_intermediate);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: f16     = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = model.hparams.n_vocab;

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *)&len, sizeof(len));

            word.resize(len);
            fin.read((char *)word.data(), len);

            if (word[0] == '#' && word[1] == '#') {
                vocab.subword_token_to_id[word.substr(2)] = i;
                vocab._id_to_subword_token[i] = word;
            }

            if (vocab.token_to_id.count(word) == 0) {
                vocab.token_to_id[word] = i;
                vocab._id_to_token[i] = word;
            }
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = GGML_TYPE_COUNT;
    switch (model.hparams.f16) {
    case 0:
        wtype = GGML_TYPE_F32;
        break;
    case 1:
        wtype = GGML_TYPE_F16;
        break;
    case 2:
        wtype = GGML_TYPE_Q4_0;
        break;
    case 3:
        wtype = GGML_TYPE_Q4_1;
        break;
    default:
    {
        fprintf(stderr, "%s: invalid model file '%s' (bad f16 value %d)\n",
                __func__, fname, model.hparams.f16);
        bert_free(new_bert);
        return nullptr;
    }
    }

    auto &ctx = model.ctx;

    size_t model_mem_req = 0;

    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_intermediate = hparams.n_intermediate;
        const int n_vocab = hparams.n_vocab;

        // Calculate size requirements

        model_mem_req += n_embd * n_vocab * ggml_type_sizef(wtype); // word_embeddings
        model_mem_req += n_embd * 2 * ggml_type_sizef(wtype); // token_type_embeddings
        model_mem_req += n_embd * n_max_tokens * ggml_type_sizef(wtype); // position_embeddings

        model_mem_req += 2 * n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_e_*

        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_*

        model_mem_req += 4 * n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // kqvo weights
        model_mem_req += 4 * n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // kqvo bias

        model_mem_req += 2 * n_layer * (n_embd * n_intermediate * ggml_type_sizef(wtype)); // ff_*_w
        model_mem_req += n_layer * (n_intermediate * ggml_type_sizef(GGML_TYPE_F32)); // ff_i_b
        model_mem_req += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ff_o_b

        model_mem_req += (5 + 16 * n_layer) * 512; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, model_mem_req / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size = model_mem_req,
            .mem_buffer = NULL,
            .no_alloc = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            bert_free(new_bert);
            return nullptr;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_intermediate = hparams.n_intermediate;
        const int n_max_tokens = hparams.n_max_tokens;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.word_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.token_type_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, 2);
        model.position_embeddings = ggml_new_tensor_2d(ctx, wtype, n_embd, n_max_tokens);

        model.ln_e_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_e_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        // map by name
        model.tensors["embeddings.word_embeddings.weight"] = model.word_embeddings;
        model.tensors["embeddings.token_type_embeddings.weight"] = model.token_type_embeddings;
        model.tensors["embeddings.position_embeddings.weight"] = model.position_embeddings;

        model.tensors["embeddings.LayerNorm.weight"] = model.ln_e_w;
        model.tensors["embeddings.LayerNorm.bias"] = model.ln_e_b;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = model.layers[i];

            layer.ln_att_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_att_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_out_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.q_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.q_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.k_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.k_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.v_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.v_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.o_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.ff_i_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_intermediate);
            layer.ff_i_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_intermediate);

            layer.ff_o_w = ggml_new_tensor_2d(ctx, wtype, n_intermediate, n_embd);
            layer.ff_o_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name

            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.weight"] = layer.q_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.query.bias"] = layer.q_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.weight"] = layer.k_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.key.bias"] = layer.k_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.weight"] = layer.v_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.self.value.bias"] = layer.v_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.weight"] = layer.ln_att_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.LayerNorm.bias"] = layer.ln_att_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.weight"] = layer.o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".attention.output.dense.bias"] = layer.o_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.weight"] = layer.ff_i_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".intermediate.dense.bias"] = layer.ff_i_b;

            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.weight"] = layer.ln_out_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.LayerNorm.bias"] = layer.ln_out_b;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.weight"] = layer.ff_o_w;
            model.tensors["encoder.layer." + std::to_string(i) + ".output.dense.bias"] = layer.ff_o_b;
        }
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ftype), sizeof(ftype));

            if (fin.eof()) {
                break;
            }

            int64_t nelements = 1;
            int64_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                int32_t ne_cur;
                fin.read(reinterpret_cast<char *>(&ne_cur), sizeof(ne_cur));
                ne[i] = ne_cur;
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                bert_free(new_bert);
                return nullptr;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                bert_free(new_bert);
                return nullptr;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%lld, %lld]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                bert_free(new_bert);
                return nullptr;
            }

            /*
            static const char *ftype_str[] = {
                "f32",
                "f16",
                "q4_0",
                "q4_1",
            };
            printf("%24s - [%5lld, %5lld], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1], ftype_str[ftype], ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            */

            size_t bpe = 0;

            switch (ftype) {
            case 0:
                bpe = ggml_type_size(GGML_TYPE_F32);
                break;
            case 1:
                bpe = ggml_type_size(GGML_TYPE_F16);
                break;
            case 2:
                bpe = ggml_type_size(GGML_TYPE_Q4_0);
                assert(ne[0] % 64 == 0);
                break;
            case 3:
                bpe = ggml_type_size(GGML_TYPE_Q4_1);
                assert(ne[0] % 64 == 0);
                break;
            default:
            {
                fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                bert_free(new_bert);
                return nullptr;
            }
            }

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %llu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                bert_free(new_bert);
                return nullptr;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            // printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    // Calculate space requirements for setting up context buffers later
    {
        bert_vocab_id tokens[] = {0, 1, 2, 3};
        // TODO: We set the initial buffer size to 32MB and hope it's enough. Maybe there is a better way to do this?
        new_bert->buf_compute.resize(32 * 1024 * 1024);
        bert_forward(new_bert, 1, tokens, 4, nullptr);
        new_bert->max_batch_n = 0;

        // TODO: Max tokens should be a param?
        int32_t N = new_bert->model.hparams.n_max_tokens;
        new_bert->mem_per_input = 1.1 * (new_bert->mem_per_token * N); // add 10% to account for ggml object overhead

    }
    printf("%s: mem_per_token %zu KB, mem_per_input %lld MB\n", __func__, new_bert->mem_per_token / (1 << 10), new_bert->mem_per_input / (1 << 20));

    return new_bert;
}

void bert_resize_ctx(bert_ctx * ctx, int32_t new_batch_size, int32_t max_len) {
    max_len = 512; // bug have to be 512
    int64_t new_mem_per_input = 1.15 * (ctx->mem_per_token * max_len);
    int64_t new_buf_size = new_mem_per_input * new_batch_size;

    // TODO: Max memory should be a param? Now just 12 GB
    int64_t GB = (1 << 30);
    GB *= 12;
    //printf("%s: requested_buf_size %lldMB\n", __func__, new_buf_size / (1 << 20));
    if (new_buf_size > GB) {
        int32_t adjusted_new_batch_size = GB / new_mem_per_input;
        if (adjusted_new_batch_size < 1) adjusted_new_batch_size = 1;
        // printf("%s: requested batch size %d, actual new batch size %d\n", __func__, new_batch_size, adjusted_new_batch_size);
        new_batch_size = adjusted_new_batch_size;
        new_buf_size = new_mem_per_input * new_batch_size;
    }
    if (new_buf_size > (ctx->mem_per_input * ctx->max_batch_n)) {
        // printf("%s: new_buf_size %lldMB, old_buf_size %lldMB\n", __func__, new_buf_size / (1 << 20), (ctx->mem_per_input * ctx->max_batch_n) / (1 << 20));
        ctx->buf_compute.resize(new_buf_size);
        ctx->max_batch_n = new_batch_size;
    }
    ctx->mem_per_input = new_mem_per_input;
}

void bert_free(bert_ctx * ctx) {
    ggml_free(ctx->model.ctx);
    delete ctx;
}

void bert_forward(
    struct bert_ctx * ctx,
            int32_t   n_threads,
      bert_vocab_id * tokens,
            int32_t   n_tokens,
              float * embeddings) {
    bert_forward_batch(ctx, n_threads, 1, &tokens, &n_tokens, embeddings ? &embeddings : nullptr);
}

void bert_forward_batch(
          bert_ctx *  ctx,
           int32_t    n_threads,
           int32_t    n_batch_size,
     bert_vocab_id ** batch_tokens,
           int32_t *  n_tokens,
             float ** batch_embeddings) {

    const bert_model& model = ctx->model;

    int cur_max_n_tokens = 0;
    for (int ba = 0; ba < n_batch_size; ba++)
    {
        if (n_tokens[ba] > cur_max_n_tokens)
            cur_max_n_tokens = n_tokens[ba];
    }
    int cur_max_len = cur_max_n_tokens;

    // batch_embeddings is nullptr for the initial memory requirements run
    bool mem_req_mode = !batch_embeddings;
    if (!mem_req_mode && n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size, cur_max_n_tokens);
        if (n_batch_size > ctx->max_batch_n) {
            fprintf(stderr, "%s: tried to increase buffers to batch size %d but failed, please increase the limitation of max memory\n", __func__, n_batch_size);
            return;
        }
    }

    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_max_tokens = hparams.n_max_tokens;
    const int n_head = hparams.n_head;

    const int d_head = n_embd / n_head;

    std::vector<float> result;
    if (cur_max_n_tokens > n_max_tokens)
    {
        fprintf(stderr, "Too many tokens, maximum is %d\n", n_max_tokens);
        return;
    }

    auto & mem_per_token = ctx->mem_per_token;
    auto & buf_compute   = ctx->buf_compute;

    struct ggml_init_params params = {
        .mem_size = buf_compute.size,
        .mem_buffer = buf_compute.data,
        .no_alloc = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};

    // Embeddings. word_embeddings + token_type_embeddings + position_embeddings
    // std::vector<std::vector<bert_vocab_id>> new_batch_tokens({
    //     {101, 102, 103},
    //     {102, 103, },
    // });
    // float new_embeddings[2][384];
    // n_batch_size = new_batch_tokens.size();
    // int cur_max_len = new_batch_tokens[0].size();
    // N = cur_max_len;
    struct ggml_tensor *token_layer = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    struct ggml_tensor *pad_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, cur_max_len, 1, n_batch_size);
    struct ggml_tensor *positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    struct ggml_tensor *sum = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, cur_max_len, 1, n_batch_size); // the avg pooler
    
    int32_t *token_layer_data = (int32_t *)token_layer->data;
    float *pad_mask_data = (float *)pad_mask->data;
    int32_t *pos_data = (int32_t *)positions->data;
    float *sum_data = (float *)sum->data;
    for (int ba = 0; ba < n_batch_size; ba++) {
        for (int i = 0; i < cur_max_len; i++) {
            int cur_len = n_tokens[ba];
            if (i < cur_len) {
                token_layer_data[ba * cur_max_len + i] = batch_tokens[ba][i];
                pad_mask_data[ba * cur_max_len + i] = 1.0f;
                sum_data[ba * cur_max_len + i] = 1 / (float)cur_len;
            }
            else {
                token_layer_data[ba * cur_max_len + i] = 101; // padding
                pad_mask_data[ba * cur_max_len + i] = 0.0f;
                sum_data[ba * cur_max_len + i] = 0.0f;
            }
            pos_data[ba * cur_max_len + i] = i;
        }
    }

    /*
    for (int ba = 0; ba < n_batch_size; ba++) {
        printf("sample %d in the batch sum_data: \n", ba);
        for (int i = 0; i < cur_max_len; i++)
        {
            printf(" %1.3f ", sum_data[ba * cur_max_len + i]);
        }
        printf("\n");
    }
    */

    /*
    for (int ba = 0; ba < n_batch_size; ba++) {
        printf("sample %d in the batch token_data: ", ba);
        for (int i = 0; i < cur_max_len; i++)
        {
            printf(" %1.3d ", token_layer_data[ba * cur_max_len + i]);
        }
        printf("\n");
    }
    */

    /*
    // print pad mask data
    for (int ba = 0; ba < n_batch_size; ba++) {
        printf("sample %d in the batch pad_mask_data: ", ba);
        for (int i = 0; i < cur_max_len; i++)
        {
            printf(" %1.3f ", pad_mask_data[ba * cur_max_len + i]);
        }
        printf("\n");
    }
    */

    struct ggml_tensor * attn_mask = ggml_mul_mat(ctx0, pad_mask, pad_mask);
    attn_mask = ggml_add1(ctx0, attn_mask, ggml_new_f32(ctx0, -1.0f)); // result -0
    attn_mask = ggml_scale(ctx0, attn_mask, ggml_new_f32(ctx0, 100000.0f)); // BUG: 1e3 will cause overflow?
    attn_mask = ggml_repeat(ctx0, attn_mask, ggml_new_tensor_4d(ctx0, GGML_TYPE_I32, cur_max_len, cur_max_len, n_head, n_batch_size));
    attn_mask = ggml_reshape_3d(ctx0, attn_mask, cur_max_len, cur_max_len, n_head * n_batch_size);

    struct ggml_tensor *token_types = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, cur_max_len * n_batch_size);
    ggml_set_zero(token_types);

    struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.word_embeddings, token_layer);

    inpL = ggml_add(ctx0,
                    ggml_get_rows(ctx0, model.token_type_embeddings, token_types),
                    inpL);
    inpL = ggml_add(ctx0,
                    ggml_get_rows(ctx0, model.position_embeddings, positions),
                    inpL);
    inpL = ggml_reshape_3d(ctx0, inpL, n_embd, cur_max_len, n_batch_size);

    // embd norm
    {
        inpL = ggml_norm(ctx0, inpL);
        inpL = ggml_add(ctx0,
                        ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.ln_e_w, inpL),
                                    inpL),
                        ggml_repeat(ctx0, model.ln_e_b, inpL));
    }

    // layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor *cur = inpL;

        // self-attention
        {
            struct ggml_tensor *Qcur = cur;
            Qcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].q_b, Qcur),
                                            ggml_mul_mat(ctx0, model.layers[il].q_w, Qcur)),
            Qcur = ggml_reshape_4d(ctx0, Qcur,
                                    d_head, n_head, cur_max_len, n_batch_size);
            struct ggml_tensor *Q = ggml_cont(ctx0, ggml_permute(ctx0, Qcur, 0, 2, 1, 3)); // -> [d_head, N, n_head, bs]
            Q = ggml_reshape_3d(ctx0, Q, d_head, cur_max_len, n_head * n_batch_size);

            struct ggml_tensor *Kcur = cur;
            Kcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].k_b, Kcur),
                                            ggml_mul_mat(ctx0, model.layers[il].k_w, Kcur)),
            Kcur = ggml_reshape_4d(ctx0, Kcur,
                                    d_head, n_head, cur_max_len, n_batch_size);
            struct ggml_tensor *K = ggml_cont(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3)); // -> [d_head, N, n_head, bs]
            K = ggml_reshape_3d(ctx0, K, d_head, cur_max_len, n_head * n_batch_size);
            

            struct ggml_tensor *Vcur = cur;
            Vcur = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[il].v_b, Vcur),
                                            ggml_mul_mat(ctx0, model.layers[il].v_w, Vcur)),
            Vcur = ggml_reshape_4d(ctx0, Vcur,
                                    d_head, n_head, cur_max_len, n_batch_size);
            struct ggml_tensor *V = ggml_cont(ctx0, ggml_permute(ctx0, Vcur, 0, 2, 1, 3)); // -> [d_head, N, n_head, bs]
            V = ggml_reshape_3d(ctx0, V, d_head, cur_max_len, n_head * n_batch_size);

            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q); // -> [N, N, n_head * bs]
            // KQ = soft_max(KQ / sqrt(head width))
            KQ = ggml_scale(ctx0,
                            KQ,
                            ggml_new_f32(ctx0, 1.0f / sqrt((float)d_head)));
            
            KQ = ggml_add(ctx0, KQ, attn_mask);
            KQ = ggml_soft_max(ctx0, KQ);

            V = ggml_cont(ctx0, ggml_transpose(ctx0, V)); // -> [N, d_head, n_head* bs]
            // V = ggml_repeat(ctx0, ggml_new_f32(ctx0, 1.999f), V);
            // KQ = ggml_repeat(ctx0, ggml_new_f32(ctx0, 1.999f), KQ);
            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ); // -> [d_head, N, n_head* bs]
            KQV = ggml_reshape_4d(ctx0, KQV, d_head, cur_max_len, n_head, n_batch_size);
            KQV = ggml_cont(ctx0, ggml_permute(ctx0, KQV, 0, 2, 1, 3)); // -> [d_head, n_head, N, n_batch_size]

            cur = ggml_cpy(ctx0,
                            KQV,
                            ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, cur_max_len, n_batch_size));
        }

        // attention output
        cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.layers[il].o_b, cur),
                        ggml_mul_mat(ctx0, model.layers[il].o_w, cur));

        // re-add the layer input
        cur = ggml_add(ctx0, cur, inpL);

        // attention norm
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_att_w, cur),
                                    cur),
                            ggml_repeat(ctx0, model.layers[il].ln_att_b, cur));
        }

        struct ggml_tensor *att_output = cur;

        // intermediate_output = self.intermediate(attention_output)
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_i_w, cur);
        cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ff_i_b, cur),
                        cur);
        cur = ggml_gelu(ctx0, cur);

        // layer_output = self.output(intermediate_output, attention_output)
        cur = ggml_mul_mat(ctx0, model.layers[il].ff_o_w, cur);
        cur = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.layers[il].ff_o_b, cur),
                        cur);

        // attentions bypass the intermediate layer
        cur = ggml_add(ctx0, att_output, cur);

        // output norm
        {
            cur = ggml_norm(ctx0, cur);

            cur = ggml_add(ctx0,
                            ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_out_w, cur),
                                    cur),
                            ggml_repeat(ctx0, model.layers[il].ln_out_b, cur));
        }
        inpL = cur;
    }


    inpL = ggml_cont(ctx0, ggml_transpose(ctx0, inpL));

    // pooler
    inpL = ggml_mul_mat(ctx0, inpL, sum); // [d_embd, N, bs] * [N, 1, bs] -> [d_embd, 1, bs]

    // normalizer
    ggml_tensor *length = ggml_sqrt(ctx0,
                                    ggml_sum_rows(ctx0, ggml_sqr(ctx0, inpL)));  // [1, 1, bs]
    inpL = ggml_div(ctx0, inpL, ggml_repeat(ctx0, length, inpL));
    inpL = ggml_reshape_2d(ctx0, inpL, n_embd, n_batch_size);

    ggml_tensor *output = inpL;

    // run the computation
    ggml_build_forward_expand(&gf, output);
    ggml_graph_compute_with_ctx(ctx0, &gf, n_threads);

    /*
    float *data = ggml_get_data_f32(output);
    printf("\n\n");
    for (int ba = 0; ba < n_batch_size; ba++) {
        for (int i = 0; i < 4; i++) {
            printf("sample[%d] token [%d]: ", ba, i);
            for (int j = 0; j < 10; j++) {
                printf(" %1.3f ", data[ba * n_embd * cur_max_len + i * n_embd + j]);
            }
            printf("\n");
        }
    }
    */

    /*
    float *data = ggml_get_data_f32(output); // attn_mask_data [N, N, n_head * bs]
    printf("\n\n");
    for (int ba = 0; ba < n_batch_size; ba++) {
        printf("sample[%d]: \n", ba);
        for (int token_id = 0; token_id < cur_max_len; token_id++) {
            for (int token_id2 = 0; token_id2 < cur_max_len; token_id2++) {
                printf(" %1.1f ", data[ba * cur_max_len * cur_max_len * n_head + 0 + token_id * cur_max_len + token_id2]);
            }
            printf("\n");
        }
    }
    */

    #ifdef GGML_PERF
        // print timing information per ggml operation (for debugging purposes)
        // requires GGML_PERF to be defined
        ggml_graph_print(&gf);
    #endif

    if (!mem_req_mode) {
        float * output_data = (float *)output->data;
        for (int ba = 0; ba < n_batch_size; ba++){
            memcpy(batch_embeddings[ba], output_data + ba * n_embd, sizeof(float) * n_embd);
        }
    } else {
        mem_per_token = ggml_used_mem(ctx0) / ( cur_max_len * n_batch_size);

        // printf("used_mem = %zu KB \n", ggml_used_mem(ctx0) / 1024);
        // printf("mem_per_token = %zu KB \n", mem_per_token / 1024);
    }

    ggml_free(ctx0);
}

void bert_encode(
    struct bert_ctx * ctx,
            int32_t   n_threads,
         const char * texts,
              float * embeddings) {
    bert_encode_batch(ctx, n_threads, 1, 1, &texts, &embeddings);
}

void bert_encode_batch(
    struct bert_ctx *  ctx,
            int32_t    n_threads,
            int32_t    n_batch_size,
            int32_t    n_inputs,
         const char ** texts,
              float ** embeddings) {

    /*
    if (n_batch_size > n_inputs) {
        n_batch_size = n_inputs;
    }
    if (n_batch_size > ctx->max_batch_n) {
        bert_resize_ctx(ctx, n_batch_size);
        n_batch_size = ctx->max_batch_n;
    }
    */

    int32_t N = bert_n_max_tokens(ctx);

    // Most of this buffer will be unused in typical case where inputs are not that long.
    std::vector<bert_vocab_id> buf_tokens;
    buf_tokens.resize(N * n_inputs);
    std::vector<int32_t> n_tokens = std::vector<int32_t>(n_inputs);
    std::vector<bert_vocab_id*> unsorted_tokens(n_inputs);
    bert_vocab_id* it_tokens = buf_tokens.data();
    for (int i = 0; i < n_inputs; i++) {
        unsorted_tokens[i] = it_tokens;
        bert_tokenize(ctx, texts[i], it_tokens, &n_tokens[i], N);
        it_tokens += n_tokens[i];
    }

    if (n_batch_size == n_inputs) {
        bert_forward_batch(ctx, n_threads, n_batch_size, unsorted_tokens.data(), n_tokens.data(), embeddings);        
    } else {
        // sort the inputs by tokenized length, batch and eval
        std::vector<int> indices;
        indices.reserve(n_inputs);
        for (int i = 0; i < n_inputs; i++)
        {
            indices.push_back(i);
        }

        std::vector<int32_t> sorted_n_tokens = std::vector<int32_t>(n_inputs);

        std::vector<bert_vocab_id *> sorted_tokens(n_inputs);

        std::sort(indices.begin(), indices.end(), [&](int a, int b)
                  { return n_tokens[a] < n_tokens[b]; });

        std::vector<float *> sorted_embeddings(n_inputs);
        memcpy(sorted_embeddings.data(), embeddings, n_inputs * sizeof(float *));

        for (int i = 0; i < n_inputs; i++) {
            sorted_embeddings[i] = embeddings[indices[i]];
            sorted_tokens[i] = unsorted_tokens[indices[i]];
            sorted_n_tokens[i] = n_tokens[indices[i]];
        }

        for (int i = 0; i < n_inputs; i += n_batch_size) {
            if (i + n_batch_size > n_inputs) {
                n_batch_size = n_inputs - i;
            }
            bert_forward_batch(ctx, n_threads, n_batch_size, &sorted_tokens[i], &sorted_n_tokens[i], &sorted_embeddings[i]);
        }
    }
}
