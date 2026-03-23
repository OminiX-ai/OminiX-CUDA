#include "CLI11.hpp"
#include "ctx_manager.h"
#include "fake_infers.hpp"
#include "ggml.h"
#include "gptsovits.h"
#include "hftokenizer.hpp"
#include "infer_session.hpp"
#include "llama.h"
#include "llm.h"
#include "model_defs.h"
#include "model_loader.h"
#include "openvla.h"
#include "proj.hpp"
#include "timer.hpp"
#include "utils.h"
#include "audio_io.h"
#include "vit.hpp"
#include "vits.hpp"
#include <memory>


struct TTSConfig {
    std::string base_dir = "/home/wjr/mount/myy/VoiceDialogue/gguf/";
    std::string ref_audio_path = "/home/wjr/mount/weights/gptsovits/data/ellen_ref.wav";
    std::string ref_text = "It might serve you better to be a little less comfortable. But wherever "
                           "you're listening to this book, please remember to turn off your cell "
                           "phone and that the taking of flash photographs is strictly forbidden.";
    std::string ref_lang = "en";
    std::string target_text = "Today is a good day, let's go playing football.";
    std::string target_lang = "en";
    std::string output_path = "o.wav";
    std::string device_name = "CUDA0";
    int n_threads = 4;
    bool profiling = false;
};

static void run_tts(const TTSConfig& cfg) {
    std::string base_dir = cfg.base_dir;
    if (!base_dir.empty() && base_dir.back() != '/') {
        base_dir += '/';
    }

    std::string cnhubert_model_path = base_dir + "cnhubert.gguf";
    std::string rvq_path = base_dir + "ssl_proj_quantizer.gguf";
    std::string bert_model_path = base_dir + "chinese-roberta-wwm-ext-large.gguf";
    std::string ref_enc_model_path = base_dir + "ref_enc.gguf";
    std::string ar_text_model_path = base_dir + "vits_text.gguf";
    std::string t2s_transformer_path = base_dir + "vits.gguf";
    std::string codebook_model_path = base_dir + "codebook.gguf";
    std::string text_encoder_model_path = base_dir + "text_encoder.gguf";
    std::string flow_model_path = base_dir + "flow.gguf";
    std::string generator_model_path = base_dir + "generator.gguf";
    std::string tokenizer_path = base_dir + "tokenizer.json";

    ContextParams ctx_params = {.device_name = cfg.device_name,
                                .n_threads = cfg.n_threads,
                                .max_nodes = 2048,
                                .verbosity = GGML_LOG_LEVEL_DEBUG};
    LlmParam llm_params = {
        .ngl = 99, .n_ctx = 2048, .tokenizer_path = "", .embeddings = false};

    TTS tts(ar_text_model_path, t2s_transformer_path, llm_params,
            ref_enc_model_path, codebook_model_path, text_encoder_model_path,
            flow_model_path, generator_model_path, cnhubert_model_path,
            rvq_path, bert_model_path, tokenizer_path, ctx_params);

    std::vector<int16_t> out;

    if (cfg.profiling) {
        TTSProfile profile;
        if (tts.run_with_profiling(cfg.ref_audio_path, cfg.ref_text, cfg.ref_lang,
                                   cfg.target_text, cfg.target_lang, out, profile, 1)) {
            profile.print();
            printf("TTS inference successful, output size: %zu\n", out.size());
        } else {
            fprintf(stderr, "TTS inference failed\n");
            return;
        }
    } else {
        if (tts.run(cfg.ref_audio_path, cfg.ref_text, cfg.ref_lang,
                    cfg.target_text, cfg.target_lang, out)) {
            printf("TTS inference successful, output size: %zu\n", out.size());
        } else {
            fprintf(stderr, "TTS inference failed\n");
            return;
        }
    }

    audio_io::save_wav(cfg.output_path, out, 32000);
    printf("Audio saved to: %s\n", cfg.output_path.c_str());
}

int main(int argc, char **argv) {
    CLI::App app{"GPT-SoVITS TTS Inference"};
    argv = app.ensure_utf8(argv);

    TTSConfig cfg;

    app.add_option("-m,--model_dir", cfg.base_dir,
                   "Base directory for GGUF models")
        ->default_val(cfg.base_dir);
    app.add_option("-t,--text", cfg.target_text,
                   "Target text to synthesize")
        ->default_val(cfg.target_text);
    app.add_option("--target_lang", cfg.target_lang,
                   "Target text language (en/zh)")
        ->default_val(cfg.target_lang);
    app.add_option("-r,--ref_audio", cfg.ref_audio_path,
                   "Reference audio file path")
        ->default_val(cfg.ref_audio_path);
    app.add_option("--ref_text", cfg.ref_text,
                   "Reference audio transcript text")
        ->default_val(cfg.ref_text);
    app.add_option("--ref_lang", cfg.ref_lang,
                   "Reference text language (en/zh)")
        ->default_val(cfg.ref_lang);
    app.add_option("-o,--output", cfg.output_path,
                   "Output audio file path")
        ->default_val(cfg.output_path);
    app.add_option("-d,--device", cfg.device_name,
                   "Device name for computation (e.g., CUDA0, CPU)")
        ->default_val(cfg.device_name);
    app.add_option("-n,--n_threads", cfg.n_threads,
                   "Number of threads for computation")
        ->default_val(cfg.n_threads);
    app.add_flag("-p,--profiling", cfg.profiling,
                 "Enable profiling mode");

    CLI11_PARSE(app, argc, argv);

    run_tts(cfg);
    return 0;
}

// int main(int argc, char **argv) {
//   CLI::App app{"OpenVLA Test Suite"};
//   argv = app.ensure_utf8(argv);

//   std::string model_dir = "";
//   std::string dinov2_path = "dinov2.gguf";
//   std::string siglip_path = "siglip.gguf";
//   std::string proj_path = "proj.gguf";
//   std::string llm_path = "llm_q8_0.gguf";
//   std::string action_head_path = "";
//   std::string img_path = "";
//   std::string prompt = "pick up the black bowl between the plate and the "
//                        "ramekin and place it on the plate";
//   std::string tokenizer_path = "";
//   std::string device_name = "CUDA0";
//   int n_threads = 4;
//   int n_ctx = 300;

//   app.add_option(
//       "-m,--model_dir", model_dir,
//       "Base directory for models (default: /mount/weights/vote_model/)");
//   app.add_option(
//       "--dinov2_model", dinov2_path,
//       "DINOv2 model filename in the model directory (default: dinov2.gguf)");
//   app.add_option(
//       "--siglip_model", siglip_path,
//       "Siglip model filename in the model directory (default: siglip.gguf)");
//   app.add_option(
//       "--proj_model", proj_path,
//       "Projection model filename in the model directory (default:
//       proj.gguf)");
//   app.add_option("--action_head_model", action_head_path,
//                  "Action head model filename in the model directory (default:
//                  " "action_head.gguf)");
//   app.add_option(
//       "--llm_model", llm_path,
//       "LLM model filename in the model directory (default: llm_q8_0.gguf)");
//   app.add_option(
//       "-t,--tokenizer", tokenizer_path,
//       "Path to the tokenizer (default: empty, use built-in tokenizer)");
//   app.add_option("-i,--img", img_path, "Path to the input image");
//   app.add_option("-p,--prompt", prompt, "Text prompt for the model");
//   app.add_option("-d,--device", device_name,
//                  "Device name for computation (default: CUDA0)");
//   app.add_option("-n,--n_threads", n_threads,
//                  "Number of threads for computation (default: 4)");
//   app.add_option("-c,--n_ctx", n_ctx, "Context size for LLM (default: 300)");

//   CLI11_PARSE(app, argc, argv);
//   if (model_dir[model_dir.size() - 1] != '/') {
//     model_dir += '/';
//   }
//   if (!model_dir.empty()) {
//     if (model_dir[model_dir.size() - 1] != '/') {
//       model_dir += '/';
//     }
//     dinov2_path = model_dir + dinov2_path;
//     siglip_path = model_dir + siglip_path;
//     proj_path = model_dir + proj_path;
//     llm_path = model_dir + llm_path;
//     if (!action_head_path.empty()) {
//       action_head_path = model_dir + action_head_path;
//     }
//     if (!tokenizer_path.empty()) {
//       tokenizer_path = model_dir + tokenizer_path;
//     }
//   }
//   ContextParams ctx_params = {.device_name = device_name,
//                               .n_threads = n_threads,
//                               .max_nodes = 2048,
//                               .verbosity = GGML_LOG_LEVEL_DEBUG};
//   LlmParam llm_params = {.ngl = 99,
//                          .n_ctx = n_ctx,
//                          .tokenizer_path =
//                              tokenizer_path.empty() ? "" : tokenizer_path,
//                          .embeddings = !action_head_path.empty()};
//   if (action_head_path.empty()) {
//     printf("test openvla...\n");
//     Openvla openvla(dinov2_path, siglip_path, proj_path, llm_path,
//     ctx_params,
//                     llm_params);
//     std::vector<float> output;
//     Timer t(true);
//     for (size_t i = 0; i < 10; i++) {
//       t.start();
//       openvla.run(img_path, prompt, output);
//       printf("Total time: %f\n", t.stop());
//       print_vector(output);
//     }
//   } else {
//     printf("test openvla with regression...\n");
//     OpenvlaWithRegression openvla(dinov2_path, siglip_path, proj_path,
//     llm_path,
//                                   action_head_path, ctx_params, llm_params);
//     std::vector<float> output;
//     Timer t(true);
//     for (size_t i = 0; i < 10; i++) {
//       t.start();
//       printf("prompt: %s\n", prompt.c_str());
//       openvla.run(img_path, prompt, output);
//       printf("Total time: %f\n", t.stop());
//       print_vector(output);
//     }
//   }

//   return 0;
// }
