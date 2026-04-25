// ============================================================================
// Phase 2.1 smoke test for TalkerCudaEngine.
//
// Usage:
//   test_talker_cuda_init <gguf_path>           # full init from GGUF
//   test_talker_cuda_init --noweights           # scratch alloc + RoPE only
//
// Exit 0 on scaffold-init OK; non-zero otherwise. The binary intentionally
// does NOT call forward_decode/forward_prefill (those are Phase 2.2).
// ============================================================================

#include "talker_cuda_engine.h"
#include "../talker.h"  // TalkerConfig

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_talker_cuda_init <gguf_path | --noweights>\n");
        return 2;
    }
    const std::string arg1 = argv[1];
    const bool noweights = (arg1 == "--noweights");

    TalkerConfig cfg;  // defaults: 28L / 16Q / 8KV / 2048 hidden / 6144 inter
    ominix_cuda::TalkerCudaEngine eng;
    bool ok;
    if (noweights) {
        // Phase 2.1 scaffold currently always opens GGUF, but a "--noweights"
        // mode is left here for the next sub-phase when we wire that branch.
        // For now we still need a path; print a clear error.
        fprintf(stderr,
                "[smoke] --noweights mode requires a real GGUF in Phase 2.1; "
                "rerun with the model path\n");
        return 2;
    } else {
        ok = eng.init_from_gguf(arg1, cfg, /*device=*/0);
    }
    if (!ok) {
        fprintf(stderr, "[smoke] init_from_gguf FAILED\n");
        return 1;
    }
    eng.reset_kv_cache();
    eng.set_rope_speed_factor(1.0f);

    fprintf(stderr,
            "[smoke] Phase 2.1 scaffold init PASS  ready=%d  "
            "use_cuda_graphs=%d  use_int8=%d  use_fp8=%d\n",
            (int)eng.is_ready(),
            (int)eng.use_cuda_graphs(),
            (int)eng.use_int8_weights(),
            (int)eng.use_fp8_weights());
    return 0;
}
