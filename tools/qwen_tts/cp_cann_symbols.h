#pragma once
// ============================================================================
// Runtime-resolved CANN symbol table.
//
// We do NOT link against libascendcl/libnnopbase/libopapi at build time to
// avoid the circular-dependency issues between CANN's stub libs and the ops
// that ggml-cann loads dynamically. Instead, we dlopen() the three libs on
// first use and resolve each function pointer via dlsym. Callers use the
// pointers in `g_cann` exactly as they would use the direct functions.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>

// Function-pointer table. All members are null-initialized by the global
// `g_cann`; cp_cann_load_symbols() fills them in. Check is_ready() before use.
struct CannSyms {
    // ---- runtime (libascendcl.so) ----
    aclError (*aclInit)(const char *);
    aclError (*aclrtSetDevice)(int32_t);
    aclError (*aclrtCreateStream)(aclrtStream *);
    aclError (*aclrtDestroyStream)(aclrtStream);
    aclError (*aclrtSynchronizeStream)(aclrtStream);
    aclError (*aclrtMalloc)(void **, size_t, aclrtMemMallocPolicy);
    aclError (*aclrtFree)(void *);
    aclError (*aclrtMemcpy)(void *, size_t, const void *, size_t,
                            aclrtMemcpyKind);
    aclError (*aclrtMemcpyAsync)(void *, size_t, const void *, size_t,
                                  aclrtMemcpyKind, aclrtStream);
    const char *(*aclGetRecentErrMsg)();

    // ---- aclnn base (libnnopbase.so) ----
    aclTensor *(*aclCreateTensor)(const int64_t *viewDims, uint64_t viewDimsNum,
                                   aclDataType dataType, const int64_t *stride,
                                   int64_t offset, aclFormat format,
                                   const int64_t *storageDims,
                                   uint64_t storageDimsNum, void *tensorData);
    aclnnStatus (*aclDestroyTensor)(const aclTensor *);
    aclScalar *(*aclCreateScalar)(void *, aclDataType);
    aclnnStatus (*aclDestroyScalar)(const aclScalar *);

    // ---- aclnn ops (libopapi.so) ----
    aclnnStatus (*aclnnMmGetWorkspaceSize)(const aclTensor *, const aclTensor *,
                                            aclTensor *, int8_t, uint64_t *,
                                            aclOpExecutor **);
    aclnnStatus (*aclnnMm)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnAddGetWorkspaceSize)(const aclTensor *, const aclTensor *,
                                             const aclScalar *, aclTensor *,
                                             uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnAdd)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnInplaceAddGetWorkspaceSize)(const aclTensor *,
                                                    const aclTensor *,
                                                    const aclScalar *,
                                                    uint64_t *,
                                                    aclOpExecutor **);
    aclnnStatus (*aclnnInplaceAdd)(void *, uint64_t, aclOpExecutor *,
                                    aclrtStream);

    aclnnStatus (*aclnnInplaceMulGetWorkspaceSize)(aclTensor *,
                                                    const aclTensor *,
                                                    uint64_t *,
                                                    aclOpExecutor **);
    aclnnStatus (*aclnnInplaceMul)(void *, uint64_t, aclOpExecutor *,
                                    aclrtStream);

    aclnnStatus (*aclnnSiluGetWorkspaceSize)(const aclTensor *, aclTensor *,
                                              uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnSilu)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnRmsNormGetWorkspaceSize)(const aclTensor *,
                                                 const aclTensor *, double,
                                                 const aclTensor *,
                                                 const aclTensor *,
                                                 uint64_t *,
                                                 aclOpExecutor **);
    aclnnStatus (*aclnnRmsNorm)(void *, uint64_t, aclOpExecutor *,
                                 aclrtStream);

    // Attention-on-NPU ops (added for the v2 engine — no host roundtrips).
    aclnnStatus (*aclnnBatchMatMulGetWorkspaceSize)(const aclTensor *,
                                                     const aclTensor *,
                                                     aclTensor *, int8_t,
                                                     uint64_t *,
                                                     aclOpExecutor **);
    aclnnStatus (*aclnnBatchMatMul)(void *, uint64_t, aclOpExecutor *,
                                     aclrtStream);

    aclnnStatus (*aclnnSoftmaxGetWorkspaceSize)(const aclTensor *, int64_t,
                                                 aclTensor *, uint64_t *,
                                                 aclOpExecutor **);
    aclnnStatus (*aclnnSoftmax)(void *, uint64_t, aclOpExecutor *,
                                 aclrtStream);

    aclnnStatus (*aclnnMulsGetWorkspaceSize)(const aclTensor *,
                                              const aclScalar *, aclTensor *,
                                              uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnMuls)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // Non-destructive rotary position embedding (single tensor at a time).
    // `mode`: 0 = NEOX half-rotated, 1 = GPT-J interleaved. We use mode 0.
    // This is what ggml-cann invokes for NEOX-style models.
    aclnnStatus (*aclnnRotaryPositionEmbeddingGetWorkspaceSize)(
        const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
        int64_t mode, aclTensor *out, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnRotaryPositionEmbedding)(void *, uint64_t,
                                                 aclOpExecutor *,
                                                 aclrtStream);

    // Dtype conversion (used to cast the F32 input embedding down to F16 at
    // engine entry, and the F16 final hidden back to F32 at exit).
    aclnnStatus (*aclnnCastGetWorkspaceSize)(const aclTensor *self,
                                              aclDataType dtype,
                                              aclTensor *out,
                                              uint64_t *workspaceSize,
                                              aclOpExecutor **executor);
    aclnnStatus (*aclnnCast)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // TensorList creation / destruction — used to wrap a single K/V tensor for
    // the FusedInferAttention op (which takes lists to support chunked KV).
    aclTensorList *(*aclCreateTensorList)(const aclTensor *const *value,
                                           uint64_t size);
    aclnnStatus (*aclDestroyTensorList)(const aclTensorList *);

    // Mega-fused attention — the exact op llama.cpp/ggml-cann uses. Replaces
    // my manual BMM+Muls+Softmax+BMM path. Natural BSND layout, GQA handled
    // internally via numKeyValueHeads, softmax in F32 regardless of input
    // precision. See aclnn_fused_infer_attention_score_v2.h for the full
    // parameter list.
    aclnnStatus (*aclnnFusedInferAttentionScoreV2GetWorkspaceSize)(
        const aclTensor *query, const aclTensorList *key,
        const aclTensorList *value, const aclTensor *pseShiftOptional,
        const aclTensor *attenMaskOptional,
        const aclIntArray *actualSeqLengthsOptional,
        const aclIntArray *actualSeqLengthsKvOptional,
        const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional,
        const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
        const aclTensor *quantOffset2Optional,
        const aclTensor *antiquantScaleOptional,
        const aclTensor *antiquantOffsetOptional,
        const aclTensor *blockTableOptional,
        const aclTensor *queryPaddingSizeOptional,
        const aclTensor *kvPaddingSizeOptional,
        const aclTensor *keyAntiquantScaleOptional,
        const aclTensor *keyAntiquantOffsetOptional,
        const aclTensor *valueAntiquantScaleOptional,
        const aclTensor *valueAntiquantOffsetOptional,
        const aclTensor *keySharedPrefixOptional,
        const aclTensor *valueSharedPrefixOptional,
        const aclIntArray *actualSharedPrefixLenOptional, int64_t numHeads,
        double scaleValue, int64_t preTokens, int64_t nextTokens,
        char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
        int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode,
        bool softmaxLseFlag, int64_t keyAntiquantMode,
        int64_t valueAntiquantMode, const aclTensor *attentionOut,
        const aclTensor *softmaxLse, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnFusedInferAttentionScoreV2)(void *, uint64_t,
                                                    aclOpExecutor *,
                                                    aclrtStream);

    bool is_ready() const { return aclrtMalloc != nullptr; }
};

extern CannSyms g_cann;

// Idempotent. Dlopens libascendcl.so, libnnopbase.so, libopapi.so and resolves
// every member of g_cann. Returns true on success. On failure, prints which
// symbol / library could not be loaded and leaves g_cann in a partial state
// (caller must check is_ready()).
bool cp_cann_load_symbols();
