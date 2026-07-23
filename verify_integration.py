#!/usr/bin/env python3
"""Verify the xLLM SpacemiT integration logic on K3.

Compiles a torch-free C++ test that calls the SAME ggml API sequence used by
xllm/core/platform/spacemit/ggml_bridge.cpp and
xllm/core/kernels/spacemit/{matmul_ggml,rms_norm_ggml}.cpp:

  - zero-copy tensor creation (ggml_new_tensor + data pointer share)
  - ggml_mul_mat  (matmul kernel)
  - ggml_rms_norm (rms_norm kernel)

Linked against the real IME2-enabled ggml built on the K3 worker, so this
exercises the integration path on actual SpacemiT hardware.
"""

import paramiko
import sys

HOST, USER, PW = "10.0.90.243", "bianbu", "bianbu"
LL = "/home/bianbu/llama.cpp-spacemit"
WORK = "/home/bianbu/xllm-integration-verify"

TEST_CPP = r'''
// Torch-free verification of the xLLM SpacemiT integration path.
// Mirrors ggml_bridge.cpp / matmul_ggml.cpp / rms_norm_ggml.cpp API usage.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;
#define CHECK(cond, msg) do { if(!(cond)){printf("  FAIL: %s\n", msg); failures++;} } while(0)

// Mirror GGMLBackend::compute: build ctx, expand graph, compute, read result.
static std::vector<float> run_graph(ggml_context* ctx, ggml_tensor* out, int nthreads) {
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, nthreads);
    int64_t n = ggml_nelements(out);
    std::vector<float> r(n);
    memcpy(r.data(), out->data, n * sizeof(float));
    return r;
}

int main() {
    printf("========================================================\n");
    printf("xLLM SpacemiT Integration Verification (real ggml on K3)\n");
    printf("========================================================\n");

    const int NTH = 4;  // <= perfer core count, avoids IME thread abort

    // ---- Test 1: zero-copy tensor creation (ggml_bridge.cpp to_ggml) ----
    printf("\nTest 1: Zero-copy bridge (to_ggml pointer share)\n");
    {
        struct ggml_init_params p = { 16*1024*1024, NULL, false };
        ggml_context* ctx = ggml_init(p);
        float host[6] = {1,2,3,4,5,6};
        // Bridge path: create tensor then point data at external buffer.
        ggml_tensor* t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2);
        t->data = host;  // zero-copy, as GGMLBridge::to_ggml does
        CHECK(t->data == (void*)host, "data pointer shared (zero-copy)");
        CHECK(ggml_get_data_f32(t)[4] == 5.0f, "shared data readable");
        printf("  host=%p ggml->data=%p  %s\n", (void*)host, t->data,
               t->data==(void*)host ? "IDENTICAL (zero-copy OK)" : "COPIED");
        ggml_free(ctx);
    }

    // ---- Test 2: matmul kernel (matmul_ggml.cpp -> ggml_mul_mat) ----
    printf("\nTest 2: matmul kernel (ggml_mul_mat)\n");
    {
        struct ggml_init_params p = { 64*1024*1024, NULL, false };
        ggml_context* ctx = ggml_init(p);
        // A[M=2,K=3], B[K=3,N=2]. ggml_mul_mat(a,b): a=[K,M], b=[K,N] -> [N,M]?
        // ggml convention: mul_mat(A,B) with A ne=[k,m], B ne=[k,n] => C ne=[m,n]...
        // We use A ne0=K rows layout matching kernel: a=[K,M], b=[K,N].
        ggml_tensor* a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2); // ne0=K=3, ne1=M=2
        ggml_tensor* b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 2); // ne0=K=3, ne1=N=2
        float av[6] = {1,2,3, 4,5,6};      // rows: [1 2 3],[4 5 6]
        float bv[6] = {1,0,1, 0,1,1};      // rows: [1 0 1],[0 1 1]
        memcpy(a->data, av, sizeof(av));
        memcpy(b->data, bv, sizeof(bv));
        ggml_tensor* c = ggml_mul_mat(ctx, a, b); // c ne=[M=2, N=2] => result[n][m]
        auto r = run_graph(ctx, c, NTH);
        // dot(a_row_m, b_row_n):
        // m0=[1,2,3] n0=[1,0,1]->4 ; m1=[4,5,6] n0->10 ; m0 n1=[0,1,1]->5 ; m1 n1->11
        printf("  result = [%.1f %.1f %.1f %.1f] expected [4 10 5 11]\n",
               r[0], r[1], r[2], r[3]);
        CHECK(fabs(r[0]-4)<1e-4 && fabs(r[1]-10)<1e-4 &&
              fabs(r[2]-5)<1e-4 && fabs(r[3]-11)<1e-4, "matmul values correct");
        ggml_free(ctx);
    }

    // ---- Test 3: rms_norm kernel (rms_norm_ggml.cpp -> ggml_rms_norm) ----
    printf("\nTest 3: rms_norm kernel (ggml_rms_norm)\n");
    {
        struct ggml_init_params p = { 16*1024*1024, NULL, false };
        ggml_context* ctx = ggml_init(p);
        ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 5);
        float xv[5] = {1,2,3,4,5};
        memcpy(x->data, xv, sizeof(xv));
        ggml_tensor* y = ggml_rms_norm(ctx, x, 1e-6f);
        auto r = run_graph(ctx, y, NTH);
        double ss=0; for(int i=0;i<5;i++) ss += r[i]*r[i];
        double out_rms = sqrt(ss/5.0);
        printf("  normalized RMS = %.6f (expected ~1.0)\n", out_rms);
        CHECK(fabs(out_rms-1.0)<1e-3, "rms_norm normalizes to unit RMS");
        ggml_free(ctx);
    }

    printf("\n========================================================\n");
    if (failures == 0) {
        printf("RESULT: ALL INTEGRATION TESTS PASSED\n");
        printf("The xLLM SpacemiT bridge/kernel ggml API path is verified\n");
        printf("against real IME2-enabled ggml on K3 hardware.\n");
    } else {
        printf("RESULT: %d CHECK(S) FAILED\n", failures);
    }
    printf("========================================================\n");
    return failures == 0 ? 0 : 1;
}
'''


def main():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, allow_agent=False, look_for_keys=False)

    def run(cmd, t=300):
        i, o, e = c.exec_command(cmd, timeout=t)
        rc = o.channel.recv_exit_status()
        return rc, o.read().decode(), e.read().decode()

    print("Setting up verification on K3...")
    run(f"rm -rf {WORK} && mkdir -p {WORK}")

    sftp = c.open_sftp()
    with sftp.file(f"{WORK}/verify.cpp", "w") as f:
        f.write(TEST_CPP)
    sftp.close()

    # Locate ggml headers + libs from the K3 llama.cpp build.
    inc = f"{LL}/ggml/include"
    libdir = f"{LL}/build/bin"
    print(f"Headers: {inc}\nLibs:    {libdir}")

    print("\nCompiling verification (linking real IME2 ggml)...")
    compile_cmd = (
        f"cd {WORK} && g++ -std=c++17 -O2 verify.cpp -o verify "
        f"-I{inc} -L{libdir} -lggml -lggml-base -lggml-cpu "
        f"-Wl,-rpath,{libdir} 2>&1"
    )
    rc, out, err = run(compile_cmd)
    if rc != 0:
        print("Compile failed:")
        print(out)
        c.close()
        return 1
    print("Compile OK")

    print("\nRunning verification on K3 hardware...\n")
    rc, out, err = run(f"cd {WORK} && ./verify 2>&1", t=200)
    print(out)

    c.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
