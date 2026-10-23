// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "function_pool.h"
#include <gtest/gtest.h>
#include <stdexcept>

// Exercises add/lookup/miss paths.  The LDS-only constructor avoids needing
// a live GPU.
TEST(rocfft_internal, function_pool_runtime_paths)
{
    // The constructor rejects 0.
    constexpr unsigned int lds_bytes = 64 * 1024;
    function_pool          pool(lds_bytes);

    // A prime length the AOT generator shouldn't have populated.
    FMKey runtime_key(99991, rocfft_precision_single, CS_KERNEL_STOCKHAM);
    if(pool.has_function(runtime_key))
        GTEST_SKIP() << "runtime_key unexpectedly pre-populated";

    EXPECT_THROW(pool.get_kernel(runtime_key), std::out_of_range);
    pool.add_new_kernel(runtime_key);
    EXPECT_TRUE(pool.has_function(runtime_key));
    EXPECT_NO_THROW(pool.get_kernel(runtime_key));

    // PPFMKey overload, never populated at runtime.
    PPFMKey pp_key(
        99991, 1, 1, rocfft_precision_single, rocfft_transform_type_complex_forward, CS_3D_PP);
    EXPECT_FALSE(pool.has_function(pp_key, 1));
    EXPECT_THROW(pool.get_kernel(pp_key, CS_3D_PP, 1), std::out_of_range);
}

// Configs from a tuning file skip the generator pass that sizes the
// workgroup to match the rest of the config, so the pool checks it.
TEST(rocfft_internal, function_pool_rejects_inconsistent_workgroup)
{
    constexpr unsigned int lds_bytes = 64 * 1024;
    function_pool          pool(lds_bytes);

    KernelConfig config;
    config.factors               = {8, 8};
    config.threads_per_transform = {8, 0};
    config.transforms_per_block  = 4;

    auto build = [&config](int wgs, ComputeScheme scheme) {
        config.workgroup_size = wgs;
        FFTKernel kernel(config, scheme);
        return kernel.workgroup_size;
    };

    // 8 threads for each of 4 transforms is 32 threads of work.
    EXPECT_EQ(build(32, CS_KERNEL_STOCKHAM), 32);

    // A wider workgroup gives the extra threads a transform index
    // past the end of the block; a narrower one drops transforms.
    EXPECT_THROW(build(64, CS_KERNEL_STOCKHAM), std::runtime_error);
    EXPECT_THROW(build(16, CS_KERNEL_STOCKHAM), std::runtime_error);

    // 2D_SINGLE sizes its workgroup from the second dimension.
    EXPECT_EQ(build(64, CS_KERNEL_2D_SINGLE), 64);

    // An empty config carries no sizes to disagree about.
    EXPECT_NO_THROW(pool.add_new_kernel(
        FMKey(99989, rocfft_precision_single, CS_KERNEL_STOCKHAM)));

    // Reaching the pool through a key is the path a tuning file takes.
    config.workgroup_size = 64;
    EXPECT_THROW(
        pool.add_new_kernel(
            FMKey(99989, rocfft_precision_single, CS_KERNEL_STOCKHAM, NONE, config)),
        std::runtime_error);
}
