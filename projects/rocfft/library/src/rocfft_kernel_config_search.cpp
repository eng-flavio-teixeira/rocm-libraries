// Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

//
// A simple standalone interface to tune rocFFT single kernel
//
// Usage examples:
//
//     -- brute-force tuning
//        rocfft_kernel_config_search brute-force -l 64
//     -- manual tuning
//        rocfft_kernel_config_search manual -l 64 -b 1 -f 8 8 -w 64 --tpt 2 --half-lds 1 --direct-reg 1
//

#include "../../shared/CLI11.hpp"
#include "../../shared/arithmetic.h"
#include "../../shared/device_properties.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "device/generator/stockham_gen.h"
#include "rtc_compile.h"
#include "rtc_kernel.h"
#include "rtc_stockham_gen.h"
#include "rtc_stockham_kernel.h"
#include "../../shared/concurrency.h"
#include "rtc_subprocess.h"
#include "rtc_twiddle_kernel.h"
#include <algorithm>
#include <filesystem>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <thread>

static const std::vector<unsigned int> supported_factors
    = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};

// A workgroup computes transforms_per_block transforms using
// threads_per_transform threads each, so the workgroup size is
// derived from those two rather than chosen independently.  Nothing
// in the shipped configs is wider than this.
static const unsigned int max_wgs = 512;

// Candidates whose output differs from a reference DFT by more than
// this are discarded rather than timed.  Indexing mistakes give
// errors of order 1, so this only needs to be well clear of the
// ~1e-6 that single-precision rounding accounts for.
static const double max_rel_error = 1e-3;

// recursively find all unique factorizations of given length.  each
// factorization is a vector of ints, sorted so they're uniquified in
// a set.
std::set<std::vector<unsigned int>> factorize(unsigned int length)
{
    std::set<std::vector<unsigned int>> ret;
    for(auto factor : supported_factors)
    {
        if(length % factor == 0)
        {
            unsigned int remain = length / factor;
            if(remain == 1)
                ret.insert({factor});
            else
            {
                // recurse into remainder
                auto remain_factorization = factorize(remain);
                for(auto& remain_factors : remain_factorization)
                {
                    std::vector<unsigned int> factors{factor};
                    std::copy(
                        remain_factors.begin(), remain_factors.end(), std::back_inserter(factors));
                    std::sort(factors.begin(), factors.end());
                    ret.insert(factors);
                }
            }
        }
    }
    return ret;
}

// recursively return power set of a range of ints
std::set<std::vector<unsigned int>> power_set(std::vector<unsigned int>::const_iterator begin,
                                              std::vector<unsigned int>::const_iterator end)
{
    std::set<std::vector<unsigned int>> ret;
    // either include the front element in the output, or don't
    if(std::distance(begin, end) == 1)
    {
        ret.insert({*begin});
        // an explicit empty vector, not insert({}) - a braced empty
        // list binds to the initializer_list overload and inserts
        // nothing, which drops every subset that omits the last
        // factor and makes the result depend on the factor order
        ret.emplace();
    }
    else
    {
        // recurse into the remainder
        auto remain = power_set(begin + 1, end);
        for(auto r : remain)
        {
            ret.insert(r);
            r.push_back(*begin);
            ret.insert(r);
        }
    }
    return ret;
}

std::set<unsigned int, std::greater<unsigned int>>
    supported_threads_per_transform(const std::vector<unsigned int>& factorization)
{
    std::set<unsigned int, std::greater<unsigned int>> tpts;
    auto tpt_candidates = power_set(factorization.begin(), factorization.end());
    for(auto tpt : tpt_candidates)
    {
        if(tpt.empty())
            continue;
        tpts.insert(product(tpt.begin(), tpt.end()));
    }
    return tpts;
}

std::string test_kernel_name(unsigned int                     length,
                             const std::vector<unsigned int>& factorization,
                             unsigned int                     wgs,
                             unsigned int                     tpt,
                             bool                             half_lds,
                             bool                             direct_to_from_reg)
{
    std::string ret = "fft_test_len_";
    ret += std::to_string(length);
    ret += "_factors";
    for(auto f : factorization)
    {
        ret += "_";
        ret += std::to_string(f);
    }
    ret += "_wgs_";
    ret += std::to_string(wgs);
    ret += "_tpt_";
    ret += std::to_string(tpt);
    if(half_lds)
        ret += "_halfLds";
    if(direct_to_from_reg)
        ret += "_dirReg";

    return ret;
}

std::string test_kernel_src(const std::string&               kernel_name,
                            hipDeviceProp_t                  device_prop,
                            unsigned int&                    transforms_per_block,
                            unsigned int                     length,
                            ComputeScheme                    compute_scheme,
                            rocfft_precision                 precision,
                            const std::vector<unsigned int>& factorization,
                            unsigned int                     wgs,
                            unsigned int                     tpt,
                            bool                             half_lds,
                            bool                             direct_to_from_reg)
{
    StockhamGeneratorSpecs specs{KIntType::U32,
                                 factorization,
                                 {},
                                 static_cast<unsigned int>(rocfft_precision_single),
                                 get_curr_gcn_arch_name(),
                                 wgs,
                                 PrintScheme(compute_scheme)};

    auto ppParams = StockhamPartialPassParams();

    specs.threads_per_transform = tpt;
    specs.half_lds              = half_lds;
    specs.direct_to_from_reg    = direct_to_from_reg;
    // the search already chose transforms_per_block and sized wgs to
    // match, so take wgs as-is instead of letting the generator
    // re-derive it from the LDS budget
    specs.wgs_is_derived = true;
    // aim for occupancy-2
    specs.lds_byte_limit = device_prop.sharedMemPerBlock / 2;

    return stockham_rtc(specs,
                        specs,
                        ppParams,
                        &transforms_per_block,
                        kernel_name,
                        compute_scheme,
                        -1,
                        precision,
                        rocfft_placement_notinplace,
                        rocfft_array_type_complex_interleaved,
                        rocfft_array_type_complex_interleaved,
                        true,
                        0,
                        0,
                        false,
                        direct_to_from_reg ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                                           : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,
                        IntrinsicAccessType::DISABLE_BOTH,
                        SBRC_TRANSPOSE_TYPE::NONE,
                        CallbackType::NONE,
                        BluesteinFuseType::BFT_NONE,
                        PartialPassType::PPT_NONE,
                        {},
                        {});
}

// things that we need to remember between kernel launches
struct device_data_t
{
    std::vector<rocfft_complex<float>> host_input_buf;
    gpubuf_t<rocfft_complex<float>>    twiddles;
    gpubuf_t<rocfft_complex<float>>    input_buf;
    gpubuf_t<rocfft_complex<float>>    output_buf;
    gpubuf_t<unsigned int>             lengths;
    gpubuf_t<unsigned int>             stride_in;
    gpubuf_t<unsigned int>             stride_out;
    size_t                             batch;
    hipEvent_wrapper_t                 start;
    hipEvent_wrapper_t                 stop;

    device_data_t()
    {
        start.alloc();
        stop.alloc();
    }
    ~device_data_t() = default;
};

// run the kernel, returning the median execution time
float launch_kernel(RTCKernel&             kernel,
                    unsigned int           blocks,
                    unsigned int           wgs,
                    unsigned int           lds_bytes,
                    unsigned int           ntrial,
                    const hipDeviceProp_t& prop,
                    device_data_t&         data)
{
    RTCKernelArgs kargs;
    kargs.append_ptr(data.twiddles.data());
    kargs.append_size_t(1);
    kargs.append_ptr(data.lengths.data());
    kargs.append_ptr(data.stride_in.data());
    kargs.append_ptr(data.stride_out.data());
    kargs.append_size_t(data.batch);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_kint(0, KIntType::U32);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(data.input_buf.data());
    kargs.append_ptr(data.output_buf.data());
    std::vector<float> times;
    for(unsigned int i = 0; i < ntrial; ++i)
    {
        // simulate rocfft-bench behaviour - memcpy input to device
        // before each execution
        if(hipMemcpy(data.input_buf.data(),
                     data.host_input_buf.data(),
                     data.host_input_buf.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");

        if(hipEventRecord(data.start) != hipSuccess)
            throw std::runtime_error("hipEventRecord start failed");
        kernel.launch(kargs, {blocks}, {wgs}, lds_bytes, prop);
        if(hipEventRecord(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord stop failed");
        if(hipEventSynchronize(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
        float time;
        if(hipEventElapsedTime(&time, data.start, data.stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");
        times.push_back(time);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

unsigned int get_lds_bytes(unsigned int length, unsigned int transforms_per_block, bool half_lds)
{
    // assume single precision complex
    return length * transforms_per_block * sizeof(rocfft_complex<float>) / (half_lds ? 2 : 1);
}

size_t batch_size(unsigned int length)
{
    // target 2 GiB memory usage (2^31), assume single precision so
    // each element is 2^3 bytes
    size_t target_elems = 1U << 28;
    return target_elems / length;
}

std::vector<rocfft_complex<float>> create_input_buf(unsigned int length, size_t batch)
{
    auto                               elems = length * batch;
    std::vector<rocfft_complex<float>> buf;
    buf.reserve(elems);
    std::mt19937 gen;
    for(unsigned int i = 0; i < elems; ++i)
    {
        float x = static_cast<float>(gen()) / static_cast<float>(gen.max());
        float y = static_cast<float>(gen()) / static_cast<float>(gen.max());
        buf.push_back({x, y});
    }
    return buf;
}

gpubuf_t<rocfft_complex<float>> create_device_buf(unsigned int length, size_t batch)
{
    auto                            elems = length * batch;
    gpubuf_t<rocfft_complex<float>> device_buf;
    if(device_buf.alloc(elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemset(device_buf.data(), 0, elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMemset");

    return device_buf;
}

// The kernels are generated with KIntType::U32, so they read lengths
// and strides as arrays of unsigned int.  These buffers have to use
// the same element type or the kernel sees every second value.
gpubuf_t<unsigned int> create_lengths(unsigned int length)
{
    gpubuf_t<unsigned int> device_buf;
    if(device_buf.alloc(sizeof(unsigned int)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");

    if(hipMemcpy(device_buf.data(), &length, sizeof(unsigned int), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

gpubuf_t<unsigned int> create_strides(unsigned int length)
{
    std::array<unsigned int, 2> strides{1, length};
    gpubuf_t<unsigned int>      device_buf;
    if(device_buf.alloc(sizeof(unsigned int) * 2) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(
           device_buf.data(), strides.data(), 2 * sizeof(unsigned int), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

// Build the twiddle table a Stockham kernel expects for a given
// factorization.  This mirrors TwiddleTable in twiddles.cpp, which
// lives in the rocFFT library that this tool doesn't link against.
// The table layout follows the order of the radices, so it has to be
// rebuilt for each permutation of the factors.
gpubuf_t<rocfft_complex<float>> create_twiddles(const std::vector<unsigned int>& factorization,
                                                rocfft_precision                 precision,
                                                const hipDeviceProp_t&           device_prop)
{
    const std::vector<size_t> radices(factorization.begin(), factorization.end());

    std::vector<size_t> radices_prod;
    std::vector<size_t> radices_sum_prod{0};
    size_t              prod      = 1;
    size_t              prod_next = radices.front();
    size_t              sum       = 0;
    for(size_t i = 0; i + 1 < radices.size(); ++i)
    {
        prod *= radices[i];
        prod_next *= radices[i + 1];
        sum += prod * (radices[i + 1] - 1);
        radices_sum_prod.push_back(sum);
        radices_prod.push_back(prod_next);
    }
    if(radices_prod.empty())
        radices_prod.push_back(radices.front());

    const auto last = radices.size() - 1;
    size_t     table_sz;
    if(last)
    {
        const auto last_radix = radices[last];
        table_sz              = radices_sum_prod[last - 1]
                   + ((radices_prod[last - 1] / last_radix) - 1) * (last_radix - 1)
                   + (last_radix - 1);
    }
    else
    {
        table_sz = radices_sum_prod[0];
    }

    // a single-radix transform needs no twiddles at all
    gpubuf_t<rocfft_complex<float>> twiddles;
    if(table_sz == 0)
        return twiddles;
    if(twiddles.alloc(table_sz * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc twiddles");

    radices_t radices_arg{};
    radices_t radices_prod_arg{};
    radices_t radices_sum_prod_arg{};
    std::copy(radices.begin(), radices.end(), radices_arg.data);
    std::copy(radices_prod.begin(), radices_prod.end(), radices_prod_arg.data);
    std::copy(radices_sum_prod.begin(), radices_sum_prod.end(), radices_sum_prod_arg.data);

    RTCKernelArgs kargs(RTCKernelTwiddle::default_itype);
    kargs.append_kint(product(radices.begin(), radices.end()));
    kargs.append_kint(radices.size());
    kargs.append_struct(radices_arg);
    kargs.append_struct(radices_prod_arg);
    kargs.append_struct(radices_sum_prod_arg);
    kargs.append_ptr(twiddles.data());

    const auto max_radix_prod = *std::max_element(radices_prod.begin(), radices_prod.end());
    const auto min_radix      = *std::min_element(radices.begin(), radices.end());

    auto kernel
        = RTCKernelTwiddle::generate(device_prop.gcnArchName, TwiddleTableType::RADICES, precision);
    kernel.get()->launch(
        kargs,
        dim3(static_cast<unsigned int>(DivRoundingUp<size_t>(radices.size(), TWIDDLES_THREADS)),
             static_cast<unsigned int>(
                 DivRoundingUp<size_t>(max_radix_prod / min_radix, TWIDDLES_THREADS))),
        dim3(TWIDDLES_THREADS, TWIDDLES_THREADS),
        0,
        device_prop);
    if(hipDeviceSynchronize() != hipSuccess)
        throw std::runtime_error("hipDeviceSynchronize failed");

    return twiddles;
}

// Spell an arch name the way the kernel config files do, e.g.
// "gfx90a" -> "supported_arch.GFX_90A.value".  A tuned config needs
// to name the arch it was tuned on, otherwise the generator treats it
// as generic and re-derives transforms_per_block from its own LDS
// budget instead of using the tuned value.
std::string config_arch_name(const std::string& gcn_arch_name)
{
    auto suffix = gcn_arch_name.substr(gcn_arch_name.find("gfx") == 0 ? 3 : 0);
    for(auto& c : suffix)
        c = std::toupper(c);
    return "supported_arch.GFX_" + suffix + ".value";
}

// Naive DFT of the first few transforms of the input, used as the
// reference for checking candidate kernels.  The input is the same
// for every candidate, so this only needs computing once per length.
std::vector<rocfft_complex<double>> reference_dft(
    const std::vector<rocfft_complex<float>>& input, unsigned int length, size_t ntransform)
{
    // TWO_PI is already negative, matching the forward transform the
    // twiddle kernel generates
    std::vector<rocfft_complex<double>> twiddle(length);
    for(unsigned int k = 0; k < length; ++k)
    {
        const double angle = TWO_PI * static_cast<double>(k) / length;
        twiddle[k]         = {std::cos(angle), std::sin(angle)};
    }

    std::vector<rocfft_complex<double>> ret(static_cast<size_t>(length) * ntransform);
    for(size_t t = 0; t < ntransform; ++t)
    {
        const auto* in  = input.data() + t * length;
        auto*       out = ret.data() + t * length;
        for(unsigned int k = 0; k < length; ++k)
        {
            double re = 0.0;
            double im = 0.0;
            for(unsigned int j = 0; j < length; ++j)
            {
                const auto& w = twiddle[(static_cast<size_t>(j) * k) % length];
                re += in[j].x * w.x - in[j].y * w.y;
                im += in[j].x * w.y + in[j].y * w.x;
            }
            out[k] = {re, im};
        }
    }
    return ret;
}

// Relative RMS error of a candidate's output against the reference.
// Indexing mistakes produce errors of order 1, far above the ~1e-6
// that single-precision rounding accounts for.
double relative_error(const std::vector<rocfft_complex<float>>&  result,
                      const std::vector<rocfft_complex<double>>& reference)
{
    double err = 0.0;
    double ref = 0.0;
    for(size_t i = 0; i < reference.size(); ++i)
    {
        const double dx = result[i].x - reference[i].x;
        const double dy = result[i].y - reference[i].y;
        err += dx * dx + dy * dy;
        ref += reference[i].x * reference[i].x + reference[i].y * reference[i].y;
    }
    return ref > 0.0 ? std::sqrt(err / ref) : std::sqrt(err);
}

// One kernel the search wants to measure.  Generating and compiling
// these is the bulk of the search's runtime and is independent per
// candidate, so it happens on several threads; timing them has to
// stay serial to mean anything.
struct candidate
{
    unsigned int tpt;
    unsigned int tpb;
    unsigned int wgs;
    bool         half_lds;
    bool         direct_to_from_reg;
    std::string  kernel_name;

    // filled in by the compile pass
    unsigned int      transforms_per_block = 0;
    std::vector<char> code;
    std::string       error;
};

// Every kernel worth measuring for one ordering of one factorization.
// The generator derives transforms_per_block as workgroup_size /
// threads_per_transform, so pick it here and size the workgroup to
// match exactly, leaving no idle threads in the block.  A workgroup
// wider than tpt*tpb has threads whose transform index runs past the
// block, and they alias onto an in-block LDS slot.
std::vector<candidate> enumerate_candidates(unsigned int                     length,
                                            const std::vector<unsigned int>& factorization,
                                            unsigned int                     min_tpb,
                                            unsigned int                     max_tpb,
                                            unsigned int                     wgs_floor,
                                            unsigned int                     wgs_limit,
                                            unsigned int                     lds_byte_limit)
{
    std::vector<candidate> ret;
    for(auto tpt : supported_threads_per_transform(factorization))
    {
        for(unsigned int tpb = min_tpb; tpb <= max_tpb; ++tpb)
        {
            const unsigned int wgs = tpt * tpb;
            // every larger tpb only makes the workgroup bigger
            if(wgs > wgs_limit)
                break;
            if(wgs < wgs_floor)
                continue;

            for(bool half_lds : {true, false})
            {
                for(bool direct_to_from_reg : {true, false})
                {
                    // half lds currently requires direct to/from reg
                    if(half_lds && !direct_to_from_reg)
                        continue;
                    // all of the block's transforms have to fit in LDS
                    if(get_lds_bytes(length, tpb, half_lds) > lds_byte_limit)
                        continue;

                    ret.push_back({tpt,
                                   tpb,
                                   wgs,
                                   half_lds,
                                   direct_to_from_reg,
                                   test_kernel_name(length,
                                                    factorization,
                                                    wgs,
                                                    tpt,
                                                    half_lds,
                                                    direct_to_from_reg)});
                }
            }
        }
    }
    return ret;
}

// The helper that compiles one kernel per invocation.  find_rtc_helper
// locates it next to the installed library, which this tool doesn't
// link against, so look next to this executable instead.
std::optional<std::string> locate_rtc_helper(const char* argv0)
{
    std::error_code ec;
    auto            candidate
        = std::filesystem::absolute(argv0, ec).parent_path() / "rocfft_rtc_helper";
    if(!ec && std::filesystem::exists(candidate, ec))
        return candidate.string();
    return std::nullopt;
}

// Generate and compile a batch of candidates in parallel.  hipRTC
// holds a process-wide lock, so compiling in threads is no faster
// than doing it serially - the work has to go to separate processes
// to overlap, which is what compile_subprocess is for.  A candidate
// the generator or compiler rejects records the reason and is skipped
// rather than aborting the search.
void compile_candidates(std::vector<candidate>&           candidates,
                        const hipDeviceProp_t&            device_prop,
                        unsigned int                      length,
                        ComputeScheme                     compute_scheme,
                        rocfft_precision                  precision,
                        const std::vector<unsigned int>&  factorization,
                        unsigned int                      njob,
                        const std::optional<std::string>& rtc_helper)
{
    std::atomic<size_t> next{0};
    const size_t        nthread = std::min<size_t>(njob, candidates.size());

    std::vector<std::thread> threads;
    threads.reserve(nthread);
    for(size_t t = 0; t < nthread; ++t)
    {
        threads.emplace_back([&]() {
            for(size_t i = next++; i < candidates.size(); i = next++)
            {
                auto& c = candidates[i];
                try
                {
                    auto src = test_kernel_src(c.kernel_name,
                                               device_prop,
                                               c.transforms_per_block,
                                               length,
                                               compute_scheme,
                                               precision,
                                               factorization,
                                               c.wgs,
                                               c.tpt,
                                               c.half_lds,
                                               c.direct_to_from_reg);
                    c.code = compile_subprocess(src, device_prop.gcnArchName, rtc_helper);
                }
                catch(std::exception& e)
                {
                    c.error = e.what();
                }
            }
        });
    }
    for(auto& t : threads)
        t.join();
}

// A candidate that ran and gave the right answer.  The search keeps
// every one of these so the winner can be reported alongside the
// runners-up - a single best config says nothing about how much of
// the space was close to it.
struct config_result
{
    float                     time;
    unsigned int              wgs;
    unsigned int              tpt;
    unsigned int              tpb;
    bool                      half_lds;
    bool                      direct_to_from_reg;
    std::vector<unsigned int> factorization;
};

// The config as kernel-generator.py wants to see it.  Naming the arch
// is what makes the generator use this wgs and threads_per_transform
// as-is, so the kernel the library runs is the one measured here.
std::string config_line(const config_result& r, unsigned int length)
{
    std::ostringstream os;
    os << "NS(length= " << length << ", workgroup_size= " << r.wgs
       << ", threads_per_transform=" << r.tpt << ", factors=(";
    bool first_factor = true;
    for(auto f : r.factorization)
    {
        if(!first_factor)
            os << ", ";
        first_factor = false;
        os << f;
    }
    os << ")";
    if(!r.half_lds)
        os << ", half_lds=False";
    if(!r.direct_to_from_reg)
        os << ", direct_to_from_reg=False";
    os << ", gcn_arch_name=" << config_arch_name(get_curr_gcn_arch_name()) << ")";
    return os.str();
}

int main(int argc, char** argv)
{
    unsigned int  length         = 0;
    unsigned int  ntrial         = 0;
    unsigned int  nbatch         = 1;
    unsigned int  min_tpb        = 1;
    unsigned int  max_tpb        = 32;
    unsigned int  first_ordering = 1;
    unsigned int  last_ordering  = 0;
    unsigned int  ntop           = 20;
    unsigned int  njob           = rocfft_concurrency();
    std::string   rtc_helper_opt;
    ComputeScheme compute_scheme = CS_KERNEL_STOCKHAM;

    rocfft_precision                        precision = rocfft_precision_single;
    std::map<std::string, rocfft_precision> precision_map{
        {"single", rocfft_precision::rocfft_precision_single},
        {"double", rocfft_precision::rocfft_precision_double},
        {"half", rocfft_precision::rocfft_precision_half}};

    CLI::App app{"rocfft kernel config search"};

    auto brute_force = app.add_subcommand(
        "brute-force", "brute force tuning kernel config with build-in combinations");

    brute_force->add_option("-l, --length", length, "Select a 1D FFT problem size")->default_val(8);
    brute_force->add_option("-N, --ntrial", ntrial, "Trial size for tuning the problem")
        ->default_val(10);
    brute_force->add_option("-b, --batchSize", nbatch, "Batch size of FFT")->default_val(1);
    brute_force
        ->add_option("--max-tpb",
                     max_tpb,
                     "Largest number of transforms per block to consider.  Search time grows "
                     "roughly linearly with this.  Lengths below 32 may need a larger value to "
                     "reach the configurations they ship with.")
        ->default_val(32);
    brute_force
        ->add_option("--min-tpb",
                     min_tpb,
                     "Smallest number of transforms per block to consider, to skip a range "
                     "already known to be uninteresting.")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    brute_force
        ->add_option("--first-ordering",
                     first_ordering,
                     "1-based factor ordering to start at.  Orderings are enumerated "
                     "deterministically for a given length, so this resumes an interrupted "
                     "search or isolates a single ordering.")
        ->default_val(1)
        ->check(CLI::PositiveNumber);
    brute_force
        ->add_option("--last-ordering",
                     last_ordering,
                     "1-based factor ordering to stop after.  Zero runs to the end.")
        ->default_val(0);
    brute_force
        ->add_option("--top",
                     ntop,
                     "How many of the fastest configurations to report.  The winner alone "
                     "doesn't show whether it was a clear win or one of many ties.")
        ->default_val(20);
    brute_force
        ->add_option("-j, --jobs",
                     njob,
                     "Candidates to generate and compile at once.  Compiling dominates the "
                     "search, and hipRTC serializes it within a process, so each one runs in "
                     "its own process to actually overlap.")
        ->default_val(rocfft_concurrency())
        ->check(CLI::PositiveNumber);
    brute_force->add_option("--rtc-helper",
                            rtc_helper_opt,
                            "Path to rocfft_rtc_helper, which compiles one candidate per "
                            "invocation.  Defaults to the copy next to this executable.");

    auto manual_tuning = app.add_subcommand("manual", "manual tuning kernel config");

    std::string               kernel_type;
    std::vector<unsigned int> factorization;
    unsigned int              wgs;
    unsigned int              tpt;
    bool                      half_lds           = true;
    bool                      direct_to_from_reg = true;

    manual_tuning
        ->add_option("--kernel-type", kernel_type, "The valid types are: sbrr/sbcc/sbrc/sbcr")
        ->default_val("sbrr");
    manual_tuning->add_option("-l, --length", length, "Select a 1D FFT problem size")
        ->default_val(8);
    manual_tuning->add_option("-N, --ntrial", ntrial, "Trial size for tuning the problem")
        ->default_val(10);
    manual_tuning
        ->add_option(
            "--precision", precision, "Transform precision: single (default), double, half")
        ->transform(CLI::CheckedTransformer(precision_map, CLI::ignore_case));
    manual_tuning->add_option("-b, --batchSize", nbatch, "Batch size of FFT")->default_val(1);
    manual_tuning->add_option(
        "-f, --factorization", factorization, "Factorization for a given FFT problem");
    manual_tuning->add_option("-w, --wgs", wgs, "Work group size")->default_val(64);
    manual_tuning->add_option("--tpt", tpt, "Thread per transform")->default_val(1);
    manual_tuning->add_option("--half-lds", half_lds, "Use half LDS or not")->default_val(true);
    manual_tuning->add_option("--direct-reg", direct_to_from_reg, "Direct load to/from reg")
        ->default_val(true);

    app.require_subcommand(0, 1);

    // Parse args and catch any errors here
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(hipInit(0) != hipSuccess)
        throw std::runtime_error("hipInit failure");

    hipDeviceProp_t device_prop;
    // Todo: support device id
    if(hipGetDeviceProperties(&device_prop, 0) != hipSuccess)
        throw std::runtime_error("hipGetDeviceProperties failure");

    if(brute_force->parsed())
    {
        // init device data
        device_data_t data;
        data.batch = nbatch;
        // construct random input on host side, allocate input/output
        // buffers on GPU.  input will be copied to GPU at launch time
        data.host_input_buf = create_input_buf(length, data.batch);
        data.input_buf      = create_device_buf(length, data.batch);
        data.output_buf     = create_device_buf(length, data.batch);
        // twiddles depend on the factorization, so they're created
        // per-factorization below
        data.lengths    = create_lengths(length);
        data.stride_in  = create_strides(length);
        data.stride_out = create_strides(length);
        std::cout << "length " << length << ", batch " << data.batch << std::endl;

        // Reference for checking that a candidate kernel actually
        // computes the transform.  Two blocks' worth of transforms is
        // enough to catch indexing mistakes that only corrupt some of
        // the transforms in a block.
        const size_t ncheck = std::min<size_t>(data.batch, 2 * max_tpb);
        std::cout << "computing reference DFT for " << ncheck << " transforms" << std::endl;
        const auto                         reference = reference_dft(data.host_input_buf, length, ncheck);
        std::vector<rocfft_complex<float>> check_buf(reference.size());

        const auto factorizations = factorize(length);

        // every candidate that gave the right answer, ranked at the end
        std::vector<config_result> results;

        // aim for occupancy-2, matching the limit handed to the generator
        const unsigned int lds_byte_limit = device_prop.sharedMemPerBlock / 2;
        const unsigned int wgs_limit
            = std::min<unsigned int>(max_wgs, device_prop.maxThreadsPerBlock);
        // a block narrower than one wave leaves lanes permanently idle
        const unsigned int wgs_floor = device_prop.warpSize;

        // the search space depends on the device, so record what it
        // was bounded by alongside the results
        std::cout << "device " << device_prop.gcnArchName << ", wave " << device_prop.warpSize
                  << ", lds " << device_prop.sharedMemPerBlock / 1024 << "KiB, lds budget "
                  << lds_byte_limit / 1024 << "KiB, workgroup " << wgs_floor << "-" << wgs_limit
                  << ", transforms_per_block " << min_tpb << "-" << max_tpb << std::endl;

        // A twiddle table can only describe this many passes, so
        // rocFFT could never use a longer factorization anyway.
        // Orderings of the factors are separate candidates, because
        // the twiddle layout follows the order of the radices.
        std::vector<std::vector<unsigned int>> orderings;
        for(auto factorization : factorizations)
        {
            if(factorization.size() > TWIDDLES_RTC_MAX_RADICES)
                continue;
            do
            {
                orderings.push_back(factorization);
            } while(std::next_permutation(factorization.begin(), factorization.end()));
        }

        if(min_tpb > max_tpb)
        {
            std::cerr << "--min-tpb (" << min_tpb << ") must not exceed --max-tpb (" << max_tpb
                      << ")" << std::endl;
            return 1;
        }

        // Orderings come out of factorize() and next_permutation() in a
        // fixed order for a given length, so a 1-based index names the
        // same ordering across runs and can be used to resume or isolate.
        const size_t ordering_begin = std::min<size_t>(first_ordering - 1, orderings.size());
        const size_t ordering_end   = last_ordering
                                          ? std::min<size_t>(last_ordering, orderings.size())
                                          : orderings.size();

        // Counting the whole space first costs nothing and keeps a
        // long length from being a surprise - the search is dominated
        // by one runtime compile per candidate.
        size_t ncandidate = 0;
        for(size_t i = ordering_begin; i < ordering_end; ++i)
            ncandidate += enumerate_candidates(length,
                                               orderings[i],
                                               min_tpb,
                                               max_tpb,
                                               wgs_floor,
                                               wgs_limit,
                                               lds_byte_limit)
                              .size();
        auto rtc_helper = rtc_helper_opt.empty() ? locate_rtc_helper(argv[0])
                                                 : std::optional<std::string>(rtc_helper_opt);
        if(!rtc_helper)
            throw std::runtime_error("unable to find rocfft_rtc_helper next to this executable, "
                                     "pass --rtc-helper to say where it is");

        std::cout << ordering_end - ordering_begin << " factor orderings (" << ordering_begin + 1
                  << " through " << ordering_end << " of " << orderings.size() << "), "
                  << ncandidate << " candidate kernels, compiling " << njob << " at a time via "
                  << *rtc_helper << std::endl;

        size_t      tested    = 0;
        size_t      rejected  = 0;
        const auto  start     = std::chrono::steady_clock::now();
        for(size_t ordering_idx = ordering_begin; ordering_idx < ordering_end; ++ordering_idx)
        {
            const auto& factorization = orderings[ordering_idx];

            // Announce the ordering before doing any work for it. A GPU
            // reset takes down the process, so only output already flushed
            // survives to name the ordering responsible.
            std::cout << "ordering " << ordering_idx + 1 << " of " << orderings.size()
                      << " entering, factors";
            for(auto f : factorization)
                std::cout << " " << f;
            std::cout << std::endl;

            auto candidates = enumerate_candidates(
                length, factorization, min_tpb, max_tpb, wgs_floor, wgs_limit, lds_byte_limit);
            if(candidates.empty())
                continue;

            // the twiddle layout follows the order of the radices
            data.twiddles = create_twiddles(factorization, precision, device_prop);

            std::cout << "twiddles ready, compiling " << candidates.size() << " candidates"
                      << std::endl;

            compile_candidates(candidates,
                               device_prop,
                               length,
                               compute_scheme,
                               precision,
                               factorization,
                               njob,
                               rtc_helper);

            for(const auto& c : candidates)
            {
                ++tested;
                if(!c.error.empty())
                {
                    ++rejected;
                    std::cout << length << ", " << c.kernel_name << ", UNAVAILABLE, " << c.error
                              << std::endl;
                    continue;
                }

                // Name the candidate before loading and launching it, for the
                // same reason as the ordering above.
                std::cout << "launch " << c.kernel_name << std::endl;

                hipModule_wrapper_t module;
                module.alloc(c.code.data());
                std::promise<hipModule_wrapper_t>       module_promise;
                std::shared_future<hipModule_wrapper_t> module_future
                    = module_promise.get_future();
                module_promise.set_value(std::move(module));

                RTCKernelStockham kernel(c.kernel_name, KIntType::U32, module_future);

                // launch the whole workgroup, which is what the
                // library does - it uses the generator's
                // workgroup_size directly
                float time = launch_kernel(
                    kernel,
                    DivRoundingUp<unsigned int>(data.batch, c.transforms_per_block),
                    c.wgs,
                    get_lds_bytes(length, c.transforms_per_block, c.half_lds),
                    ntrial,
                    device_prop,
                    data);

                // a wrong answer makes the time meaningless, so check
                // the output before considering this config
                if(hipMemcpy(check_buf.data(),
                             data.output_buf.data(),
                             check_buf.size() * sizeof(rocfft_complex<float>),
                             hipMemcpyDeviceToHost)
                   != hipSuccess)
                    throw std::runtime_error("failed to hipMemcpy");
                auto error = relative_error(check_buf, reference);
                if(!(error < max_rel_error))
                {
                    ++rejected;
                    std::cout << length << ", " << c.kernel_name << ", INCORRECT, relative error "
                              << error << std::endl;
                    continue;
                }

                // print median time for this length in a format that
                // can be easily grepped for and shoved into a
                // database if desired
                std::cout << length << ", " << c.kernel_name << ", " << std::setprecision(3)
                          << static_cast<double>(time) << "ms " << std::endl;

                results.push_back({time,
                                   c.wgs,
                                   c.tpt,
                                   c.transforms_per_block,
                                   c.half_lds,
                                   c.direct_to_from_reg,
                                   factorization});
            }

            // Report against a rate measured here rather than a
            // guess, so the estimate reflects this machine and this
            // length's compile cost.
            const double elapsed
                = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::cout << "ordering " << ordering_idx + 1 << " of " << orderings.size() << ", "
                      << tested << " of " << ncandidate << " candidates, " << std::fixed
                      << std::setprecision(0) << elapsed << "s elapsed, ~"
                      << elapsed * (static_cast<double>(ncandidate) / std::max<size_t>(tested, 1) - 1.0)
                      << "s remaining" << std::defaultfloat << std::endl;
        }

        if(results.empty())
        {
            std::cout << "no correct configuration found for length " << length << ", "
                      << tested << " tested, " << rejected << " incorrect" << std::endl;
            return 1;
        }

        std::sort(results.begin(),
                  results.end(),
                  [](const config_result& a, const config_result& b) { return a.time < b.time; });
        const auto& best = results.front();

        // print a line with the best config, to go into kernel-generator.py
        std::cout << "  " << config_line(best, length) << " best_time," << best.time
                  << " transforms_per_block," << best.tpb << " configs_tested," << tested
                  << " configs_incorrect," << rejected << std::endl;

        // The winner on its own can't distinguish a real optimum from
        // the fastest draw out of many ties, and says nothing about
        // which parameter the time was actually sensitive to.  Report
        // the leading configs as a table that can go straight into a
        // dataframe, with factors joined by '-' to keep it one field
        // per column.
        const size_t nshow = std::min<size_t>(ntop, results.size());
        std::cout << "top " << nshow << " of " << results.size() << " correct configs for length "
                  << length << " on " << device_prop.gcnArchName << std::endl;
        std::cout << "rank,time_ms,rel_to_best,workgroup_size,threads_per_transform,"
                     "transforms_per_block,half_lds,direct_to_from_reg,factors"
                  << std::endl;
        for(size_t i = 0; i < nshow; ++i)
        {
            const auto& r = results[i];
            std::cout << i + 1 << ',' << std::setprecision(4) << static_cast<double>(r.time) << ','
                      << std::setprecision(4) << static_cast<double>(r.time) / best.time << ','
                      << r.wgs << ',' << r.tpt << ',' << r.tpb << ','
                      << (r.half_lds ? "True" : "False") << ','
                      << (r.direct_to_from_reg ? "True" : "False") << ',';
            bool first_factor = true;
            for(auto f : r.factorization)
            {
                if(!first_factor)
                    std::cout << '-';
                first_factor = false;
                std::cout << f;
            }
            std::cout << std::endl;
        }

        // The leading configs tend to be near-duplicates of each
        // other, so the table above can be filled by one value of
        // transforms_per_block and show nothing about the rest.  Also
        // report the best time reached at each one, which is what
        // says whether a wider block helped and how sharp the
        // optimum is.  results is sorted, so the first entry seen for
        // a given tpb is that tpb's best.
        std::map<unsigned int, const config_result*> best_by_tpb;
        for(const auto& r : results)
            best_by_tpb.emplace(r.tpb, &r);
        std::cout << "best per transforms_per_block for length " << length << std::endl;
        std::cout << "transforms_per_block,time_ms,rel_to_best,workgroup_size,"
                     "threads_per_transform,half_lds,direct_to_from_reg"
                  << std::endl;
        for(const auto& [tpb, r] : best_by_tpb)
            std::cout << tpb << ',' << std::setprecision(4) << static_cast<double>(r->time) << ','
                      << std::setprecision(4) << static_cast<double>(r->time) / best.time << ','
                      << r->wgs << ',' << r->tpt << ',' << (r->half_lds ? "True" : "False") << ','
                      << (r->direct_to_from_reg ? "True" : "False") << std::endl;
    }
    else if(manual_tuning->parsed())
    {
        // init device data
        device_data_t data;
        data.batch = nbatch;
        // construct random input on host side, allocate input/output
        // buffers on GPU.  input will be copied to GPU at launch time
        data.host_input_buf = create_input_buf(length, data.batch);
        data.input_buf      = create_device_buf(length, data.batch);
        data.output_buf     = create_device_buf(length, data.batch);
        // Stand-in twiddles table, same length as the FFT.  These are
        // the wrong values, so the transform this measures is not the
        // one rocFFT would compute - close enough for timing, but the
        // output can't be checked against a reference.
        auto host_twiddles = create_input_buf(length, 1);
        data.twiddles      = create_device_buf(length, 1);
        if(hipMemcpy(data.twiddles.data(),
                     host_twiddles.data(),
                     host_twiddles.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");
        data.lengths    = create_lengths(length);
        data.stride_in  = create_strides(length);
        data.stride_out = create_strides(length);

        // Fixme: support other kernel types properly
        if(kernel_type == "sbcc")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
        else if(kernel_type == "sbrc")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;
        else if(kernel_type == "sbcr")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_CR;

        // Todo: verify factorization
        // const auto factorizations = factorize(length);

        // Todo: verify tpt
        // auto tpts = supported_threads_per_transform(factorization);
        // tpt == wgs is legitimate: it means one transform per block.
        if(tpt <= wgs)
        {
            // Fixme:
            // half lds currently requires direct to/from reg

            auto kernel_name
                = test_kernel_name(length, factorization, wgs, tpt, half_lds, direct_to_from_reg);
            unsigned int transforms_per_block = 0;

            auto kernel_src = test_kernel_src(kernel_name,
                                              device_prop,
                                              transforms_per_block,
                                              length,
                                              compute_scheme,
                                              precision,
                                              factorization,
                                              wgs,
                                              tpt,
                                              half_lds,
                                              direct_to_from_reg);

            auto                code = compile_inprocess(kernel_src, device_prop.gcnArchName);
            hipModule_wrapper_t module;
            module.alloc(code.data());
            std::promise<hipModule_wrapper_t>       module_promise;
            std::shared_future<hipModule_wrapper_t> module_future = module_promise.get_future();
            module_promise.set_value(std::move(module));
            RTCKernelStockham kernel(kernel_name, KIntType::U32, module_future);

            float time
                = launch_kernel(kernel,
                                DivRoundingUp<unsigned int>(data.batch, transforms_per_block),
                                tpt * transforms_per_block,
                                get_lds_bytes(length, transforms_per_block, half_lds),
                                ntrial,
                                device_prop,
                                data);

            // print median time for this length in a format that can be easily
            // grepped for and shoved into a database if desired
            std::cout << kernel_name << ", " << std::setprecision(3) << static_cast<double>(time)
                      << std::endl;
        }
        else
        {
            std::cerr << "threads_per_transform (" << tpt << ") must not exceed workgroup_size ("
                      << wgs << ")" << std::endl;
            return 1;
        }
    }

    return 0;
}
