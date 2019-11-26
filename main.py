import testers.tester_impls.test_conv_gpu_mem_comp_split
import testers.inference_sdks.tflite_gpu_mem_comp_split
import testers.sampling.conv_sampler


def main():
    tester = testers.tester_impls.test_conv_gpu_mem_comp_split.TestConvGpuMemCompSplit(
        adb_device_id="8A9Y0G80H",
        inference_sdk=testers.inference_sdks.tflite_gpu_mem_comp_split.TfliteGpuMemCompSplit({
            "benchmark_model_path": "/data/local/tmp/master-20191015/split_841_benchmark_model",
        }),
        sampler=testers.sampling.conv_sampler.ConvSampler({
            "filter": lambda sample: sample[2: 5] == [56, 88, 81] and sample[-1] == 1
        }))

    tester.run(settings={}, benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": True,
        "gpu_precision_loss_allowed": False
    })


if __name__ == '__main__':
    main()
