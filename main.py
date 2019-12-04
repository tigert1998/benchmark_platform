import testers.tester_impls.test_conv_gpu_mem_comp_split
import testers.inference_sdks.tflite_gpu_mem_comp_split
import testers.sampling.conv_sampler


def main():
    tester = testers.tester_impls.test_conv_gpu_mem_comp_split.TestConvGpuMemCompSplit(
        adb_device_id="5e6fecf",
        inference_sdk=testers.inference_sdks.tflite_gpu_mem_comp_split.TfliteGpuMemCompSplit({
            "benchmark_model_path": "/data/local/tmp/tf-r2.1-60afa4e/benchmark_model_split_io_comp",
            "su": True
        }),
        sampler=testers.sampling.conv_sampler.ConvSampler({
            "filter": lambda sample: sample[-1] == 1
        }))

    tester.run(settings={}, benchmark_model_flags={
        "num_runs": 30,
        "use_gpu": True,
        "precision": "F32_F16",
        "work_group_size": "4,4,1"
    })


if __name__ == '__main__':
    main()
