import testers.tester_impls.test_conv
import testers.inference_sdks.tflite
import testers.sampling.conv_sampler


def main():
    tester = testers.tester_impls.test_conv.TestConv(
        adb_device_id="8A9Y0G80H",
        inference_sdk=testers.inference_sdks.tflite.Tflite(
            "/data/local/tmp/master-20191015/log_benchmark_model", None),
        sampler=testers.sampling.conv_sampler.ConvSampler())
    tester.run(settings={
        "filter": lambda sample: sample[-1] == 1,
        "push_to_max_freq": True,
        "mkshrc": "/data/local/tmp/mkshrc"
    }, benchmark_model_flags={
        "num_runs": 30
    })


if __name__ == '__main__':
    main()
