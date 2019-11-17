import testers.tester_impls.test_conv
import testers.inference_sdks.rknn
import testers.sampling.conv_sampler


def main():
    tester = testers.tester_impls.test_conv.TestConv(
        adb_device_id="TD033101190100171",
        inference_sdk=testers.inference_sdks.rknn.Rknn(),
        sampler=testers.sampling.conv_sampler.ConvSampler())
    tester.run(settings={
        "filter": lambda sample: sample[-1] == 3
    }, benchmark_model_flags={
        "num_runs": 30
    })


if __name__ == '__main__':
    main()
