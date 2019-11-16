import testers.tester_impls.test_fc
import testers.inference_sdks.hiai
import testers.sampling.fc_sampler


def main():
    tester = testers.tester_impls.test_fc.TestFc(
        adb_device_id="AQH7N17B14007975",
        inference_sdk=testers.inference_sdks.hiai.Hiai(),
        sampler=testers.sampling.fc_sampler.FcSampler())
    tester.run({}, {})

if __name__ == '__main__':
    main()