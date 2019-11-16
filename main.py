import testers.tester
import testers.inference_sdks.hiai
import testers.sampling.dwconv_sampler


testers.tester.Tester()

testers.inference_sdks.hiai.Hiai()

print(list(testers.sampling.dwconv_sampler._get_dwconv_profiles()))