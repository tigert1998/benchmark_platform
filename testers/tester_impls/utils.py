def append_layerwise_info(data, layerwise_info):
    if layerwise_info is not None:
        for dic in layerwise_info:
            data[dic["name"] + "_avg_ms"] = dic["time"]["avg_ms"]
            data[dic["name"] + "_std_ms"] = dic["time"]["std_ms"]
    return data
