def append_layerwise_info(data, layerwise_info):
    if layerwise_info is not None:
        for value in layerwise_info.values():
            data[value["name"] + "_avg_ms"] = value["time"]["avg_ms"]
            data[value["name"] + "_std_ms"] = value["time"]["std_ms"]
    return data
