import csv
import os

import matplotlib
matplotlib.use('Agg')  # noqa: E402
import matplotlib.pyplot as plt

from .utils import set_multilevel_dict


class DiagramGenerator:
    CIN = "current_cin"
    COUT = "current_cout"
    METRIC_START = "latency_ms"
    IRRELEVANTS = ["model", "original_cin", "original_cout"]

    def __init__(self, output_folder, metric_lists, point_threshold=10):
        self.output_folder = output_folder
        self.metric_lists = metric_lists
        self.point_threshold = point_threshold
        assert os.path.isdir(output_folder)

    @staticmethod
    def _concatenate_diagram_title(xs, ys, space):
        if space:
            eq = " = "
            sep = ", "
        else:
            eq = "="
            sep = "_"
        ls = []
        for x, y in zip(xs, ys):
            if isinstance(y, str) and y.strip() == "":
                continue
            ls.append("{}{}{}".format(str(x), eq, str(y)))
        return sep.join(ls)

    def _plot_figure(self, xs, ys_list, xlabel, ylabel_list, title, filename):
        fig = plt.figure()
        fig.set_size_inches(16, 9)
        for ys, ylabel in zip(ys_list, ylabel_list):
            plt.plot(xs, ys, 'bo', label=ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        plt.savefig("{}/{}.png".format(
            self.output_folder,
            filename))
        plt.close()

    def _generate_with_fixed_cin_or_cout(self, dic, sample_titles, metric_titles):
        fixed_cin_dic = {}
        fixed_cout_dic = {}
        for item in dic.items():
            key = item[0][2:]
            cin, cout = item[0][:2]
            metrics = item[1]

            set_multilevel_dict(
                fixed_cin_dic,
                keys=[key, cin, cout], value=metrics)
            set_multilevel_dict(
                fixed_cout_dic,
                keys=[key, cout, cin], value=metrics)

        for fixed_name, fixed_dic in zip(["cin", "cout"], [fixed_cin_dic, fixed_cout_dic]):
            for sample, inner_dic in fixed_dic.items():
                for fixed_channel in inner_dic:
                    channel_to_metrics = sorted(
                        list(inner_dic[fixed_channel].items()))
                    if len(channel_to_metrics) <= self.point_threshold:
                        continue
                    xs = list(map(lambda item: item[0], channel_to_metrics))
                    for metric_list in self.metric_lists:
                        diagram_filename = "{}_{}={}_{}".format(
                            "_".join(metric_list), fixed_name, fixed_channel,
                            self._concatenate_diagram_title(sample_titles, sample, False))
                        diagram_title = "{}, {} = {}, {}".format(
                            ", ".join(metric_list), fixed_name, fixed_channel,
                            self._concatenate_diagram_title(sample_titles, sample, True))

                        ys_list = []
                        for metric in metric_list:
                            i = metric_titles.index(metric)
                            ys_list.append(list(map(
                                lambda item: item[1][i], channel_to_metrics)))

                        xlabel = "#output_channels" if fixed_name == "cin" else "#input_channels"

                        self._plot_figure(
                            xs, ys_list,
                            xlabel, metric_list,
                            diagram_title, diagram_filename
                        )

    def _generate_cin_eq_cout(self, dic, sample_titles, metric_titles):
        new_dic = {}
        for item in filter(lambda item: item[0][0] == item[0][1], dic.items()):
            key = item[0][2:]
            set_multilevel_dict(new_dic, keys=[key, item[0][0]], value=item[1])

        for sample, channel_to_metrics in new_dic.items():
            channel_to_metrics = sorted(list(channel_to_metrics.items()))
            xs = list(map(lambda item: item[0], channel_to_metrics))
            if len(xs) <= self.point_threshold:
                continue

            for metric_list in self.metric_lists:
                diagram_filename = "{}_cin=cout_{}".format(
                    "_".join(metric_list),
                    self._concatenate_diagram_title(sample_titles, sample, False))
                diagram_title = "{}, cin = cout, {}".format(
                    ", ".join(metric_list),
                    self._concatenate_diagram_title(sample_titles, sample, True))

                ys_list = []
                for metric in metric_list:
                    i = metric_titles.index(metric)
                    ys_list.append(
                        list(map(lambda item: item[1][i], channel_to_metrics)))

                self._plot_figure(
                    xs, ys_list,
                    "#channels", metric_list,
                    diagram_title, diagram_filename
                )

    def generate(self, csv_filepath):
        with open(csv_filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                break
            metric_titles = row[row.index(self.METRIC_START):]

            sample_titles = []
            for i in range(row.index(self.METRIC_START)):
                if row[i] in ([self.CIN, self.COUT] + self.IRRELEVANTS):
                    continue
                sample_titles.append(row[i])

        dic = {}

        with open(csv_filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cin = int(row[self.CIN])
                cout = int(row[self.COUT])

                samples = []
                for sample_title in sample_titles:
                    tmp = row[sample_title]
                    try:
                        samples.append(int(tmp))
                    except:
                        samples.append(tmp)

                metrics = []
                for metric_title in metric_titles:
                    tmp = row[metric_title]
                    try:
                        metrics.append(float(tmp))
                    except:
                        metrics.append(tmp)

                dic[(cin, cout, *samples)] = metrics

        self._generate_cin_eq_cout(dic, sample_titles, metric_titles)
        self._generate_with_fixed_cin_or_cout(
            dic, sample_titles, metric_titles)
