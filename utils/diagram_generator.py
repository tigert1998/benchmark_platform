import csv
import os

import matplotlib.pyplot as plt


class DiagramGenerator:
    CIN = "current_cin"
    COUT = "current_cout"
    METRIC_START = "latency_ms"

    def __init__(self, filename_filter=lambda filename: True):
        self.filename_filter = filename_filter

    @staticmethod
    def _concatenate_diagram_title(xs, ys, space):
        ls = []
        for x, y in zip(xs, ys):
            if isinstance(y, str) and y.strip() == "":
                continue
            ls.append("{}{}{}".format(str(x), " = " if space else "=", str(y)))
        if space:
            return ", ".join(ls)
        else:
            return "_".join(ls)

    def _generate_cin_eq_cout(self, dic, sample_titles, metric_titles, output_folder):
        new_dic = {}
        for item in filter(lambda item: item[0][0] == item[0][1], dic.items()):
            key = item[0][2:]
            if new_dic.get(key) is None:
                new_dic[key] = {}
            new_dic[key][item[0][0]] = item[1]

        for sample, channel_to_metrics in new_dic.items():
            channel_to_metrics = sorted(list(channel_to_metrics.items()))
            xs = list(map(lambda item: item[0], channel_to_metrics))

            for i in range(len(metric_titles)):
                metric_title = metric_titles[i]
                ys = list(map(lambda item: item[1][i], channel_to_metrics))

                diagram_title = "{}_cin=cout_{}".format(
                    metric_title,
                    self._concatenate_diagram_title(sample_titles, sample, False))
                if not self.filename_filter(diagram_title):
                    continue

                fig = plt.figure()
                fig.set_size_inches(19, 11)
                plt.plot(xs, ys)
                plt.xlabel("#channels")
                plt.ylabel(metric_title)
                plt.title("{}, cin = cout, {}".format(
                    metric_title,
                    self._concatenate_diagram_title(sample_titles, sample, True))
                )
                plt.savefig(
                    "{}/{}.png".format(
                        output_folder,
                        diagram_title)
                )
                plt.close()

    def generate(self, csv_filepath, output_folder):
        assert os.path.isdir(output_folder)

        plt.ioff()

        with open(csv_filepath, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                break
            metric_titles = row[row.index(self.METRIC_START):]

            sample_titles = row[:row.index(self.METRIC_START)]
            sample_titles = list(
                filter(lambda title: title not in [self.CIN, self.COUT], sample_titles))

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

        self._generate_cin_eq_cout(
            dic, sample_titles, metric_titles, output_folder)
