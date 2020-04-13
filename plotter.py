# import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot(df, x):
        Plotter.plot_design(time="before")
        ax = sns.lineplot(x=x, y="log_loss", hue="method", data=df)
        Plotter.plot_design(time="after", ax=ax)
        ax.set(ylabel="log(loss)")
        plt.tight_layout()
        fig = ax.get_figure()
        fig.set_size_inches(9, 7)
        plt.show()
        # return ax, fig

    @staticmethod
    def interpolate_between_steps(vals: list):
        def find_intervals(vals: list):
            last_val = -1E10
            start_idx = 0
            intervals = []
            for idx, val in enumerate(vals):
                if val != last_val:
                    if idx - start_idx > 1:
                        # add interval
                        intervals.append([start_idx, idx])
                    start_idx = idx
                last_val = val
            return intervals

        intervals = find_intervals(vals)
        new_vals = vals.copy()
        for (start, end) in intervals:
            length = end - start
            for dist, i in enumerate(range(start, end)):
                new_vals[i] -= (vals[start] - vals[end]) * float(dist) / float(length)
        return new_vals

    @staticmethod
    def plot_design(time="before", ax=None):
        def set_style():
            plt.clf()
            sns.set_context("paper")
            sns.set(font='serif', font_scale=1.3)
            sns.set_style("white", {
                "font.family": "serif",
                "font.serif": ["Times", "Palatino", "serif"]
            })

        def set_size(ax):
            fig = ax.get_figure()
            fig.set_size_inches(7, 5)

        def fix_lims(ax):
            ax.set_xlim(ax.get_xlim())
            ax.set_ylim(ax.get_ylim())


        def add_fom_plot(ax, x_thr):
            fix_lims(ax)
            ax.axvline(x=x_thr, linestyle="-", color="k", linewidth=0.5)


        def adjust_axes(ax):
            ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 4), useMathText=True)

        if time == "before":
            set_style()
        elif time == "after":
            adjust_axes(ax)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles[1:], labels=labels[1:])
            Plotter.adjust_lines(ax, labels, handles)

    @staticmethod
    def adjust_lines(ax, labels, handles):
        for line, label in zip(ax.get_lines()[:len(labels) - 1], labels[1:]):
            if "TR" in label:
                plt.setp(line, zorder=100)
            else:
                line.set_linestyle("--")

        for line in ax.legend(handles=handles[1:], labels=labels[1:]).get_lines():
            if not "TR" in line.get_label():
                line.set_linestyle("--")
