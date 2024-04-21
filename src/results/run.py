from src.results.cira_report import generate_cira_report
from src.results.distribution_plot import generate_distribution_plot
from src.results.distribution_report import generate_distribution_report
from src.results.graph_characteristics_report import generate_graphs_report
from src.results.precision_recall_plot import generate_precision_recall_plots
from src.results.validation_report import generate_validation_report


def run():
    generate_distribution_plot()
    generate_distribution_report()
    generate_graphs_report()
    generate_precision_recall_plots()
    generate_validation_report()
    generate_cira_report()


if __name__ == "__main__":
    run()
