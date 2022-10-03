#!/usr/bin/env python3

# Copyright 2020 Department of Computational Biology for Infection Research - Helmholtz Centre for Infection Research
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import errno
import logging
import os
import sys

import matplotlib

from src import amber_html
from src import plot_by_genome
from src import plots
from version import __version__

matplotlib.use("Agg")
import pandas as pd
from src.utils import load_data
from src.utils import argparse_parents
from src.utils import labels as utils_labels
from src import binning_classes
from src.utils import load_ncbi_taxinfo


def get_logger(output_dir, silent):
    make_sure_path_exists(output_dir)
    logger = logging.getLogger("amber")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    logging_fh = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)

    if not silent:
        logging_stdout = logging.StreamHandler(sys.stdout)
        logging_stdout.setFormatter(formatter)
        logger.addHandler(logging_stdout)
    return logger


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def create_output_directories(output_dir, sample_id_to_queries_list):
    logging.getLogger("amber").info("Creating output directories")
    for sample_id in sample_id_to_queries_list:
        for query in sample_id_to_queries_list[sample_id]:
            make_sure_path_exists(os.path.join(output_dir, query.binning_type, query.label))


def get_labels(labels, bin_files):
    if labels:
        labels_list = [x.strip() for x in labels.split(",")]
        if len(set(labels_list)) != len(bin_files):
            logging.getLogger("amber").critical(
                "Number of different labels does not match the number of binning files. Please check parameter -l, --labels."
            )
            exit(1)
        return labels_list
    tool_id = []
    for bin_file in bin_files:
        tool_id.append(bin_file.split("/")[-1].split(".binning")[0])
    return tool_id


def plot_genome_binning(
    color_indices, sample_id_to_queries_list, df_summary, pd_bins, labels, coverages_pd, output_dir
):
    df_summary_g = df_summary[df_summary[utils_labels.BINNING_TYPE] == "genome"]
    if len(df_summary_g) == 0:
        return

    logging.getLogger("amber").info("Creating genome binning plots")

    for sample_id in sample_id_to_queries_list:
        for query in sample_id_to_queries_list[sample_id]:
            if isinstance(query, binning_classes.GenomeQuery):
                query.plot()

    available_tools = list(df_summary_g[utils_labels.TOOL].unique())
    available_tools = [tool for tool in labels if tool in available_tools]

    if not coverages_pd.empty:
        plots.plot_precision_recall_by_coverage(
            sample_id_to_queries_list, pd_bins, coverages_pd, available_tools, output_dir
        )

    if color_indices:
        color_indices = [int(i) - 1 for i in color_indices.split(",")]
    plots.create_legend(color_indices, available_tools, output_dir)
    plots.plot_avg_precision_recall(color_indices, df_summary_g, labels, output_dir)
    plots.plot_precision_recall(color_indices, df_summary_g, labels, output_dir)
    plots.plot_adjusted_rand_index_vs_assigned_bps(color_indices, df_summary_g, labels, output_dir)

    plots.plot_boxplot(sample_id_to_queries_list, "recall_bp", output_dir, available_tools)
    plots.plot_boxplot(sample_id_to_queries_list, "precision_bp", output_dir, available_tools)

    plots.plot_counts(pd_bins, available_tools, output_dir, "bin_counts", plots.get_number_of_hq_bins)
    plots.plot_counts(pd_bins, available_tools, output_dir, "high_scoring_bins", plots.get_number_of_hq_bins_by_score)

    plot_by_genome.plot_precision_recall_per_bin(pd_bins, output_dir)

    plots.plot_contamination(
        pd_bins,
        "genome",
        "Contamination",
        "Index of bin (sorted by contamination (bp))",
        "Contamination (bp)",
        plots.create_contamination_column,
        output_dir,
    )
    plots.plot_contamination(
        pd_bins,
        "genome",
        "Completeness - contamination",
        "Index of bin (sorted by completeness - contamination (bp))",
        "Completeness - contamination (bp)",
        plots.create_completeness_minus_contamination_column,
        output_dir,
    )


def plot_taxonomic_binning(color_indices, df_summary, pd_bins, labels, output_dir):
    df_summary_t = df_summary[df_summary[utils_labels.BINNING_TYPE] == "taxonomic"]
    if len(df_summary_t) == 0:
        return

    logging.getLogger("amber").info("Creating taxonomic binning plots")

    available_tools = list(df_summary_t[utils_labels.TOOL].unique())
    available_tools = [tool for tool in labels if tool in available_tools]

    if color_indices:
        color_indices = [int(i) - 1 for i in color_indices.split(",")]
    for rank, pd_group in df_summary_t.groupby("rank"):
        plots.plot_avg_precision_recall(color_indices, pd_group, available_tools, output_dir, rank)
        plots.plot_precision_recall(color_indices, pd_group, available_tools, output_dir, rank)
        plots.plot_adjusted_rand_index_vs_assigned_bps(color_indices, pd_group, available_tools, output_dir, rank)

    metrics_list = [utils_labels.AVG_PRECISION_BP, utils_labels.AVG_RECALL_BP]
    errors_list = [utils_labels.AVG_PRECISION_BP_SEM, utils_labels.AVG_RECALL_BP_SEM]
    plots.plot_taxonomic_results(df_summary_t, metrics_list, errors_list, "avg_precision_recall_bp", output_dir)

    metrics_list = [utils_labels.AVG_PRECISION_SEQ, utils_labels.AVG_RECALL_SEQ]
    errors_list = [utils_labels.AVG_PRECISION_SEQ_SEM, utils_labels.AVG_RECALL_SEQ_SEM]
    plots.plot_taxonomic_results(df_summary_t, metrics_list, errors_list, "avg_precision_recall_seq", output_dir)

    metrics_list = [
        utils_labels.PRECISION_PER_BP,
        utils_labels.RECALL_PER_BP,
        utils_labels.PRECISION_PER_SEQ,
        utils_labels.RECALL_PER_SEQ,
    ]
    plots.plot_taxonomic_results(df_summary_t, metrics_list, [], "precision_recall", output_dir)

    for rank in load_ncbi_taxinfo.RANKS:
        pd_bins_rank = pd_bins[pd_bins["rank"] == rank]
        plots.plot_contamination(
            pd_bins_rank,
            "taxonomic",
            rank + " | Contamination",
            "Index of bin (sorted by contamination (bp))",
            "Contamination (bp)",
            plots.create_contamination_column,
            output_dir,
        )
        plots.plot_contamination(
            pd_bins_rank,
            "taxonomic",
            rank + " | Completeness - contamination",
            "Index of bin (sorted by completeness - contamination (bp))",
            "Completeness - contamination (bp)",
            plots.create_completeness_minus_contamination_column,
            output_dir,
        )


def evaluate(queries_list, sample_id):
    breakpoint()
    pd_bins_all = pd.DataFrame()
    df_summary = pd.DataFrame()

    for query in queries_list:
        if not query.compute_metrics():
            continue

        query_metrics_df = query.get_metrics_df()
        query_metrics_df[utils_labels.SAMPLE] = sample_id

        df_summary = pd.concat([df_summary, query_metrics_df], ignore_index=True, sort=True)

        query.precision_df[utils_labels.TOOL] = query.label
        pd_bins_all = pd.concat([pd_bins_all, query.precision_df.reset_index()], ignore_index=True, sort=True)

    pd_bins_all["sample_id"] = sample_id

    return df_summary, pd_bins_all


def evaluate_samples_queries(sample_id_to_queries_list):
    pd_bins_all = pd.DataFrame()
    df_summary_all = pd.DataFrame()

    for sample_id in sample_id_to_queries_list:
        df_summary, pd_bins = evaluate(sample_id_to_queries_list[sample_id], sample_id)
        pd_bins_all = pd.concat([pd_bins_all, pd_bins], ignore_index=True)
        df_summary_all = pd.concat([df_summary_all, df_summary], ignore_index=True)

    # Gold standard only has unfiltered metrics, so copy values to unfiltered columns
    for col in df_summary_all.columns:
        if col.endswith(utils_labels.UNFILTERED):
            df_summary_all.loc[df_summary_all[utils_labels.TOOL] == utils_labels.GS, col] = df_summary_all.loc[
                df_summary_all[utils_labels.TOOL] == utils_labels.GS, col[: -len(utils_labels.UNFILTERED)]
            ]

    return df_summary_all, pd_bins_all


def save_metrics(sample_id_to_queries_list, df_summary, pd_bins, output_dir, stdout):
    logging.getLogger("amber").info("Saving computed metrics")
    df_summary.to_csv(os.path.join(output_dir, "results.tsv"), sep="\t", index=False)
    pd_bins.to_csv(os.path.join(output_dir, "bin_metrics.tsv"), index=False, sep="\t")
    if stdout:
        summary_columns = [utils_labels.TOOL] + [col for col in df_summary if col != utils_labels.TOOL]
        print(df_summary[summary_columns].to_string(index=False))
    for tool, pd_group in pd_bins[pd_bins["rank"] == "NA"].groupby(utils_labels.TOOL):
        bins_columns = amber_html.get_genome_bins_columns()
        table = pd_group[["sample_id"] + list(bins_columns.keys())].rename(columns=dict(bins_columns))
        table.to_csv(os.path.join(output_dir, "genome", tool, "metrics_per_bin.tsv"), sep="\t", index=False)
    for tool, pd_group in pd_bins[pd_bins["rank"] != "NA"].groupby(utils_labels.TOOL):
        bins_columns = amber_html.get_tax_bins_columns()
        if "name" not in pd_bins.columns or pd_group["name"].isnull().any():
            del bins_columns["name"]
        table = pd_group[["sample_id"] + list(bins_columns.keys())].rename(columns=dict(bins_columns))
        table.to_csv(os.path.join(output_dir, "taxonomic", tool, "metrics_per_bin.tsv"), sep="\t", index=False)

    pd_genomes_all = pd.DataFrame()
    for sample_id in sample_id_to_queries_list:
        pd_genomes_sample = pd.DataFrame()
        for query in sample_id_to_queries_list[sample_id]:
            if isinstance(query, binning_classes.GenomeQuery):
                query.recall_df_cami1[utils_labels.TOOL] = query.label
                pd_genomes_sample = pd.concat(
                    [pd_genomes_sample, query.recall_df_cami1], ignore_index=True, sort=False
                )
        pd_genomes_sample["sample_id"] = sample_id
        pd_genomes_all = pd.concat([pd_genomes_all, pd_genomes_sample], ignore_index=True, sort=False)
    if not pd_genomes_all.empty:
        pd_genomes_all.to_csv(os.path.join(output_dir, "genome_metrics_cami1.tsv"), index=False, sep="\t")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="AMBER: Assessment of Metagenome BinnERs", parents=[argparse_parents.PARSER_MULTI2], prog="AMBER"
    )
    parser.add_argument("-p", "--filter", help=argparse_parents.HELP_FILTER)
    parser.add_argument("-n", "--min_length", help="Minimum length of sequences", type=int, required=False)
    parser.add_argument("-o", "--output_dir", help="Directory to write the results to", required=True)
    parser.add_argument("--stdout", help="Print summary to stdout", action="store_true")
    parser.add_argument("-d", "--desc", help="Description for HTML page", required=False)
    parser.add_argument("--colors", help="Color indices", required=False)
    parser.add_argument("--silent", help="Silent mode", action="store_true")
    parser.add_argument("-v", "--version", action="version", version="%(prog)s " + __version__)

    group_g = parser.add_argument_group("genome binning-specific arguments")
    group_g.add_argument(
        "-x", "--min_completeness", help=argparse_parents.HELP_THRESHOLDS_COMPLETENESS, required=False
    )
    group_g.add_argument(
        "-y", "--max_contamination", help=argparse_parents.HELP_THRESHOLDS_CONTAMINATION, required=False
    )
    group_g.add_argument("-r", "--remove_genomes", help=argparse_parents.HELP_GENOMES_FILE, required=False)
    group_g.add_argument("-k", "--keyword", help=argparse_parents.HELP_KEYWORD, required=False)
    group_g.add_argument("--genome_coverage", help="genome coverages", required=False)

    group_t = parser.add_argument_group("taxonomic binning-specific arguments")
    group_t.add_argument(
        "--ncbi_dir",
        help="Directory containing the NCBI taxonomy database dump files nodes.dmp, merged.dmp, and names.dmp",
        required=False,
    )
    # group_t.add_argument('--rank_as_genome_binning',
    #                      help="Assess taxonomic binning at a rank also as genome binning. Valid ranks: superkingdom, phylum, class, order, family, genus, species, strain",
    #                      required=False)

    args = parser.parse_args(args)
    output_dir = os.path.abspath(args.output_dir)
    logger = get_logger(output_dir, args.silent)

    labels = get_labels(args.labels, args.bin_files)

    genome_to_unique_common = load_data.load_unique_common(args.remove_genomes, args.keyword)

    options = binning_classes.Options(
        filter_tail_percentage=args.filter,
        genome_to_unique_common=genome_to_unique_common,
        filter_keyword=args.keyword,
        min_length=args.min_length,
        rank_as_genome_binning=None,  # args.rank_as_genome_binning,
        output_dir=output_dir,
        min_completeness=args.min_completeness,
        max_contamination=args.max_contamination,
    )
    options_gs = binning_classes.Options(
        filter_tail_percentage=0.0,
        genome_to_unique_common=genome_to_unique_common,
        filter_keyword=args.keyword,
        min_length=args.min_length,
        rank_as_genome_binning=None,  # args.rank_as_genome_binning,
        output_dir=output_dir,
    )

    load_data.load_ncbi_info(args.ncbi_dir)
    breakpoint()
    sample_id_to_queries_list, sample_ids_list = load_data.load_queries_mthreaded(
        args.gold_standard_file, args.bin_files, labels, options, options_gs
    )

    coverages_pd = load_data.open_coverages(args.genome_coverage)

    create_output_directories(output_dir, sample_id_to_queries_list)

    df_summary, pd_bins = evaluate_samples_queries(sample_id_to_queries_list)

    save_metrics(sample_id_to_queries_list, df_summary, pd_bins, output_dir, args.stdout)

    plot_genome_binning(
        args.colors,
        sample_id_to_queries_list,
        df_summary,
        pd_bins[pd_bins["rank"] == "NA"],
        labels,
        coverages_pd,
        output_dir,
    )
    plot_taxonomic_binning(args.colors, df_summary, pd_bins, labels, output_dir)

    amber_html.create_html(df_summary, pd_bins, [utils_labels.GS] + labels, sample_ids_list, options, args.desc)
    logger.info("AMBER finished successfully. All results have been saved to {}".format(output_dir))


if __name__ == "__main__":
    main()
