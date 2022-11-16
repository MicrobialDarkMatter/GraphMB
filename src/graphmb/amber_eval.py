from collections import OrderedDict
from collections import defaultdict
import itertools
import pandas as pd
from graphmb.contigsdataset import process_node_name


def write_amber_bins(contig_to_bin, outputfile):
    with open(outputfile, "w") as f:
        f.write("#\n@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
        for c in contig_to_bin:
            f.write(f"{str(c)}\t{str(contig_to_bin[c])}\n")

# adapt eval scripts from amber but simplified


def load_binnings(file_path_query, assemblytype, columns=["SEQUENCEID", "BINID"]):
    # columns = ["SEQUENCEID", "BINID", "TAXID", "LENGTH", "_LENGTH"]
    # sample_id_to_query_df = OrderedDict()
    #for metadata in columns:
        # logging.getLogger('amber').info('Loading ' + metadata[2]['SAMPLEID'])
        # nrows = metadata[1] - metadata[0] + 1
        # col_indices = [k for k, v in metadata[3].items() if v in columns]
        # amber files start with 4 header lines
    df = pd.read_csv(file_path_query, sep="\t", comment="#", skiprows=3, header=0)  # , usecols=col_indices)
    df = df.rename(columns={df.columns[i]: c for i, c in enumerate(columns)}, inplace=False)
    df = df.astype({'SEQUENCEID':'string'})
    df["SEQUENCEID"]= df["SEQUENCEID"].apply(process_node_name, assembly_type=assemblytype)
    if "_LENGTH" in columns:
        df.rename(columns={"_LENGTH": "LENGTH"}, inplace=True)
        df["LENGTH"] = pd.to_numeric(df["LENGTH"])
    return df


def amber_eval(gs_path, bin_path, labels=["graphmb"], assemblytype="flye"):
    gs_df = load_binnings(gs_path, columns=["SEQUENCEID", "BINID", "_LENGTH"], assemblytype=assemblytype)
    bin_df = load_binnings(bin_path, assemblytype=assemblytype)
    # load_queries(gs_df, bin_df, labels=labels, options, options_gs)
    gs_df = gs_df[["SEQUENCEID", "BINID", "LENGTH"]].rename(columns={"LENGTH": "seq_length",
                                                                       "BINID": "genome_id",
                                                                       "SEQUENCEID": "SEQUENCEID"})
    bin_df = bin_df[["SEQUENCEID", "BINID"]].rename(columns={"SEQUENCEID": "SEQUENCEID"})
    gs_df["seq_length"] = pd.to_numeric(gs_df["seq_length"])
    query_df = bin_df[["SEQUENCEID", "BINID"]]
    query_w_length = pd.merge(query_df, gs_df.drop_duplicates("SEQUENCEID"), on="SEQUENCEID", sort=False)
    query_w_length_no_dups = query_w_length.drop_duplicates("SEQUENCEID")
    gs_df_no_dups = gs_df.drop_duplicates("SEQUENCEID")
    percentage_of_assigned_bps = query_w_length_no_dups["seq_length"].sum() / gs_df_no_dups["seq_length"].sum()
    percentage_of_assigned_seqs = query_w_length_no_dups.shape[0] / gs_df_no_dups["SEQUENCEID"].shape[0]
    query_w_length_mult_seqs = query_df.reset_index().merge(gs_df, on="SEQUENCEID", sort=False)
    """if query_w_length.shape[0] < query_w_length_mult_seqs.shape[0]:
        query_w_length_mult_seqs.drop_duplicates(['index', 'genome_id'], inplace=True)
        confusion_df = query_w_length_mult_seqs.groupby(['BINID', 'genome_id'], sort=False).agg({'seq_length': 'sum', 'SEQUENCEID': 'count'}).rename(columns={'seq_length': 'genome_length', 'SEQUENCEID': 'genome_seq_counts'})

        most_abundant_genome_df = confusion_df.loc[confusion_df.groupby('BINID', sort=False)['genome_length'].idxmax()]
        most_abundant_genome_df = most_abundant_genome_df.reset_index()[['BINID', 'genome_id']]

        matching_genomes_df = pd.merge(query_w_length_mult_seqs, most_abundant_genome_df, on=['BINID', 'genome_id']).set_index('index')
        query_w_length_mult_seqs.set_index('index', inplace=True)
        difference_df = query_w_length_mult_seqs.drop(matching_genomes_df.index).groupby(['index'], sort=False).first()
        query_w_length = pd.concat([matching_genomes_df, difference_df])

        # Modify gs such that multiple binnings of the same sequence are not required
        matching_genomes_df = pd.merge(gs_df, query_w_length[['SEQUENCEID', 'genome_id']], on=['SEQUENCEID', 'genome_id'])
        matching_genomes_df = matching_genomes_df[['SEQUENCEID', 'genome_id', 'seq_length']].drop_duplicates(['SEQUENCEID', 'genome_id'])
        condition = gs_df_no_dups['SEQUENCEID'].isin(matching_genomes_df['SEQUENCEID'])
        difference_df = gs_df_no_dups[~condition]
        gs_df = pd.concat([difference_df, matching_genomes_df])"""

    # query_w_length_mult_seqs.reset_index(inplace=True)
    # query_w_length_mult_seqs = pd.merge(query_w_length_mult_seqs, most_abundant_genome_df, on=['BINID'])
    # grouped = query_w_length_mult_seqs.groupby(['index'], sort=False, as_index=False)
    # query_w_length = grouped.apply(lambda x: x[x['genome_id_x'] == x['genome_id_y'] if any(x['genome_id_x'] == x['genome_id_y']) else len(x) * [True]])
    # query_w_length = query_w_length.groupby(['index'], sort=False).first().drop(columns='genome_id_y').rename(columns={'genome_id_x': 'genome_id'})

    df = query_w_length

    confusion_df = (
        query_w_length.groupby(["BINID", "genome_id"], sort=False)
        .agg({"seq_length": "sum", "SEQUENCEID": "count"})
        .rename(columns={"seq_length": "genome_length", "SEQUENCEID": "genome_seq_counts"})
    )
    # self.confusion_df = confusion_df

    # rand_index_bp, adjusted_rand_index_bp = Metrics.compute_rand_index(
    #    confusion_df, "BINID", "genome_id", "genome_length"
    # )
    # rand_index_seq, adjusted_rand_index_seq = Metrics.compute_rand_index(
    #    confusion_df, "BINID", "genome_id", "genome_seq_counts"
    # )

    most_abundant_genome_df = (
        confusion_df.loc[confusion_df.groupby("BINID", sort=False)["genome_length"].idxmax()]
        .reset_index()
        .set_index("BINID")
    )

    query_w_length["seq_length_mean"] = query_w_length["seq_length"]

    precision_df = (
        query_w_length.groupby("BINID", sort=False)
        .agg({"seq_length": "sum", "seq_length_mean": "mean", "SEQUENCEID": "count"})
        .rename(columns={"seq_length": "total_length", "SEQUENCEID": "total_seq_counts"})
    )
    precision_df = pd.merge(precision_df, most_abundant_genome_df, on="BINID")
    precision_df.rename(columns={"genome_length": "tp_length", "genome_seq_counts": "tp_seq_counts"}, inplace=True)
    precision_df["precision_bp"] = precision_df["tp_length"] / precision_df["total_length"]
    precision_df["precision_seq"] = precision_df["tp_seq_counts"] / precision_df["total_seq_counts"]

    """if self.options.filter_tail_percentage:
        precision_df['total_length_pct'] = precision_df['total_length'] / precision_df['total_length'].sum()
        precision_df.sort_values(by='total_length', inplace=True)
        precision_df['cumsum_length_pct'] = precision_df['total_length_pct'].cumsum(axis=0)
        precision_df['precision_bp'].mask(precision_df['cumsum_length_pct'] <= self.options.filter_tail_percentage / 100, inplace=True)
        precision_df['precision_seq'].mask(precision_df['precision_bp'].isna(), inplace=True)
        precision_df.drop(columns=['cumsum_length_pct', 'total_length_pct'], inplace=True)
    if self.options.genome_to_unique_common:
        precision_df = precision_df[~precision_df['genome_id'].isin(self.options.genome_to_unique_common)]"""

    precision_avg_bp = precision_df["precision_bp"].mean()
    precision_avg_bp_sem = precision_df["precision_bp"].sem()
    precision_avg_bp_var = precision_df["precision_bp"].var()
    precision_avg_seq = precision_df["precision_seq"].mean()
    precision_avg_seq_sem = precision_df["precision_seq"].sem()
    precision_weighted_bp = precision_df["tp_length"].sum() / precision_df["total_length"].sum()
    precision_weighted_seq = precision_df["tp_seq_counts"].sum() / precision_df["total_seq_counts"].sum()

    genome_sizes_df = (
        gs_df.groupby("genome_id", sort=False)
        .agg({"seq_length": "sum", "SEQUENCEID": "count"})
        .rename(columns={"seq_length": "length_gs", "SEQUENCEID": "seq_counts_gs"})
    )
    precision_df = (
        precision_df.reset_index().join(genome_sizes_df, on="genome_id", how="left", sort=False).set_index("BINID")
    )
    precision_df["recall_bp"] = precision_df["tp_length"] / precision_df["length_gs"]
    precision_df["recall_seq"] = precision_df["tp_seq_counts"] / precision_df["seq_counts_gs"]
    precision_df["rank"] = "NA"

    recall_df = confusion_df.loc[confusion_df.groupby("genome_id", sort=False)["genome_length"].idxmax()]
    recall_df = (
        recall_df.reset_index().join(genome_sizes_df, on="genome_id", how="right", sort=False).set_index("BINID")
    )
    recall_df.fillna({"genome_length": 0, "genome_seq_counts": 0}, inplace=True)
    recall_df["recall_bp"] = recall_df["genome_length"] / recall_df["length_gs"]
    recall_df["recall_seq"] = recall_df["genome_seq_counts"] / recall_df["seq_counts_gs"]

    recall_df = recall_df.join(precision_df[["total_length", "seq_length_mean"]], how="left", sort=False)

    # if self.options.genome_to_unique_common:
    #    recall_df = recall_df[~recall_df["genome_id"].isin(self.options.genome_to_unique_common)]

    recall_avg_bp = recall_df["recall_bp"].mean()
    recall_avg_bp_var = recall_df["recall_bp"].var()
    recall_avg_bp_sem = recall_df["recall_bp"].sem()
    recall_avg_seq = recall_df["recall_seq"].mean()
    recall_avg_seq_sem = recall_df["recall_seq"].sem()
    recall_weighted_bp = recall_df["genome_length"].sum() / recall_df["length_gs"].sum()
    recall_weighted_seq = recall_df["genome_seq_counts"].sum() / recall_df["seq_counts_gs"].sum()

    # Compute recall as in CAMI 1
    """unmapped_genomes = set(gs_df["genome_id"].unique()) - set(precision_df["genome_id"].unique())
    #if self.options.genome_to_unique_common:
    #    unmapped_genomes -= set(self.options.genome_to_unique_common)
    num_unmapped_genomes = len(unmapped_genomes)
    prec_copy = precision_df.reset_index()
    if num_unmapped_genomes:
        prec_copy = prec_copy.reindex(
            prec_copy.index.tolist() + list(range(len(prec_copy), len(prec_copy) + num_unmapped_genomes))
        ).fillna(0.0)
    self.metrics.recall_avg_bp_cami1 = prec_copy["recall_bp"].mean()
    self.metrics.recall_avg_seq_cami1 = prec_copy["recall_seq"].mean()
    self.metrics.recall_avg_bp_sem_cami1 = prec_copy["recall_bp"].sem()
    self.metrics.recall_avg_seq_sem_cami1 = prec_copy["recall_seq"].sem()
    self.metrics.recall_avg_bp_var_cami1 = prec_copy["recall_bp"].var()
    self.recall_df_cami1 = prec_copy"""
    # End Compute recall as in CAMI 1

    accuracy_bp = precision_df["tp_length"].sum() / recall_df["length_gs"].sum()
    accuracy_seq = precision_df["tp_seq_counts"].sum() / recall_df["seq_counts_gs"].sum()

    precision_df = precision_df.sort_values(by=["recall_bp"], axis=0, ascending=False)
    recall_df = recall_df
    metrics = {
        "precision_avg_bp": precision_avg_bp,
        "precision_avg_bp_sem": precision_avg_bp_sem,
        "precision_avg_bp_var": precision_avg_bp_var,
        "precision_avg_seq": precision_avg_seq,
        "precision_avg_seq_sem": precision_avg_seq_sem,
        "precision_weighted_bp": precision_weighted_bp,
        "precision_weighted_seq": precision_weighted_seq,
        "recall_avg_bp": recall_avg_bp,
        "recall_avg_bp_sem": recall_avg_bp_sem,
        "recall_avg_bp_var": recall_avg_bp_var,
        "recall_avg_seq": recall_avg_seq,
        "recall_avg_seq_sem": recall_avg_seq_sem,
        "recall_weighted_bp": recall_weighted_bp,
        "recall_weighted_seq": recall_weighted_seq,
        "accuracy_bp": accuracy_bp,
        "accuracy_seq": accuracy_seq,
        "f1_avg_bp": (2*precision_avg_bp*recall_avg_bp)/(precision_avg_bp+recall_avg_bp) if (precision_avg_bp+recall_avg_bp) > 0 else 0
    }
    bins_eval = calc_num_recovered_genomes(precision_df, [0.9, 0.5], [0.05, 0.1])
    return metrics, bins_eval


def calc_num_recovered_genomes(bins, min_completeness, max_contamination):
    counts_list = []
    for x in itertools.product(min_completeness, max_contamination):
        count = bins[(bins["recall_bp"] > x[0]) & (bins["precision_bp"] > (1 - x[1]))].shape[0]
        counts_list.append(("> " + str(int(x[0] * 100)) + "% completeness", "< " + str(int(x[1] * 100)) + "%", count))

    pd_counts = pd.DataFrame(counts_list, columns=["Completeness", "Contamination", "count"])
    pd_counts = pd.pivot_table(
        pd_counts,
        values="count",
        index=["Contamination"],
        columns=["Completeness"],
    ).reset_index()
    return pd_counts
