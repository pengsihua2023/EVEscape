import pandas as pd
import numpy as np
import scipy.stats
from sklearn.preprocessing import StandardScaler
from Bio.PDB import PDBParser
from weighted_contact_number import *
from seq_utils import *

```
这段代码涉及到多种生物数据的处理，包括病毒复制和突变数据的读取与分析。具体到输入和输出文件：

### 输入数据文件：
1. **通用数据**：
   - `dissimilarity_metrics.csv`：包含氨基酸属性（如疏水性和电荷）。

2. **流感（H1N1）数据**：
   - `Doud2016_h1_replication.csv`：实验数据，包含H1N1病毒的复制水平变化。
   - `DMS_Doud2018_H1-WSN33_antibodies.csv`：包含抗体逃逸水平的实验数据。
   - `I4EPC4_t0.95_b0.1_evol_indices.csv`：进化指数数据。
   - `1rvx_no_HETATM.pdb`：蛋白质结构数据。
   - `A0A2Z5U3Z0_9INFA.fasta`：目标序列数据。

3. **HIV数据**：
   - `DMS_Haddox2018_hiv_BG505_env_replication_pref.csv`：HIV病毒BG505株的复制偏好数据。
   - `DMS_Dingens2019a_hiv_env_antibodies_x10.csv`：抗体逃逸实验数据。
   - `Q2N0S5_20-709_b0.1_evol_indices.csv`：进化指数数据。
   - `5FYL_Env_trimer.pdb` 和 `7tfo_env.pdb`：HIV病毒的结构数据。
   - `Q2N0S6_9HIV1.fasta`：HIV病毒的目标序列数据。

4. **SARS-CoV-2 RBD数据**：
   - `Starr2020_rbd_bind_expr.csv` 和 `escape_data_20220109.csv`：SARS-CoV-2 RBD的绑定表达数据和逃逸数据。
   - `abf1738_processed_data_file_from_deep_mutagenesis_of_sars-cov-2_protein_s.xlsx`：Chan实验室的数据。
   - `P0DTC2_321-541_sc0.5_cc0.3_b0.3_pre2020_evol_indices.csv`：进化指数数据。
   - `6vxx.pdb`、`6vyb.pdb`、`7bnn.pdb`、`7cab.pdb`：结构数据。
   - `SPIKE_SARS2.fasta`：目标序列数据。

5. **SARS-CoV-2 Spike数据**：
   - 使用与RBD相同的结构数据和目标序列数据。
   - `P0DTC2_sc0.5_cc0.3_b0.1_pre2020_evol_indices.csv`：进化指数数据。

6. **拉沙热病毒数据**：
   - `GLYC_LASSJ_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv`：进化指数数据。
   - `7puy_no_hetatm.pdb`：结构数据。
   - `GLYC_LASSJ.fasta`：目标序列数据。

7. **尼帕病毒数据**：
   - `GLYCP_NIPAV_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv` 和 `FUS_NIPAV_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv`：进化指数数据。
   - `7tyo_7txz_no_hetatm.pdb` 和 `5evm_no_hetatm.pdb`：结构数据。
   - `GLYCP_NIPAV.fasta` 和 `FUS_NIPAV.fasta`：目标序列数据。

### 输出数据文件：
1. `h1_experiments_and_scores.csv`：整合后的H1N1实验数据和分数。
2. `bg505_experiments_and_scores.csv`：整合后的HIV BG505实验数据和分数。
3. `rbd_experiments_and_scores.csv`：整合后的SARS-CoV-2 RBD实验

数据和分数。
4. `spike_scores.csv`：整合后的SARS-CoV-2 Spike蛋白数据和分数。
5. `lassa_glycoprotein_scores.csv`：整合后的拉沙热病毒数据和分数。
6. `nipah_glycoprotein_scores.csv` 和 `nipah_fusion_scores.csv`：整合后的尼帕病毒糖蛋白和融合蛋白数据和分数。

这些文件被用于进一步的数据分析和科研研究。

```



##############################################
# General Paths
##############################################

# AA properties () 通用数据：疏水性，电荷，BLOSUM62在突变前后的差异

aa_charge_hydro = '../data/aa_properties/dissimilarity_metrics.csv' 

##############################################
# Flu Paths
##############################################

# Experimental data
h1_replication = '../data/experiments/doud2016/Doud2016_h1_replication.csv' ## 复制水平变化实验数据
h1_escape = '../data/experiments/doud2018/DMS_Doud2018_H1-WSN33_antibodies.csv' ## 抗体逃逸变化水平实验数据
h1_experiment_range = (1, 565)

# Models
h1_eve = '../results/evol_indices/I4EPC4_t0.95_b0.1_evol_indices.csv'  ##进化指数数据（已经计算得到）

# Structure data

h1_pdb_id = '1RVX'
h1_pdb_path = '../data/structures/1rvx_no_HETATM.pdb'
h1_chains = ['A', 'B']
h1_trimer_chains = ['A', 'B', 'C', 'D', 'E', 'F']

h1_target_seq_path = '../data/sequences/A0A2Z5U3Z0_9INFA.fasta'

##############################################
# HIV Paths
##############################################

# Experimental data
bg505_replication = '../data/experiments/haddox2018/DMS_Haddox2018_hiv_BG505_env_replication_pref.csv'
bg505_escape = '../data/experiments/dingens2019/DMS_Dingens2019a_hiv_env_antibodies_x10.csv'
bg505_experiment_range = (30, 699)

# Models
bg505_eve = '../results/evol_indices/Q2N0S5_20-709_b0.1_evol_indices.csv'  ##进化指数数据（已经计算得到）

# Structure data
bg505_structure_list = [{
    'name':
    '5FYL',
    'chains': ['A', 'X'],
    'trimer_chains': ['A', 'B', 'C', 'X', 'Y', 'Z'],
    'pdb_path':
    '../data/structures/5FYL_Env_trimer.pdb'
}, {
    'name': '7tfo',
    'chains': ['A', 'X'],
    'trimer_chains': ['A', 'B', 'C', 'X', 'Y', 'Z'],
    'pdb_path': '../data/structures/7tfo_env.pdb'
}]

bg505_target_seq_path = '../data/sequences/Q2N0S6_9HIV1.fasta'

##############################################
# SARS2 RBD Paths
##############################################

# Experimental data
rbd_replication = '../data/experiments/starr2020/Starr2020_rbd_bind_expr.csv'
rbd_escape = '../data/experiments/bloom_rbd_escape/escape_data_20220109.csv'
rbd_chan = '../data/experiments/chan2020/abf1738_processed_data_file_from_deep_mutagenesis_of_sars-cov-2_protein_s.xlsx'
rbd_studies_to_drop = ['2021_Greaney_B1351']

# 认为rbd区域有201个氨基酸，从331-531.
rbd_experiment_range = (331, 531)

# Models
rbd_eve_pre2020 = '../results/evol_indices/P0DTC2_321-541_sc0.5_cc0.3_b0.3_pre2020_evol_indices.csv'  ##进化指数数据（已经计算得到）

# Structure data
rbd_structure_list = [{
    'name': '6VXX',
    'chains': ['A'],
    'trimer_chains': ['A', 'B', 'C'],
    'pdb_path': '../data/structures/6vxx.pdb'
}, {
    'name': '6VYB',
    'chains': ['B'],
    'trimer_chains': ['A', 'B', 'C'],
    'pdb_path': '../data/structures/6vyb.pdb'
}, {
    'name': '7BNN',
    'chains': ['B'],
    'trimer_chains': ['A', 'B', 'C'],
    'pdb_path': '../data/structures/7bnn.pdb'
}, {
    'name': '7CAB',
    'chains': ['A'],
    'trimer_chains': ['A', 'B', 'C'],
    'pdb_path': '../data/structures/7cab.pdb'
}]

rbd_target_seq_path = '../data/sequences/SPIKE_SARS2.fasta'

## RBD metadata save paths ##

bloom_ab_list = '../data/antibody_properties/Bloom_abs_to_use.txt'
xie_ab_list = '../data/antibody_properties/Xie_abs_to_use.txt'
rbd_ab_metadata = '../data/antibody_properties/rbd_antibody_metadata.csv'

##############################################
# SARS2 Spike Paths
##############################################

spike_target_seq_path = '../data/sequences/SPIKE_SARS2.fasta'
spike_experiment_range = (1, 1273)

# Models
spike_eve_pre2020 = '../results/evol_indices/P0DTC2_sc0.5_cc0.3_b0.1_pre2020_evol_indices.csv'  ##进化指数数据（已经计算得到）

##############################################
# Lassavirus Paths
##############################################

# Models
lassa_eve = '../results/evol_indices/GLYC_LASSJ_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv'   ##进化指数数据（已经计算得到）

# Structure data

lassa_pdb_id = '7PUY'
lassa_pdb_path = '../data/structures/7puy_no_hetatm.pdb'
lassa_chains = ['A', 'a']
lassa_trimer_chains = ['A', 'B', 'C', 'a', 'b', 'c']

lassa_target_seq_path = '../data/sequences/GLYC_LASSJ.fasta'
lassa_experiment_range = (59, 491) #signal peptide is 1-58


##############################################
# Nipahvirus glycoprotein Paths
##############################################

# Models
nipahg_eve = '../results/evol_indices/GLYCP_NIPAV_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv'  ##进化指数数据（已经计算得到）

# Structure data

nipahg_pdb_id = 'combo'
nipahg_pdb_path = '../data/structures/7tyo_7txz_no_hetatm.pdb'
nipahg_chains = [['A'], ['B'], ['C'], ['D']]
nipahg_multimer_chains = ['A', 'B', 'C', 'D']

nipahg_target_seq_path = '../data/sequences/GLYCP_NIPAV.fasta'

##############################################
# Nipahvirus Fusion Paths
##############################################

# Models
nipahf_eve = '../results/evol_indices/FUS_NIPAV_b0.05_theta_0.01_22oct14_20000_samples_No_distances_singles_22oct17.csv'  ##进化指数数据（已经计算得到）

# Structure data

nipahf_pdb_id = '5EVM'
nipahf_pdb_path = '../data/structures/5evm_no_hetatm.pdb'
nipahf_chains = ['A']
nipahf_trimer_chains = ['A', 'B', 'C']

nipahf_target_seq_path = '../data/sequences/FUS_NIPAV.fasta'

##############################################
# Data Processing Functions
##############################################
## 5月5日看到这里
## 提取进化指数结果文件中的数据
def process_eve_smm(eve_path):
    '''
    Processes EVE single mutation matrix table
    '''
    eve = pd.read_csv(eve_path)
    eve = eve[1:]
    eve.columns = eve.columns.str.replace("_ensemble", "")
    eve['wt'] = eve.mutations.str[0]
    eve['mut'] = eve.mutations.str[-1]
    eve['i'] = eve.mutations.str[1:-1].astype(int)
    eve['evol_indices'] = -eve.evol_indices
    to_drop = ['protein_name', 'mutations']
    to_drop.extend([col for col in eve.columns if "semantic_change" in col])
    eve = eve.drop(columns=to_drop)
    return eve

## 将进化向量编码（EVE）的预测结果与实验数据表合并
def add_model_outputs(exps, eve_path):
    '''
    Merges EVE predictions on to experimental data table
    '''
    exps = exps.merge(process_eve_smm(eve_path),
                      on=['wt', 'mut', 'i'],
                      how='outer')
    return exps

# 计算蛋白质结构中的加权接触数（WCN, Weighted Contact Number）并将这些信息合并到实验数据表中
def get_wcn(exps, pdb_path, trimer_chains, target_chains, map_table):
    '''
    Computes weighted contact number by alpha-carbon and sidechain 
    center of mass and merges on to experimental data table
    '''
# 使用 add_wcn_to_site_annotations 函数根据 PDB 文件和三聚体链的信息计算加权接触数。这涉及到蛋白质结构的 alpha-carbon 和侧链的质心。
    wcn = add_wcn_to_site_annotations(pdb_path, ''.join(trimer_chains))
    wcn = wcn.rename(columns={'pdb_position': 'i', 'pdb_aa': 'wt'})
    wcn['i'] = wcn.i.apply(lambda x: alphanumeric_index_to_numeric_index(x)
                           if (x != '') else x)
    wcn['i'] = wcn.i.replace('', np.nan)
    wcn = remap_struct_df_to_target_seq(wcn, target_chains, map_table)

    exps = exps.merge(wcn[['i', 'wcn_sc']], how='left', on='i')
    exps = exps.sort_values('i')
    exps['wcn_bfil'] = exps.wcn_sc.fillna(method='bfill')
    exps['wcn_ffil'] = exps.wcn_sc.fillna(method='ffill')
    exps['wcn_fill'] = (
        exps[['wcn_ffil', 'wcn_bfil']].sum(axis=1, min_count=2) / 2)
    exps = exps.drop(columns=['wcn_bfil', 'wcn_ffil'])
    return exps

# 将氨基酸的疏水性和电荷差异的标准化数据合并到实验数据表中，并计算两者的综合分数
def hydrophobicity_charge(exps, table):

    props = pd.read_csv(table, index_col=0)

    scale = StandardScaler()
    props['eisenberg_weiss_diff_std'] = scale.fit_transform(
        props['eisenberg_weiss_diff'].abs().values.reshape(-1, 1))
    props['charge_diff_std'] = scale.fit_transform(
        props['charge_diff'].abs().values.reshape(-1, 1))
    exps = exps.merge(props, how='left', on=['wt', 'mut'])

    exps['charge_ew-hydro'] = exps[[
        'eisenberg_weiss_diff_std', 'charge_diff_std'
    ]].sum(axis=1)
    exps = exps.drop(columns=['eisenberg_weiss_diff_std', 'charge_diff_std'])
    return exps

# 通过野生型的值将实验变量进行标准化处理，这有助于消除基线差异，使不同的实验数据点可以在相同的基准下进行比较，进而分析突变的效果。
def norm_to_wt(df, prefvar):
    '''
    Normalize experimental variables to wildtype (for "prefs" style data) 
    '''
    newvar = 'norm_' + prefvar

    def grp_func(grp):

        ref = grp[grp['wt'] == grp['mut']][prefvar].mean()
        grp[newvar] = grp[prefvar] / ref
        return grp

    df[newvar] = df[prefvar]
    df = df.groupby(['i', 'wt']).apply(grp_func)
    return df
# 整理和存储与 RBD 实验条件相关的元数据，同时为特定实验室的分析提供便捷的文本文件，从而支持数据的进一步处理和分析。
def rbd_metadata(escape_df, bloom_path, xie_path, metadata_path):
    escape = escape_df[['condition','condition_type',
                        'condition_subtype','condition_year',
                        'eliciting_virus','study',
                        'lab']].drop_duplicates()
    with open(xie_path, "w") as textfile:
        for element in escape[
                          (escape.lab=='Xie_XS')].condition.tolist():
            textfile.write(element + "\n")
    with open(bloom_path, "w") as textfile:
        for element in escape[
                          (escape.lab=='Bloom_JD')].condition.tolist():
            textfile.write(element + "\n")
    escape.to_csv(metadata_path)
    return(escape)


##############################################
# Summary workbook functions
##############################################

# 整合和处理针对 H1N1 流感病毒的实验数据和模型预测结果，并且关联这些数据到蛋白质结构信息
def load_H1():

    # Read in and combine experimental data
    escape = pd.read_csv(h1_escape).drop(columns=['resi'])
    cols = ['wt', 'mut', 'i'] + [
        col for col in escape.columns if 'median_mutfracsurvive' in col
    ]
    escape = escape[cols]
    rep = pd.read_csv(h1_replication)
    rep = rep.rename(columns={'norm_tf_prefs': 'flu_h1_replication'})
    data = escape.merge(rep[['wt', 'mut', 'i', 'flu_h1_replication']],
                        on=['wt', 'mut', 'i'],
                        how='outer')

    # Read in and combine model data
    data = add_model_outputs(data, h1_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    map_table = remap_pdb_seq_to_target_seq(h1_pdb_path, h1_chains,
                                            h1_target_seq_path)

    # Calculated weighted contact counts
    data = get_wcn(data, h1_pdb_path, h1_trimer_chains, h1_chains, map_table)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= h1_experiment_range[0])
                & (data.i <= h1_experiment_range[1])]

    return data, map_table

# 整合和处理针对 HIV-1 BG505 病毒的实验数据和模型预测结果，并且关联这些数据到蛋白质结构信息
def load_bg505():

    # Read in and combine experimental data
    escape = pd.read_csv(bg505_escape)
    cols = (['wt', 'mut', 'i'] + [
        col for col in escape.columns
        if 'summary' in col and 'medianmutfracsurvive' in col
    ])
    escape = escape[cols]

    # DATA IS NOT WT NORMALIZED - fixed here
    rep = pd.read_csv(bg505_replication)
    rep = norm_to_wt(rep, 'prefs')
    rep['norm_tf_prefs'] = np.log(rep['norm_prefs'])
    rep = rep.rename(columns={'norm_tf_prefs': 'hiv_env_replication'})
    data = escape.merge(rep[['wt', 'mut', 'i', 'hiv_env_replication']],
                        on=['wt', 'mut', 'i'],
                        how='outer')

    # Read in and combine model data
    data = add_model_outputs(data, bg505_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    # Add WCNs to dataframe
    map_table_dict = {}

    for struct in bg505_structure_list:

        map_table = remap_pdb_seq_to_target_seq(struct['pdb_path'],
                                                struct['chains'],
                                                bg505_target_seq_path)

        data = get_wcn(data, struct['pdb_path'], struct['trimer_chains'],
                       struct['chains'], map_table)

        data = data.rename(
            columns={
                'wcn_sc': 'wcn_sc_' + struct['name'],
                'wcn_fill': 'wcn_fill_' + struct['name']
            })

        map_table_dict[struct['name']] = map_table

    # Take min of wcn from structures
    data['wcn_fill'] = data[[
        col for col in data.columns if 'wcn_fill_' in col
    ]].min(axis=1)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= bg505_experiment_range[0])
                & (data.i <= bg505_experiment_range[1])]

    return data, map_table_dict

# 加载和处理与 SARS-CoV-2 的受体结合域（RBD）相关的实验数据，并将这些数据与预测模型和蛋白质结构信息整合
def load_rbd():

    # Read in and combine experimental data
    escape = pd.read_csv(rbd_escape)
    escape = escape[~escape.study.isin(rbd_studies_to_drop)]
    escape['condition'] = ('escape_' + escape['condition'] + '_' +
                           escape['lab'].str.split('_').str[0])
    
    _ = rbd_metadata(escape, bloom_ab_list, xie_ab_list, rbd_ab_metadata)
    
    escape = escape[[
        'condition', 'site', 'wildtype', 'mutation', 'mut_escape'
    ]]

    escape = pd.pivot_table(
        escape,
        index=['site', 'wildtype', 'mutation'],
        columns="condition",
        values="mut_escape").reset_index().rename_axis(None).rename_axis(
            None, axis=1)

    escape = escape.rename(columns={
        'site': 'i',
        'wildtype': 'wt',
        'mutation': 'mut'
    })
    escape = escape.fillna(0)

    rep = pd.read_csv(rbd_replication)
    rep = rep.rename(columns={
        'bind_avg': 'rbd_ace2_binding',
        'expr_avg': 'rbd_expression'
    })

    data = escape.merge(
        rep[['wt', 'mut', 'i', 'rbd_ace2_binding', 'rbd_expression']],
        on=['wt', 'mut', 'i'],
        how='outer')

    #chan data
    chan = pd.read_excel(
        '../data/experiments/chan2020/abf1738_processed_data_file_from_deep_mutagenesis_of_sars-cov-2_protein_s.xlsx',
        skiprows=8)

    chan['Unnamed: 0'] = chan['Unnamed: 0'].ffill()
    chan['WT a.a.'] = chan['WT a.a.'].ffill()
    chan['Position #'] = chan['Position #'].ffill()
    chan = chan[chan.Mutation != '*']
    chan = chan.drop(columns=[
        'Unnamed: 0', 'WT-specific 1', 'WT-specific 2', 'v2.4-specific 1',
        'v2.4-specific 2'
    ])
    chan['chan_expression'] = chan['ACE2-High'] + chan['ACE2-Low']
    chan['chan_ace2_binding'] = chan['ACE2-High']
    chan = chan.drop(columns=['ACE2-High', 'ACE2-Low'])
    chan = chan.rename(columns={
        'WT a.a.': 'wt',
        'Position #': 'i',
        'Mutation': 'mut'
    })

    data = data.merge(chan, how='left', on=['wt', 'i', 'mut'])

    # Read in and combine model data
    data = add_model_outputs(data, rbd_eve_pre2020)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    # Add WCNs to dataframe
    map_table_dict = {}

    for struct in rbd_structure_list:

        map_table = remap_pdb_seq_to_target_seq(struct['pdb_path'],
                                                struct['chains'],
                                                rbd_target_seq_path)

        data = get_wcn(data, struct['pdb_path'], struct['trimer_chains'],
                       struct['chains'], map_table)

        data = data.rename(
            columns={
                'wcn_sc': 'wcn_sc_' + struct['name'],
                'wcn_fill': 'wcn_fill_' + struct['name']
            })

        map_table_dict[struct['name']] = map_table

    # Take min of structures
    data['wcn_fill'] = data[[
        col for col in data.columns if 'wcn_fill_' in col
    ]].min(axis=1)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= rbd_experiment_range[0])
                & (data.i <= rbd_experiment_range[1])]

    return data, map_table_dict

# 加载和处理 SARS-CoV-2 Spike 蛋白的相关数据，并且将其与结构数据及预测模型输出整合
def load_spike():

    # Make starting dataframe
    data = make_mut_table(spike_target_seq_path)

    # Read in and combine pre-2020 model data
    data = add_model_outputs(data, spike_eve_pre2020)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    # Add WCNs to dataframe
    map_table_dict = {}

    for struct in rbd_structure_list:

        map_table = remap_pdb_seq_to_target_seq(struct['pdb_path'],
                                                struct['chains'],
                                                spike_target_seq_path)
        data = get_wcn(data, struct['pdb_path'], struct['trimer_chains'],
                       struct['chains'], map_table)
        data = data.rename(
            columns={
                'wcn_sc': 'wcn_sc_' + struct['name'],
                'wcn_fill': 'wcn_fill_' + struct['name']
            })

        map_table_dict[struct['name']] = map_table

    # Take min of structures
    data['wcn_fill'] = data[[
        col for col in data.columns if 'wcn_fill_' in col
    ]].min(axis=1)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= spike_experiment_range[0])
                & (data.i <= spike_experiment_range[1])]

    return data, map_table_dict

# 加载和处理拉沙热病毒（Lassa virus）的序列突变数据，并将其与结构和模型预测数据结合
def load_lassa():
    data = make_mut_table(lassa_target_seq_path)

    # Read in and combine model data
    data = add_model_outputs(data, lassa_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    map_table = remap_pdb_seq_to_target_seq(lassa_pdb_path, lassa_chains,
                                            lassa_target_seq_path)

    # Calculated weighted contact counts
    data = get_wcn(data, lassa_pdb_path, lassa_trimer_chains, lassa_chains, map_table)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    # Drop any rows not in experiment
    data = data[(data.i >= lassa_experiment_range[0])
                & (data.i <= lassa_experiment_range[1])]

    return data, map_table

# 加载和处理与尼帕病毒(Nipah virus)糖蛋白相关的突变数据，并将其与模型预测结果及结构分析数据整合
def load_nipahg():
    data = make_mut_table(nipahg_target_seq_path)

    # Read in and combine model data
    data = add_model_outputs(data, nipahg_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Calculated weighted contact counts
    map_table_dict = {}

    for chain in nipahg_chains:

        map_table = remap_pdb_seq_to_target_seq(nipahg_pdb_path,
                                                chain,
                                                nipahg_target_seq_path)

        data = get_wcn(data, nipahg_pdb_path, nipahg_multimer_chains,
                       chain, map_table)

        data = data.rename(
            columns={
                'wcn_sc': 'wcn_sc_chain_' + ''.join(chain),
                'wcn_fill': 'wcn_fill_chain_' + ''.join(chain)
            })

        map_table_dict[''.join(chain)] = map_table

    # Take min of structures
    data['wcn_fill'] = data[[
        col for col in data.columns if 'wcn_fill_' in col
    ]].min(axis=1)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    return data, map_table_dict

# 加载和处理尼帕病毒融合蛋白（Nipah virus fusion protein, F protein）的突变数据
def load_nipahf():
    data = make_mut_table(nipahf_target_seq_path)

    # Read in and combine pre-2020 model data
    data = add_model_outputs(data, nipahf_eve)

    # Get rid of wt data
    data = data[data.wt != data.mut]

    # Get mapping to PDB
    map_table = remap_pdb_seq_to_target_seq(nipahf_pdb_path, nipahf_chains,
                                            nipahf_target_seq_path)

    # Calculated weighted contact counts
    data = get_wcn(data, nipahf_pdb_path, nipahf_trimer_chains, nipahf_chains, map_table)

    # Add aa properties to data
    data = hydrophobicity_charge(data, aa_charge_hydro)
    data = data.sort_values(['i', 'mut'])

    return data, map_table
# 加载H1实验数据
h1, _ = load_H1()
h1.to_csv('../results/summaries/h1_experiments_and_scores.csv', index=False)

# 加载bg505实验数据
bg505, _ = load_bg505()
bg505.to_csv('../results/summaries/bg505_experiments_and_scores.csv',
             index=False)

# 加载rbd实验数据
rbd, _ = load_rbd()
rbd.to_csv('../results/summaries/rbd_experiments_and_scores.csv', index=False)

# 加载spike实验数据
spike, _ = load_spike()
spike.to_csv('../results/summaries/spike_scores.csv', index=False)

# 加载lassa实验数据
lassa, _ = load_lassa()
lassa.to_csv('../results/summaries/lassa_glycoprotein_scores.csv', index=False)

# 加载nipahg实验数据
nipahg, _ = load_nipahg()
nipahg.to_csv('../results/summaries/nipah_glycoprotein_scores.csv', index=False)

# 加载nipahf实验数据
nipahf, _ = load_nipahf()
nipahf.to_csv('../results/summaries/nipah_fusion_scores.csv', index=False)
