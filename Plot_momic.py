import os
import shutil
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency
import re
import subprocess
import matplotlib.pyplot as plt
import numpy as np
# 1. Split Files Based on AD and calculate average attention
def split_files_and_calculate_average_attention(patient_type_one, patient_type_two):
    map_dict_path = 'ROSMAP-graph-data/survival_label_map_dict.csv'
    map_dict_df = pd.read_csv(map_dict_path)
    num_to_id_dict = pd.Series(map_dict_df['individualID'].values, index=map_dict_df['individualID_Num']).to_dict()

    label_path = 'ROSMAP-graph-data/survival-label.csv'
    label_df = pd.read_csv(label_path)
    if patient_type_one in ['AD', 'NOAD']:
        id_to_dict = pd.Series(label_df['ceradsc'].values, index=label_df['individualID']).to_dict()
    else:
        id_to_dict = pd.Series(label_df['msex'].values, index=label_df['individualID']).to_dict()

    survival_dir = './ROSMAP-analysis/avg/'
    files = os.listdir(survival_dir)
    os.makedirs('./ROSMAP-analysis/avg_analysis', exist_ok=True)
    patient_type_one_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_one}'
    patient_type_two_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_two}'

    # Make directories if they don't exist
    os.makedirs(patient_type_one_dir, exist_ok=True)
    os.makedirs(patient_type_two_dir, exist_ok=True)

    for file in files:
        if file.endswith('.csv'):
            num = int(file.split('survival')[1].split('.csv')[0])

            if num in num_to_id_dict:
                individual_id = num_to_id_dict[num]

                if individual_id in id_to_dict:
                    value = id_to_dict[individual_id]

                    if value == 0:
                        shutil.copy(os.path.join(survival_dir, file), os.path.join(patient_type_two_dir, file))
                    elif value == 1:
                        shutil.copy(os.path.join(survival_dir, file), os.path.join(patient_type_one_dir, file))

    def calculate_average_attention(folder_path):
        all_data = []
        key_columns = ['From', 'To', 'Hop', 'SignalingPath', 'SpNotation', 'EdgeType']
        
        # Read each file and collect the data
        for filename in os.listdir(folder_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(folder_path, filename)
                df = pd.read_csv(file_path)
                
                # Select relevant columns, ensuring 'individualID' is not included
                if 'individualID' in df.columns:
                    df = df.drop(columns=['individualID'])
                
                # Check if all necessary columns are present
                if all(col in df.columns for col in key_columns + ['Attention']):
                    all_data.append(df)
                else:
                    print(f"File {filename} is missing one of the required columns.")
        
        # Concatenate all the dataframes in the list
        if not all_data:
            print("No valid files to process.")
            return None
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Group by the key columns and calculate the mean of 'Attention'
        result_df = combined_df.groupby(key_columns)['Attention'].mean().reset_index()
        
        return result_df

    patient_type_one_result_df = calculate_average_attention(patient_type_one_dir)
    patient_type_one_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_one}.csv', index=False)
    patient_type_two_result_df = calculate_average_attention(patient_type_two_dir)
    patient_type_two_result_df.to_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_two}.csv', index=False)

    
    def filter_edges(patient_type):

        df = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type}.csv')
        filtered_df = df[df['EdgeType'] == 'Gene-PROT-Gene-PROT']
        filtered_df.to_csv(f'./ROSMAP-analysis/avg_analysis/filtered_average_attention_{patient_type}.csv', index=False)
        
    filter_edges(patient_type_one)
    filter_edges(patient_type_two)

# 2. Generate the gene_name list based on the filtered average attention and the threshold
def filter_gene_name(patient_type,threshold):   
    # Read network edge weight data
    net_edge_weight = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/filtered_average_attention_{patient_type}.csv')
    net_edge_weight.columns = ['From', 'To', 'Hop', 'SignalingPath', 'SpNotation', 'EdgeType','Attention']
    
    # Read node data
    net_node = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')  # NODE LABEL
    
    ### 2.1 FILTER EDGE BY [edge_weight]
    edge_threshold = threshold
    filter_net_edge = net_edge_weight[net_edge_weight['Attention'] > edge_threshold]
    filter_net_edge_node = pd.unique(filter_net_edge[['From', 'To']].values.ravel('K'))
    filter_net_node = net_node[net_node['Gene_num'].isin(filter_net_edge_node)]
    
    ### 2.2 FILTER WITH GIANT COMPONENT
    tmp_net = nx.from_pandas_edgelist(filter_net_edge, 'From', 'To')
    all_components = list(nx.connected_components(tmp_net))
    
    # COLLECT ALL LARGE COMPONENTS
    giant_comp_node = []
    giant_comp_threshold = 20
    for component in all_components:
        if len(component) >= giant_comp_threshold:
            giant_comp_node.extend(component)
    
    refilter_net_edge = filter_net_edge[(filter_net_edge['From'].isin(giant_comp_node)) | 
                                        (filter_net_edge['To'].isin(giant_comp_node))]
    refilter_net_edge_node = pd.unique(refilter_net_edge[['From', 'To']].values.ravel('K'))
    refilter_net_node = filter_net_node[filter_net_node['Gene_num'].isin(refilter_net_edge_node)]
    refilter_net_node_name = refilter_net_node['Gene_name']
    refilter_net_node_name.to_csv(f'./ROSMAP-analysis/avg_analysis/{patient_type}_gene_names.csv', index=False)

# 3. Calculate pvalue for each gene_name
def calculate_pvalue(patient_type_one, patient_type_two):
   # count survival numbers for each patient type
    patient_type_one_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_one}'
    patient_type_two_dir = f'./ROSMAP-analysis/avg_analysis/{patient_type_two}'

    def get_survival_numbers(folder_path):
        files = os.listdir(folder_path)
        survival_numbers = []
        for file in files:
            match = re.match(r'survival(\d+)\.csv', file)
            if match:
                number = int(match.group(1))
                survival_numbers.append(number)
        return survival_numbers

    survival_numbers_patient_type_one = sorted(get_survival_numbers(patient_type_one_dir))
    survival_numbers_patient_type_two = sorted(get_survival_numbers(patient_type_two_dir))
    df = pd.read_csv('./ROSMAP-graph-data/survival_label_map_dict.csv')

    def num_to_id(num):
        return df.loc[df['individualID_Num'] == num, 'individualID'].values[0]


    survival_numbers_patient_type_one = [num_to_id(num) for num in survival_numbers_patient_type_one]
    survival_numbers_patient_type_two = [num_to_id(num) for num in survival_numbers_patient_type_two]
    print("Survival numbers:", survival_numbers_patient_type_one)
    print("Survival numbers:", survival_numbers_patient_type_two)

    # count genes for each patient type
    gene_names_df_patient_type_one = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/{patient_type_one}_gene_names.csv')
    gene_names_patient_type_one = gene_names_df_patient_type_one['Gene_name'].tolist()
    gene_names_patient_type_one = [name.replace('-PROT', '') for name in gene_names_patient_type_one]
    print(gene_names_patient_type_one)
    gene_names_df_patient_type_two = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/{patient_type_two}_gene_names.csv')
    gene_names_patient_type_two = gene_names_df_patient_type_two['Gene_name'].tolist()
    gene_names_patient_type_two = [name.replace('-PROT', '') for name in gene_names_patient_type_two]
    print(gene_names_patient_type_two)

    files_to_process = [
        './ROSMAP-process/processed-genotype-cnv_del.csv',
        './ROSMAP-process/processed-genotype-cnv_dup.csv',
        './ROSMAP-process/processed-genotype-cnv_mcnv.csv',
        './ROSMAP-process/processed-genotype-gene-expression.csv',
        './ROSMAP-process/processed-genotype-methy-Core-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Distal-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Downstream.csv',
        './ROSMAP-process/processed-genotype-methy-Proximal-Promoter.csv',
        './ROSMAP-process/processed-genotype-methy-Upstream.csv',
    ]

    output_dir_patient_type_one = f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/'
    output_dir_patient_type_two = f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/'
    # mkdir
    os.makedirs(output_dir_patient_type_one, exist_ok=True)
    os.makedirs(output_dir_patient_type_two, exist_ok=True)

    #split files for each patient type
    def process_data(file_path, gene_names, survival_numbers, output_path):
        data = pd.read_csv(file_path)

        columns_to_keep = [num for num in survival_numbers]
        columns_to_keep = ['gene_name'] + columns_to_keep 
        filtered_data = data[columns_to_keep]

        filtered_data = filtered_data[filtered_data['gene_name'].isin(gene_names)]

        filtered_data.to_csv(output_path, index=False)
        print(f"Filtered data saved to {output_path}")

    for file_path in files_to_process:
        file_name = file_path.split('/')[-1].replace('.csv', f'_{patient_type_one}.csv')
        output_path = f"{output_dir_patient_type_one}{file_name}"
        process_data(file_path, gene_names_patient_type_one, survival_numbers_patient_type_one, output_path)

    for file_path in files_to_process:
        file_name = file_path.split('/')[-1].replace('.csv', f'_{patient_type_two}.csv')
        output_path = f"{output_dir_patient_type_two}{file_name}"
        process_data(file_path, gene_names_patient_type_two, survival_numbers_patient_type_two, output_path)
    

    patient_type_one_files = [
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_del_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_dup_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-cnv_mcnv_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-gene-expression_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Core-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Distal-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Downstream_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Proximal-Promoter_{patient_type_one}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_one}/processed-genotype-methy-Upstream_{patient_type_one}.csv',
    ]

    patient_type_two_files = [
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_del_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_dup_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-cnv_mcnv_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-gene-expression_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Core-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Distal-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Downstream_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Proximal-Promoter_{patient_type_two}.csv',
        f'./ROSMAP-analysis/node_analysis_processed/{patient_type_two}/processed-genotype-methy-Upstream_{patient_type_two}.csv',
    ]

    # mkdir pvalues folder
    os.makedirs('./ROSMAP-analysis/node_analysis_processed/pvalues/', exist_ok=True)

    def calculate(AD_file_path, NOAD_file_path):
        AD_data = pd.read_csv(AD_file_path)
        NOAD_data = pd.read_csv(NOAD_file_path)

        results = []

        for gene in AD_data['gene_name']:
            AD_row = AD_data[AD_data['gene_name'] == gene].iloc[:, 1:].values.flatten()
            NOAD_row = NOAD_data[NOAD_data['gene_name'] == gene].iloc[:, 1:].values.flatten()
            # for gene only in ad or noad
            if len(AD_row) == 0 or len(NOAD_row) == 0:
                print(f"Skipping gene {gene}: One of the rows is empty")
                continue

            # # If the gene data is only present in one file, assign a p-value of 1e-25
            # if len(AD_row) == 0 or len(NOAD_row) == 0:
            #     results.append({'gene_name': gene, 'p_value':1e-25})
            #     print(f"Including gene {gene} with missing data in one group; assigned p-value of 1e-25")
            #     continue

            contingency_table = pd.crosstab(AD_row, NOAD_row)
            chi2, p, dof, expected = chi2_contingency(contingency_table)
            results.append({'gene_name': gene, 'p_value': p})
        results_df = pd.DataFrame(results)
        return results_df

    for patient_type_one_file, patient_type_two_file in zip(patient_type_one_files, patient_type_two_files):
        results_df = calculate(patient_type_one_file, patient_type_two_file)
        file_name = patient_type_one_file.split('/')[-1].replace(f'_{patient_type_one}.csv', '_pvalues.csv')
        results_df.to_csv(f'./ROSMAP-analysis/node_analysis_processed/pvalues/{file_name}', index=False)
        print(f"P-values calculated and saved for {file_name}")



    
# 4. Filter top pvalues
def filter_top_pvalues(top):
    pvalues_files = [
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-cnv_del_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-cnv_dup_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-cnv_mcnv_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-gene-expression_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-methy-Core-Promoter_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-methy-Distal-Promoter_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-methy-Downstream_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-methy-Proximal-Promoter_pvalues.csv',
        './ROSMAP-analysis/node_analysis_processed/pvalues/processed-genotype-methy-Upstream_pvalues.csv',
    ]
    node_type_mapping = {
        'cnv_del': 'TRAN',
        'cnv_dup': 'TRAN',
        'cnv_mcnv': 'TRAN',
        'gene-expression': 'TRAN',
        'methy-Core-Promoter': 'METH',
        'methy-Distal-Promoter': 'METH',
        'methy-Downstream': 'METH',
        'methy-Proximal-Promoter': 'METH',
        'methy-Upstream': 'METH'
    }
    filtered_results = pd.DataFrame(columns=['NodeType', 'Gene_name', 'Category', 'P_value'])

    for file_path in pvalues_files:
        category = file_path.split('-genotype-')[1].split('_pvalues')[0]
        data = pd.read_csv(file_path)
        significant_data = data[data['p_value'] < 0.2]
        output_data = pd.DataFrame({
            'NodeType': 'Gene-' + node_type_mapping[category],
            'Gene_name': significant_data['gene_name'].astype(str) + '-' + node_type_mapping[category],
            'Category': category,
            'P_value': significant_data['p_value']
        })
        filtered_results = pd.concat([filtered_results, output_data], ignore_index=True)

    filtered_results = filtered_results.sort_values(by='P_value', ascending=True)

    if len(filtered_results) < top:
        print(f"The filtered pvalues has already smaller than {top}.")
        print(filtered_results)
        raise Exception(f"The filtered pvalues has already smaller than {top}. Process terminated.")
    else:
        top_results = filtered_results.head(top)
        filename = f'./ROSMAP-analysis/p_values_{top}.csv'
        print(f"Your number of filtered pvalues are: {len(filtered_results)}, and picked top {top} values.")
        print(filename)
        top_results.to_csv(filename, index=False)

    # Modify the pvalues file for plotting
    df = pd.read_csv(f'./ROSMAP-analysis/p_values_{top}.csv')
    new_rows = []
    for index, row in df.iterrows():
        category = row['Category']
        
        if category.startswith('cnv'):
            df.at[index, 'Category'] = 'mutation'
        elif category in ['gene-expression', 'methy-Core-Promoter', 'methy-Distal-Promoter',
                        'methy-Downstream', 'methy-Proximal-Promoter', 'methy-Upstream']:
            new_row = row.copy()
            new_row['Category'] = 'mutation'
            new_rows.append(new_row)
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv(f'./ROSMAP-analysis/modified_p_values_{top}.csv', index=False)

# 5. Data filtering and processing
def data_filtering_and_processing(patient_type_one, patient_type_two,top,threshold):
    # Data filtering
    def filter_attention_data(patient_type,threshold):

        df = pd.read_csv(f'./ROSMAP-analysis/avg_analysis/filtered_average_attention_{patient_type}.csv')
        
        rows_to_keep = []

        for _, row in df.iterrows():
            # Define conditions for keeping rows based on attention thresholds
            forward_condition = (df['From'] == row['From']) & (df['To'] == row['To']) & (df['Attention'] > threshold)
            backward_condition = (df['From'] == row['To']) & (df['To'] == row['From']) & (df['Attention'] > threshold)

            # Check if any row meets the forward or backward conditions
            if df[forward_condition].any(axis=0).any() or df[backward_condition].any(axis=0).any():
                rows_to_keep.append(row)

        result_df = pd.DataFrame(rows_to_keep).drop_duplicates()
        return result_df

    patient_type_one_ht = filter_attention_data(patient_type_one,threshold)
    patient_type_two_ht = filter_attention_data(patient_type_two,threshold)

    #bfs
    def clean_attention_data(patient_type,top,df_ht):
        # Load gene number mapping data
        map_all_gene = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
        gene_name_to_num = map_all_gene.set_index('Gene_name')['Gene_num'].to_dict()

        # Load p_values data and convert gene names
        p_values = pd.read_csv(f'./ROSMAP-analysis/modified_p_values_{top}.csv')
        important_gene_names = p_values['Gene_name'].apply(lambda x: x.split('-')[0] + '-PROT').unique()

        # Get important gene numbers
        important_gene_nums = set()
        for gene_name in important_gene_names:
            if gene_name in gene_name_to_num:
                important_gene_nums.add(gene_name_to_num[gene_name])

        changes = True
        while changes:
            # Recompute the number of unique 'To' for each 'From'
            unique_links = df_ht.groupby('From')['To'].nunique()
            
            # Apply the filter condition: remove rows where 'From' connects to a unique 'To' and does not involve important genes
            initial_count = len(df_ht)
            to_remove = df_ht[(df_ht['From'].map(unique_links) == 1) &
                        (~df_ht['From'].isin(important_gene_nums)) &
                        (~df_ht['To'].isin(important_gene_nums))]

            if not to_remove.empty:
                # Create a reversed rows DataFrame
                reversed_rows = to_remove[['To', 'From']].copy()
                reversed_rows.columns = ['From', 'To']

                # Check if the reversed rows exist in the original data
                reversed_rows['is_reversed'] = reversed_rows.apply(lambda x: ((df_ht['From'] == x['From']) & 
                                                                            (df_ht['To'] == x['To']) ).any(), axis=1)

                # Get the indices of all reversed rows marked as True
                reversed_indices = df_ht[df_ht.apply(lambda x: ((reversed_rows['From'] == x['From']) & 
                                                        (reversed_rows['To'] == x['To']) & 
                                                        reversed_rows['is_reversed']).any(), axis=1)].index

                # Merge the indices of the original and reversed rows
                indices_to_remove = to_remove.index.union(reversed_indices)
                df_ht = df_ht.drop(indices_to_remove)

            changes = len(df_ht) < initial_count
        return df_ht

    patient_type_one_ht_step1 = clean_attention_data(patient_type_one,top,patient_type_one_ht)
    patient_type_two_ht_step1 = clean_attention_data(patient_type_two,top,patient_type_two_ht)

    ## Find the top 2 path between each hub gene nodes (prot nodes' unique link > 2) and remove the rest node

    def load_gene_information(top):
        # Load p_values data and convert gene names
        p_values = pd.read_csv(f'./ROSMAP-analysis/modified_p_values_{top}.csv')
        important_gene_names = p_values['Gene_name'].apply(lambda x: x.split('-')[0] + '-PROT').unique()

        # Load gene number mapping data
        map_all_gene = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
        gene_name_to_num = map_all_gene.set_index('Gene_name')['Gene_num'].to_dict()
        
        # Create reverse mapping from gene number to gene name
        num_to_gene_name = {v: k for k, v in gene_name_to_num.items()}

        # Get important gene numbers
        important_gene_nums = set()
        for gene_name in important_gene_names:
            if gene_name in gene_name_to_num:
                important_gene_nums.add(gene_name_to_num[gene_name])
        print("Important gene numbers:", important_gene_nums)
        return important_gene_nums, num_to_gene_name


    def analyze_paths(patient_type, top, important_gene_nums, df):

        # Calculate unique links for each 'From'
        unique_links = df.groupby('From')['To'].nunique()

        # Identify nodes with more than 2 unique links
        nodes_with_multiple_links = unique_links[unique_links > 2].index

        # Identify all potential paths through any middle node
        potential_starts = df[df['From'].isin(nodes_with_multiple_links)]
        potential_ends = df[df['To'].isin(nodes_with_multiple_links)]

        # Combine to find all possible paths through any middle nodes
        potential_paths = pd.merge(potential_starts, potential_ends, left_on='To', right_on='From', suffixes=('_start', '_end'))

        # Create a new DataFrame to display the path structure
        paths_df = potential_paths[['From_start', 'To_start', 'To_end']]
        paths_df.columns = ['From', 'Middle', 'To']

        # Calculate the mean attention for each path
        paths_df = paths_df.copy()
        paths_df['Attention_start'] = potential_paths['Attention_start']
        paths_df['Attention_end'] = potential_paths['Attention_end']
        paths_df['Mean Attention'] = (paths_df['Attention_start'] + paths_df['Attention_end']) / 2

        # Aggregate the data for identical paths to calculate mean values
        aggregated_paths = paths_df.groupby(['From', 'To', 'Middle']).agg({
            'Attention_start': 'mean',
            'Attention_end': 'mean',
            'Mean Attention': 'mean'
        }).reset_index()

        # Average the values for each path and its reverse path
        aggregated_paths['Key'] = aggregated_paths.apply(lambda x: tuple(sorted([x['From'], x['To']])), axis=1)
        grouped = aggregated_paths.groupby(['Key', 'Middle']).agg({
            'Attention_start': 'mean',
            'Attention_end': 'mean',
            'Mean Attention': 'mean'
        }).reset_index()

        # Merge back to get the full structure with averaged values
        final_paths = grouped.merge(aggregated_paths[['From', 'To', 'Middle', 'Key']], on=['Middle', 'Key'])
        final_paths = final_paths.drop(columns=['Key'])

        # Remove paths where From and To are the same
        final_paths = final_paths[final_paths['From'] != final_paths['To']]

        # Only keep paths where From < To to maintain a single direction per pair
        final_paths = final_paths[final_paths['From'] < final_paths['To']].drop_duplicates()

        # display(final_paths)
        
        # Sort paths by mean attention and select the top N paths
        def top_n_paths(group, top_n = 2):
            top_paths = group.sort_values(by='Mean Attention', ascending=False).head(top_n)
            return top_paths

        # Apply the filtering function to the final paths grouped by 'From' and 'To'
        filtered_top_n_paths = final_paths.groupby(['From', 'To']).apply(lambda x: top_n_paths(x)).reset_index(drop=True)

        # Record removable middle nodes, ensuring important nodes are not included and their 'To' is not important
        all_middles = set(final_paths['Middle'])
        important_middles = set(final_paths[final_paths['Middle'].isin(important_gene_nums)]['Middle'])
        removable_middles = set()

        # Nodes that will be kept:
        # 1. Hub Nodes (nodes_with_multiple_links)
        # 2. Top N paths sorted by mean attention (filtered_top_n_paths)
        # 3. P value middle nodes (important_middles)
        # 4. Middle nodes next to P value nodes
        
        for middle in all_middles - set(nodes_with_multiple_links) - set(filtered_top_n_paths['Middle']) - important_middles:
            next_nodes = final_paths[final_paths['From'] == middle]['To']
            if not any(node in important_gene_nums for node in next_nodes):
                removable_middles.add(middle)
        
        return removable_middles


    def remove_node(patient_type,df, important_gene_nums, num_to_gene_name):
        removable_middles = analyze_paths(patient_type, top, important_gene_nums,df)
        removable_gene_names = {num_to_gene_name[node] for node in removable_middles if node in num_to_gene_name}
        removable_gene_names_df = pd.DataFrame(removable_gene_names, columns=['Gene_name'])
        #removable_gene_names_df.to_csv(f'./ROSMAP-analysis/{patient_type}_middle_nodes_to_remove_{top}.csv', index=False)
        return removable_gene_names_df

    important_gene_nums, num_to_gene_name = load_gene_information(top)
    patient_type_one_remove = remove_node(patient_type_one,patient_type_one_ht_step1, important_gene_nums, num_to_gene_name)
    patient_type_two_remove = remove_node(patient_type_two,patient_type_two_ht_step1, important_gene_nums, num_to_gene_name)

    # Find the common rows
    common_rows = pd.merge(patient_type_one_remove, patient_type_two_remove, on='Gene_name')

    def clean_gene_connections(patient_type,df):
        # Load the gene number mapping data
        map_all_gene = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
        gene_name_to_num = map_all_gene.set_index('Gene_name')['Gene_num'].to_dict()

        # Load the list of genes to remove with header now included
        gene_list = common_rows['Gene_name'].tolist()

        # Convert names to numbers using dictionary comprehension
        gene_numbers = {gene_name_to_num[gene] for gene in gene_list if gene in gene_name_to_num}

        # df is the connection data
        # Filter out all connections involving the genes in your list
        filtered_df = df[~(df['From'].isin(gene_numbers) | df['To'].isin(gene_numbers))]

        # Save the cleaned data to a new CSV file
        output_file = f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type}_final_{top}.csv'
        filtered_df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

    clean_gene_connections(patient_type_one,patient_type_one_ht_step1)
    clean_gene_connections(patient_type_two,patient_type_two_ht_step1)






# 6. Plotting using R
# def plot(top,giant_comp_threshold,patient_type_one, patient_type_two):
#     # configure R_HOME and PATH
#     os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.4.0'
#     os.environ['PATH'] = os.environ['R_HOME'] + '\\bin\\x64;' + os.environ['PATH']
#     subprocess.run(['Rscript', 'Process_graph_momic.R', str(top), str(giant_comp_threshold), patient_type_one, patient_type_two], capture_output=True, text=True)

def plot(top, giant_comp_threshold, patient_type_one, patient_type_two):
    # configure R_HOME and PATH
    os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.4.0'
    os.environ['PATH'] = os.environ['R_HOME'] + '\\bin\\x64;' + os.environ['PATH']

    try:
        result = subprocess.run(
            ['Rscript', 'Process_graph_momic.R', str(top), str(giant_comp_threshold), patient_type_one, patient_type_two],
            capture_output=True, text=True, check=True
        )
        print("Output:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error:\n", e.stderr)

# 7. Plotting Bar
def plot_bar(top,patient_type_one,patient_type_two):
    def load_gene_information():
        map_all_gene = pd.read_csv('./ROSMAP-graph-data/map-all-gene.csv')
        gene_name_to_num = map_all_gene.set_index('Gene_name')['Gene_num'].to_dict()
        num_to_gene_name = {num: name for name, num in gene_name_to_num.items()}
        return gene_name_to_num, num_to_gene_name

    def process_file(filename, num_to_gene_name, gene_nums):
        df = pd.read_csv(filename)
        average_attention_list = []

        for num in gene_nums:
            if num in num_to_gene_name:
                relevant_rows = df[(df['From'] == num) | (df['To'] == num)]
                average_attention = relevant_rows['Attention'].mean()
                average_attention_list.append((num, average_attention))
        
        return average_attention_list

    def update_p_values_file(p_values_file, gene_name_to_num, num_to_gene_name,patient_type_one,patient_type_two,top):

        p_values_df = pd.read_csv(p_values_file)

        # Create a new column for modified Gene_name
        p_values_df['Modified_Gene_name'] = p_values_df['Gene_name'].apply(lambda x: re.sub(r'-.*$', '-PROT', x))

        # Map Modified_Gene_name to Gene_num
        p_values_df['Gene_num'] = p_values_df['Modified_Gene_name'].map(gene_name_to_num)

        # Drop rows where Gene_num is NaN (no mapping found)
        p_values_df = p_values_df.dropna(subset=['Gene_num'])

        gene_nums = p_values_df['Gene_num'].astype(int).tolist()
        
        # Process attention files for both patient types
        _, num_to_gene_name = load_gene_information()
        patient_type_one_attention = dict(process_file(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_one}.csv', num_to_gene_name, gene_nums))
        patient_type_two_attention = dict(process_file(f'./ROSMAP-analysis/avg_analysis/average_attention_{patient_type_two}.csv', num_to_gene_name, gene_nums))

        p_values_df[f'{patient_type_one}_Gene_weight'] = p_values_df['Gene_num'].apply(lambda x: patient_type_one_attention.get(x, 0))
        p_values_df[f'{patient_type_two}_Gene_weight'] = p_values_df['Gene_num'].apply(lambda x: patient_type_two_attention.get(x, 0))

        # Save to new CSV filea
        p_values_df.to_csv(f'./ROSMAP-analysis/p_values_{top}_for_plot.csv', index=False)

    gene_name_to_num, num_to_gene_name = load_gene_information()
    update_p_values_file(f'./ROSMAP-analysis/p_values_{top}.csv', gene_name_to_num, num_to_gene_name,patient_type_one,patient_type_two,top)


    #plot
    file_path = f'./ROSMAP-analysis/p_values_{top}'
    df = pd.read_csv(file_path + '_for_plot.csv')

    # Reverse the DataFrame and reset the index
    df = df.iloc[::-1].reset_index(drop=True)
    df['Index'] = range(len(df))

    superscript_map = {
        'gene-expression': '$^{1}$',
        'cnv_del': '$^{2}$',
        'cnv_dup': '$^{3}$',
        'cnv_mcnv': '$^{4}$',
        'methy-Downstream': '$^{5}$',
        'methy-Core-Promoter': '$^{6}$',
        'methy-Proximal-Promoter': '$^{7}$'
    }

    def clean_gene_name(name):
        return name.replace('-METH', '').replace('-TRAN', '')

    df['Gene Name With Superscript'] = df.apply(lambda row: clean_gene_name(row['Gene_name']) + superscript_map[row['Category']], axis=1)

    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(12, 18))

    ax.barh(df['Index'], -df['P_value'], color='lightsteelblue', label='P-Value', height=bar_width)

    # Plotting the gene weights for females and males with adjusted bar positions
    ax.barh(df['Index'] + bar_width/4, df[f'{patient_type_one}_Gene_weight'], color='deepskyblue', label=f'{patient_type_one} Gene Weight', height=bar_width/2)
    ax.barh(df['Index'] - bar_width/4, df[f'{patient_type_two}_Gene_weight'], color='hotpink', label=f'{patient_type_two} Gene Weight', height=bar_width/2)


    # Adding annotations for gene weights with font size
    for i, row in df.iterrows():
        ax.text(row[f'{patient_type_one}_Gene_weight'] + 0.02, i - bar_width/4 - 0.1, f"{row[f'{patient_type_one}_Gene_weight']:.4f}", va='center', ha='left', color='black', fontsize=6)
        ax.text(row[f'{patient_type_two}_Gene_weight'] + 0.02, i + bar_width/4 + 0.1, f"{row[f'{patient_type_two}_Gene_weight']:.4f}", va='center', ha='left', color='black', fontsize=6)

 
    # Setting y-ticks to show gene names with superscripts
    ax.set_yticks(df['Index'])
    ax.set_yticklabels(df['Gene Name With Superscript'])

    # Additional axes and grid setup
    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.axvline(x=-0.05, color='red', linestyle='--', linewidth=0.5)

    start, end = -0.25, 0.7
    xticks = np.arange(start + 0.05, end, 0.1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{abs(x):.2f}" if x != 0 else "0" for x in xticks])

    ax.set_xlim(start, end)

    # Add legend with superscript descriptions
    legend_labels = [f"{superscript_map[category]}: {category}" for category in superscript_map]
    legend_text = "\n".join(legend_labels)
    plt.text(0.77, 0.01, legend_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(facecolor='lightyellow', alpha=0.5), linespacing=0.8)

    ax.legend(loc='upper center')
    
    plt.savefig(f'./ROSMAP-plot/' + patient_type_one + '_top_' + str(top) + '_plot.png', bbox_inches='tight', dpi=600)

    # Horizontal Version
    fig, ax = plt.subplots(figsize=(20, 12))

    ax.bar(df['Index'], -df['P_value'], color='lightsteelblue', label='P-Value', width=bar_width)

    # Plotting the gene weights for females and males with adjusted bar positions
    ax.bar(df['Index'] - bar_width/4, df[f'{patient_type_one}_Gene_weight'], color='hotpink', label=f'{patient_type_one} Gene Weight', width=bar_width/2)
    ax.bar(df['Index'] + bar_width/4, df[f'{patient_type_two}_Gene_weight'], color='deepskyblue', label=f'{patient_type_two} Gene Weight', width=bar_width/2)

    # Adding annotations for gene weights with font size
    for i, row in df.iterrows():
        ax.text(i - bar_width/4 - 0.1, row[f'{patient_type_one}_Gene_weight'] + 0.01, f"{row[f'{patient_type_one}_Gene_weight']:.4f}", va='bottom', ha='center', color='black', fontsize=4, rotation=90)
        ax.text(i + bar_width/4 + 0.1, row[f'{patient_type_two}_Gene_weight'] + 0.01, f"{row[f'{patient_type_two}_Gene_weight']:.4f}", va='bottom', ha='center', color='black', fontsize=4, rotation=90)

    # Setting x-ticks to show gene names with superscripts
    ax.set_xticks(df['Index'])
    ax.set_xticklabels(df['Gene Name With Superscript'], rotation=90)

    # Additional axes and grid setup
    ax.axhline(y=0, color='gray', linewidth=0.8)
    ax.axhline(y=-0.05, color='red', linestyle='--', linewidth=0.5)

    start, end = -0.25, 0.7
    yticks = np.arange(start + 0.05, end, 0.1)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(x):.2f}" if x != 0 else "0" for x in yticks])

    ax.set_ylim(start, end)

    # Add legend with superscript descriptions
    legend_labels = [f"{superscript_map[category]}: {category}" for category in superscript_map]
    legend_text = "\n".join(legend_labels)
    plt.text(0.865, 0.985, legend_text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='lightyellow', alpha=0.5), linespacing=0.8)

    ax.legend(loc='upper center')

    plt.savefig(f'./ROSMAP-plot/' + patient_type_one + '_top_' + str(top) + '_plot_Horizontal.png', bbox_inches='tight', dpi=600)



###############Main workflow###############

# Define patient types

patient_type_one = 'AD'
patient_type_two = 'NOAD'

# patient_type_one = 'female'
# patient_type_two = 'male'

threshold = 0.12
top = 70
giant_comp_threshold = 8

# 1. Split Files Based on AD and calculate average attention
split_files_and_calculate_average_attention(patient_type_one, patient_type_two)

# 2. Generate the gene_name list based on the filtered average attention and the threshold
filter_gene_name(patient_type_one, threshold)
filter_gene_name(patient_type_two, threshold)

# 3. Calculate pvalue for each gene_name
calculate_pvalue(patient_type_one, patient_type_two)

# 4. Filter top pvalues
filter_top_pvalues(top)

# 5. Data filtering and processing
data_filtering_and_processing(patient_type_one, patient_type_two,top,threshold)

# 6. Plotting using R
plot(top,giant_comp_threshold,patient_type_one, patient_type_two)

# 7. Plotting Bar
plot_bar(top,patient_type_one,patient_type_two)