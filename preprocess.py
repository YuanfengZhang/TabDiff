# TabDiff preprocess
# 1. Drop rows where the targe (Y) is NA.
# 2. Merge one-hot encoded columns to single column and remove original one-hot columns.
# 3. Fill the NA values in categorical columns with 'empty'.
# 4. Drop rows where one of the numerical columns is NA.
# 5. Split the data into categorical_df and numerical_df.
# 6. Divide every df into train_df (45%), valid_df (45%), and test_df (10%),
# and save them as .npy files using numpy.save.
from argparse import ArgumentParser, ArgumentTypeError, Namespace, RawDescriptionHelpFormatter
from json import dump as json_dump
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


DESCRIPTION: str = (
    'This script is meant to preprocess the dataset for TabDiff.\n'
    '1. Rows where the target (Y) column is NA will be dropped.\n'
    '2. One-hot encoded columns will be merged into a single column and removed if specified.\n'
    '3. NA values in categorical columns will be filled with "empty".\n'
    '4. Rows where any numerical column is NA will be dropped.\n'
    '5. The data will be split into categorical and numerical DataFrames.\n'
    '6. Each DataFrame will be divided into train , validation, and test datasets, and saved as .npy files.'
    'Use this script like this:\n'
    'python tabdiff_preprocess.py \n'
    '  --name <name to label this dataset>\n'
    '  --input <path to input file>\n'
    '  --format <format of the input file, [csv/tsv/parquet]>\n'
    '  --output-dir <directory to save the output files>\n'
    '  --target-column <the target column to predict>\n'
    '  --rm <columns to delete at the very first>\n'
    '       this param shall be given only once as a list with ,\n'
    '  --cat-merge <merged_cat_col_name:original_col1,original_col2,...>\n'
    '              This param can be specified multiple times\n'
    '  --remove-original-cat-col specify if the original categorical columns should be removed after merging\n'
    '  --split-props <train_ratio,valid_ratio,test_ratio>\n'
    '                three int / float values with a sum of 100.0, such as "45,45,10"\n')


def check_split_props(string: str) -> list[float]:
    """
    make sure the split-props is a string of three float values separated by commas,
    and that they sum to 100.0.
    """
    parts = list(map(float, string.split(',')))
    if len(parts) != 2:
        raise ArgumentTypeError('split-props must contain three values.')
    if not (abs(sum(parts) - 100.0) < 1e-6):
        raise ArgumentTypeError('split-props must sum to 100.')
    return [p / 100.0 for p in parts]


def parse_cat_merge_list(cat_merge_list: list[str]) -> dict[str, list[str]]:
    cat_merge_cols: dict[str, list[str]] = {}
    for item in cat_merge_list:
        parts: list[str] = item.split(':')
        if len(parts) != 2:
            raise ArgumentTypeError(f'Invalid format for cat-merge: {item}. '
                                    'Expected format: merged_cat_col_name:original_col1,original_col2,...')
        merged_col_name = parts[0]
        original_cols = parts[1].split(',')
        cat_merge_cols[merged_col_name] = original_cols
    return cat_merge_cols


# It's much safer and easy to read to write a function to get column info here
def get_column_info(df: pd.DataFrame,
                    numeric_cols: list[str],
                    categorical_cols: list[str]):
    column_list: list[str] = df.columns.to_list()
    column_info = {}
    metadata = {}
    for idx, col in enumerate(column_list):
        if col in numeric_cols:
            column_info[str(idx)] = {
                'type': 'numerical',
                'max': float(df[col].max()),
                'min': float(df[col].min())}
            metadata[str(idx)] = {
                'sdtype': 'numerical', 'computer_representation': 'Float'}
        elif col in categorical_cols:
            column_info[str(idx)] = {
                'categories': df[col].unique().tolist()}
            metadata[str(idx)] = {
                'sdtype': 'categorical'}
        else:  # target_col
            column_info[str(idx)] = {
                'type': 'numerical',
                'max': float(df[col].max()),
                'min': float(df[col].min())}
            metadata[str(idx)] = {
                'sdtype': 'numerical', 'computer_representation': 'Float'}
    return column_info, metadata


# The funtion to get columns mapping is refined with column names
# to avoid meaningless and confusing index manipulation.
def get_column_name_mapping(column_list: list[str],
                            numeric_cols: list[str],
                            categorical_cols: [list[str]],
                            target_col: str):
    name_to_idx = {name: idx for idx, name in enumerate(column_list)}

    # 获取各组列的索引
    num_col_idx = [name_to_idx[col] for col in numeric_cols]
    cat_col_idx = [name_to_idx[col] for col in categorical_cols]
    target_col_idx = [name_to_idx[target_col]]

    # 创建新索引的起始位置
    starts = {
        'num': 0,
        'cat': len(num_col_idx),
        'target': len(num_col_idx) + len(cat_col_idx)
    }

    # 构建索引映射
    idx_mapping = {}
    for col_idx in range(len(column_list)):
        if col_idx in num_col_idx:
            new_idx = starts['num']
            starts['num'] += 1
        elif col_idx in cat_col_idx:
            new_idx = starts['cat']
            starts['cat'] += 1
        else:  # 目标列
            new_idx = starts['target']
            starts['target'] += 1
        idx_mapping[int(col_idx)] = int(new_idx)

    # 构建反向映射
    inverse_idx_mapping = {v: k for k, v in idx_mapping.items()}

    # 构建索引到列名的映射
    idx_name_mapping = {int(i): name for i, name in enumerate(column_list)}

    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    arg_parser: ArgumentParser = ArgumentParser(description=DESCRIPTION,
                                                formatter_class=RawDescriptionHelpFormatter)
    arg_parser.add_argument('-n', '--name', dest='run_name', required=True,
                            help='run name, use only [0-9a-zA-Z_-]')
    arg_parser.add_argument('-i', '--input', type=str, dest='input', required=True,
                            help='Path to the input file.')
    arg_parser.add_argument('-f', '--format', type=str, dest='format', choices=['csv', 'tsv', 'parquet'],
                            required=True, help=('Format of the input file. '
                                                 'To avoid bugs, auto-detect is not implemented.'))

    arg_parser.add_argument('-o', '--output-dir', type=str, dest='output_dir', required=True,
                            help='Directory to save the output files.')

    arg_parser.add_argument('-t', '--target-column', type=str, dest='target_col', required=True,
                            help='Name of the target column (Y) which will be predicted.')
    arg_parser.add_argument('-r', '--rm', type=str, dest='columns_to_remove',
                            default='', help='columns to remove. E.g.: a,b,c')
    arg_parser.add_argument('-m', '--cat-merge', action='append', dest='cat_merge_list', default=[],
                            help='Merge one-hot encoded columns into a single categorical column. '
                                 'Format: merged_cat_col_name:original_col1,original_col2,... '
                                 'This parameter can be specified multiple times.')

    arg_parser.add_argument('--remove-original-cat-col', dest='rm_cat', action='store_true',
                            help='Remove original categorical columns after merging.')

    arg_parser.add_argument('--split-props', type=check_split_props, default=[.8, .2],
                            help='Train, and test split proportions, like "80,20" Three numbers summing to 100.')

    args: Namespace = arg_parser.parse_args()

    logger.info('Parsing parameters')
    print('Run NAME:', args.run_name)
    print('Input File:', args.input)
    print('File Format:', args.format)
    print('Output Directory:', args.output_dir)
    print('Target Column:', args.target_col)
    if args.columns_to_remove:
        print('Columns to Remove:', args.columns_to_remove)
    else:
        print('No columns to remove directly')
    print('Cat Merge:', args.cat_merge_list)
    print('Remove Original Cat Col:', args.rm_cat)
    print('Split Props:', args.split_props)

    # check if input file exists and read if yes.
    input_file: Path = Path(args.input)
    if not input_file.exists():
        raise FileNotFoundError(f'Input file {input_file} does not exist.')

    output_dir: Path = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info('Output Directory created: %s', output_dir)

    cat_merge_columns: dict[str, list[str]] = parse_cat_merge_list(args.cat_merge_list)

    input_df: pd.DataFrame
    if args.format == 'csv':
        input_df = pd.read_csv(input_file)
    elif args.format == 'tsv':
        input_df = pd.read_table(input_file)
    elif args.format == 'parquet':
        input_df = pd.read_parquet(input_file)
    else:
        raise ValueError(f'Unsupported file format: {args.format}')
    logger.info(f'Input DataFrame read with shape: {input_df.shape}')

    print(f'input_df columns:\n{input_df.columns.to_list()}')
    if args.columns_to_remove:
        for col in args.columns_to_remove.split(','):
            if col in input_df.columns:
                input_df.drop(columns=[col], inplace=True)
            logger.info(f'Column {col} deleted')

    for col in [c for c_ls in cat_merge_columns.values() for c in c_ls] + [args.target_col]:
        if col not in input_df.columns:
            raise ValueError(f'Column {col} specified for merging does not exist in the input DataFrame.')

    input_df.dropna(subset=[args.target_col], inplace=True)

    # merge categorical columns as specified.
    if cat_merge_columns:
        for merged_col, original_cols in cat_merge_columns.items():
            input_df[merged_col] = input_df[original_cols].astype('int').to_numpy().argmax(axis=1)
            if args.rm_cat:
                input_df.drop(columns=original_cols, inplace=True)
            logger.info(f'Merged columns {original_cols} into {merged_col}')

    # 确认所有列的数据类型, 仅有2个独特值的数值列也被视为分类列
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for col in [c for c in input_df.columns if c != args.target_col]:
        if pd.api.types.is_numeric_dtype(input_df[col]):
            if input_df[col].nunique() > 2:
                numeric_cols.append(col)
            else:
                # transform to 0 and 1 and treat it as categorical column
                input_df[col] = input_df[col].map({
                    original_value: str(index)
                    for index, original_value in enumerate(np.sort(input_df[col].unique()))})
                categorical_cols.append(col)
        elif pd.api.types.is_boolean_dtype(input_df[col]):
            input_df[col] = input_df[col].astype(int)
            categorical_cols.append(col)
        else:
            categorical_cols.append(col)

    # sort the order of columns: numeric features -> categorical features -> target
    input_df = input_df[numeric_cols + categorical_cols + [args.target_col]]
    final_columns: list[str] = input_df.columns.to_list()

    # get integer columns
    integer_cols = []
    int_col_idx_wrt_num = []
    for i, col_name in enumerate(numeric_cols):
        if pd.api.types.is_integer_dtype(input_df[col_name]):
            integer_cols.append(col_name)
            int_col_idx_wrt_num.append(i)

    # 填充分类列的空值
    for col in categorical_cols:
        input_df[col].fillna('empty', inplace=True)
    # 删除数值列为空的行
    input_df.dropna(subset=numeric_cols, inplace=True)
    input_df.reset_index(drop=True, inplace=True)

    # 将数据集分割成训练集、验证集和测试集
    logger.info('Splitting the DataFrame into train, validation, and test sets.')

    X_df: pd.DataFrame
    y_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.DataFrame
    y_test: pd.DataFrame

    y_df = input_df[[args.target_col]].copy(deep=True)
    # X_df = input_df.drop(columns=[args.target_col])
    (X_train, X_test,
     y_train, y_test) = train_test_split(input_df, y_df, random_state=28,
                                         test_size=args.split_props[1])
    # 分离分类列和数值列
    np.save(output_dir / 'X_cat_train.npy', X_train[categorical_cols].to_numpy())
    np.save(output_dir / 'X_num_train.npy',
            X_train[numeric_cols].to_numpy().astype(np.float32))
    np.save(output_dir / 'X_cat_test.npy', X_test[categorical_cols].to_numpy())
    np.save(output_dir / 'X_num_test.npy',
            X_test[numeric_cols].to_numpy().astype(np.float32))
    np.save(output_dir / 'y_train.npy', y_train)
    np.save(output_dir / 'y_test.npy', y_test)

    input_df.to_csv(output_dir / f'{args.run_name}.csv', index=False)
    X_train.to_csv(output_dir / 'train.csv', index=False)
    X_test.to_csv(output_dir / 'test.csv', index=False)

    # TODO: generate info.json
    final_columns: list[str] = input_df.columns.to_list()
    (column_info,
     metadata) = get_column_info(df=input_df,
                                 numeric_cols=numeric_cols,
                                 categorical_cols=categorical_cols)
    (idx_mapping,
     rev_idx_mapping,
     idx_name_mapping) = get_column_name_mapping(column_list=final_columns,
                                                 numeric_cols=numeric_cols,
                                                 categorical_cols=categorical_cols,
                                                 target_col=args.target_col)
    info = {
        'name': args.run_name,
        'task_type': 'regression',
        'header': 'infer',
        'column_names': final_columns,
        'num_col_idx': [final_columns.index(c) for c in numeric_cols
                        if c in final_columns],
        'cat_col_idx': [final_columns.index(c) for c in categorical_cols
                        if c in final_columns],
        'target_col_idx': [final_columns.index(args.target_col)],
        'file_type': 'csv',
        'test_path': 'null',
        'val_path': 'null',
        'data_path': (output_dir / f'{args.run_name}.csv').as_posix(),
        'int_col_idx': [final_columns.index(c) for c in integer_cols
                       if c in final_columns],
        'int_columns': integer_cols,
        'int_col_idx_wrt_num': int_col_idx_wrt_num,
        'column_info': column_info,
        'train_num': y_train.shape[0],
        'test_num': y_test.shape[0],
        'val_num': 0,
        'idx_mapping': idx_mapping,
        'inverse_idx_mapping': rev_idx_mapping,
        'idx_name_mapping': idx_name_mapping,
        'metadata': metadata}
    with open(output_dir / 'info.json', 'w+') as jf:
        json_dump(info, jf, indent=4)

    logger.info(f'Finally we done preprocessing for dataset {args.run_name}')


if __name__ == "__main__":
    main()

