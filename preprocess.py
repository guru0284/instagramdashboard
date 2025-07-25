import os
import json
import pandas as pd
import re
import hashlib
import logging
from flatten_json import flatten
from dateutil import parser
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_FOLDER = 'instagram_jsons'  # Your Instagram JSON export folder root
OUTPUT_FOLDER = 'cleaned_csvs'       # Folder to save cleaned CSVs
REMOVE_EMOJIS = False                      # Set True to remove emojis and special chars from strings
GENERATE_UNIQUE_IDS = True                 # Auto-generate IDs if missing
DATE_FEATURES_EXTRACTION = True            # Add year/month/day etc. from date columns
LOG_FILE = 'processing.log'                # Log file for actions and errors

logging.basicConfig(filename=LOG_FILE,
                    filemode='w',
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# === UTILS ===
def remove_emojis_and_specials(text):
    # Removes emojis and control special characters, retaining normal text chars
    if not isinstance(text, str):
        return text
    try:
        # Pattern covers a wide emoji range
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Symbols & pictographs
            "\U0001F680-\U0001F6FF"  # Transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # Flags
            "\U00002500-\U00002BEF"  # Chinese chars + symbols
            "\U00002702-\U000027B0"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001f926-\U0001f937"
            "\U00010000-\U0010ffff"
            "\u2640-\u2642"
            "\u2600-\u2B55"
            "\u200d"
            "\u23cf"
            "\u23e9"
            "\u231a"
            "\ufe0f"  # Dingbats
            "\u3030"
            "]+", flags=re.UNICODE)
        cleaned = emoji_pattern.sub(r'', text)
        # Also strip other control or non-printable chars
        cleaned = re.sub(r'[\x00-\x1F\x7F]', '', cleaned)
        return cleaned.strip()
    except Exception:
        return text

def flatten_json(y, parent_key='', sep='_'):
    """
    Recursively flattens a nested JSON/dictionary.
    Nested keys get joined with 'sep'.
    For lists of dicts, each item gets an index in key.
    For lists of primitives, joins as comma-separated string.
    """
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], f"{name}{a}{sep}")
        elif isinstance(x, list):
            if all(isinstance(i, dict) for i in x):
                for idx, a in enumerate(x):
                    flatten(a, f"{name}{idx}{sep}")
            else:
                out[name[:-1]] = ', '.join([str(i) for i in x])
        else:
            out[name[:-1]] = x
    flatten(y, parent_key)
    return out

def json_file_to_flat_records(json_path):
    """
    Loads a JSON file and returns a list of flat dicts—one per top-level record.
    Handles varied Instagram JSON structures.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Detect the outer structure:
    if isinstance(data, dict):
        # Try common keys with list values
        for try_key in ['list', 'messages', 'media', 'connections', 'string_list_data', 'profile_changes', 'followers', 'following', 'ads', 'ads_information', 'items']:
            if try_key in data and isinstance(data[try_key], list):
                data = data[try_key]
                break
        else:  # no known key; wrap dict as list
            data = [data]
    elif isinstance(data, list):
        pass  # already a list
    else:
        data = [data]

    flat_records = []
    for record in data:
        flat_records.append(flatten_json(record))
    return flat_records

def clean_and_standardize(df):
    # Remove duplicates & fully empty cols
    df = df.drop_duplicates()
    df = df.dropna(axis=1, how='all')
    df = df.fillna('')

    # Clean column names (strip, replace spaces/dots/hyphens with underscore, lowercase)
    df.columns = [c.strip().replace(' ', '_').replace('.', '_').replace('-', '_').lower() for c in df.columns]

    # Optional: Remove emojis/special chars in all string columns
    if REMOVE_EMOJIS:
        str_cols = df.select_dtypes(include=['object']).columns
        for c in str_cols:
            df[c] = df[c].apply(remove_emojis_and_specials)

    # Attempt to parse dates; unify to UTC ISO format
    date_cols = [col for col in df.columns if any(x in col for x in ['date', 'time', 'timestamp'])]
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
        except Exception:
            pass  # ignore parse errors

    # Extract date parts for easier BI grouping (optional)
    if DATE_FEATURES_EXTRACTION:
        for col in date_cols:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_weekday'] = df[col].dt.weekday
                df[f'{col}_hour'] = df[col].dt.hour

    # Numeric type enforcement: try to cast obvious count/metric fields
    metric_like_cols = [c for c in df.columns if any(k in c for k in ['count', 'likes', 'comments', 'engagement', 'views', 'followers', 'following'])]
    for col in metric_like_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        except Exception:
            continue

    # Map certain coded values (example: media_type)
    if 'media_type' in df.columns:
        media_map = {1: 'photo', 2: 'video', 8: 'carousel'}
        df['media_type'] = df['media_type'].map(media_map).fillna(df['media_type'])

    # Trim whitespace from strings in object columns
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    return df

def create_unique_id(row, keys=[]):
    """
    Create a hash-based unique ID from row values of specified keys.
    If keys empty, use all columns.
    """
    if keys:
        key_str = '||'.join(str(row[k]) for k in keys if k in row and pd.notna(row[k]))
    else:
        key_str = '||'.join(str(v) for v in row)
    return hashlib.md5(key_str.encode('utf-8')).hexdigest()

def extract_one_to_many_tables(df, original_prefix, key_name='id'):
    """
    For columns with comma-separated lists (likely lists flattened),
    split them into separate tables with linkage keys.
    E.g., hashtags, tagged_users, locations.
    Returns dict {table_name: dataframe}
    """
    tables = {}
    # Candidates for splitting: columns where values contain commas or index-like suffixes
    # We'll pick columns ending with '_list', or columns containing ','

    # Try detecting hashtags or similar list fields
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or df[col].dtype == object:
            if df[col].str.contains(',').any():
                # Split comma-separated values to rows with link to main table
                records = []
                for idx, val in df[[key_name, col]].dropna().iterrows():
                    main_id = val[key_name]
                    if isinstance(val[col], str):
                        items = [x.strip() for x in val[col].split(',')]
                        for i, item in enumerate(items):
                            if item:
                                records.append({key_name: main_id, f'{col}_item': item, 'index': i})

                if records:
                    table_df = pd.DataFrame(records)
                    table_name = f"{original_prefix}_{col}_table"
                    tables[table_name] = table_df

                    # Clean main df column since split out
                    df[col] = df[col].apply(lambda s: '' if isinstance(s, str) else s)
    return tables

def process_all_jsons(input_root, output_root):
    summary = {'processed_files': [], 'errors': [], 'empty_files': [], 'tables_created': []}

    for folder, _, files in os.walk(input_root):
        for filename in files:
            if not filename.lower().endswith('.json'):
                continue
            json_path = os.path.join(folder, filename)
            rel_folder = os.path.relpath(folder, input_root)
            out_dir = os.path.join(output_root, rel_folder)
            os.makedirs(out_dir, exist_ok=True)
            csv_out_base = filename.replace('.json', '_cleaned.csv')
            csv_out = os.path.join(out_dir, csv_out_base)

            try:
                records = json_file_to_flat_records(json_path)
                if not records:
                    logging.warning(f"Skipped empty file: {json_path}")
                    summary['empty_files'].append(json_path)
                    continue

                df = pd.DataFrame(records)
                if df.empty:
                    logging.warning(f"Empty dataframe after loading records: {json_path}")
                    summary['empty_files'].append(json_path)
                    continue

                # Generate unique ids if missing; prefer keys named *_id else generate
                id_cols = [c for c in df.columns if c.endswith('_id')]
                if GENERATE_UNIQUE_IDS and not id_cols:
                    df['unique_id'] = [create_unique_id(row) for i, row in df.iterrows()]
                    id_cols = ['unique_id']

                df = clean_and_standardize(df)

                # Extract one-to-many relationship tables (e.g., hashtags)
                aux_tables = {}
                if id_cols:
                    key_col = id_cols[0]
                    aux_tables = extract_one_to_many_tables(df, original_prefix=os.path.splitext(filename)[0], key_name=key_col)

                # Save main cleaned table
                df.to_csv(csv_out, index=False)
                summary['processed_files'].append(csv_out)
                logging.info(f"Processed and saved: {csv_out}")

                # Save auxiliary tables, if any
                for tbl_name, tbl_df in aux_tables.items():
                    aux_path = os.path.join(out_dir, f"{tbl_name}.csv")
                    tbl_df.to_csv(aux_path, index=False)
                    summary['tables_created'].append(aux_path)
                    logging.info(f"Created relational table: {aux_path}")

            except Exception as e:
                logging.error(f"Error processing {json_path}: {e}")
                summary['errors'].append((json_path, str(e)))

    # Summary report
    print("Processing complete.")
    print(f"Files processed: {len(summary['processed_files'])}")
    print(f"Relational tables created: {len(summary['tables_created'])}")
    print(f"Empty files skipped: {len(summary['empty_files'])}")
    print(f"Errors: {len(summary['errors'])}")
    logging.info("SUMMARY:")
    logging.info(f"Files processed: {len(summary['processed_files'])}")
    logging.info(f"Relational tables: {len(summary['tables_created'])}")
    logging.info(f"Empty files: {len(summary['empty_files'])}")
    logging.info(f"Errors: {len(summary['errors'])}")

if __name__ == '__main__':
    print("Starting Instagram JSON cleaning and structuring process...")
    process_all_jsons(ROOT_FOLDER, OUTPUT_FOLDER)
    print(f"All done! Check folder '{OUTPUT_FOLDER}' and log file '{LOG_FILE}' for details.")
