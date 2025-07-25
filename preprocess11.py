import os
import json
import pandas as pd

def flatten_json(y, parent_key='', sep='_'):
    """
    Recursively flattens a nested JSON/dictionary.
    Nested keys get joined with 'sep'.
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

    # Try to determine the outer structure:
    if isinstance(data, dict):
        # Sometimes the list of records is under a specific key (e.g., 'list', 'messages', etc.)
        for try_key in ['list', 'messages', 'media', 'connections', 'string_list_data', 'profile_changes', 'followers', 'following']:
            if try_key in data and isinstance(data[try_key], list):
                data = data[try_key]
                break
        else:
            data = [data]
    elif isinstance(data, list):
        pass  # it's fine
    else:
        data = [data]

    # Flatten each record
    flat_records = []
    for record in data:
        flat_records.append(flatten_json(record))
    return flat_records

def clean_and_standardize(df):
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Remove columns with all empty values, optional:
    df = df.dropna(axis=1, how='all')
    # Fill missing values (adjust as preferred)
    df = df.fillna('')
    # Attempt to parse dates
    for col in df.columns:
        if any(kw in col.lower() for kw in ['date', 'time', 'timestamp']):
            try:
                df[col] = pd.to_datetime(df[col], errors='ignore')
            except Exception:
                pass
    # Clean column names
    df.columns = [c.strip().replace(' ', '_').replace('.', '_').replace('-', '_') for c in df.columns]
    return df

def process_all_jsons(input_root, output_root):
    for folder, _, files in os.walk(input_root):
        for filename in files:
            if not filename.lower().endswith('.json'):
                continue
            json_path = os.path.join(folder, filename)
            rel_folder = os.path.relpath(folder, input_root)
            out_dir = os.path.join(output_root, rel_folder)
            os.makedirs(out_dir, exist_ok=True)
            csv_out = os.path.join(out_dir, filename.replace('.json', '_cleaned.csv'))
            try:
                records = json_file_to_flat_records(json_path)
                if not records:
                    print(f"Skipped empty: {json_path}")
                    continue
                df = pd.DataFrame(records)
                df = clean_and_standardize(df)
                df.to_csv(csv_out, index=False)
                print(f"Processed: {csv_out}")
            except Exception as e:
                print(f"Error processing {json_path}: {e}")

if __name__ == '__main__':
    input_dir = './instagram_jsons'
    output_dir = './cleaned_csvs'
    process_all_jsons(input_dir, output_dir)
    print("All files processed.")


