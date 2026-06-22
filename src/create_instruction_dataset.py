import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



INSTRUCTION =(
     "Classify the telecom fault,identify the root cause,"
     "determine severity,and recommend resolution steps."
    )



def create_input_text(row):
    return (
        f"Network Element: {row['network_element']}\n"
        f"Region: {row['region']}\n"
        f"Alarm Type: {row['alarm_type']}\n"
        f"KPI Summary: {row['kpi_summary']}\n"
        f"Raw Log: {row['raw_log']}"
    )


def create_output_text(row):
    return (
        f"Fault Category: {row['fault_category']}\n"
        f"Root Cause: {row['root_cause']}\n"
        f"Severity: {row['severity']}\n"
        f"Resolution Steps: {row['resolution_steps']}"
    )





def dataframe_to_jsonl(df, output_file):
    
    with  open(output_file,"w",encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {
                "instruction": INSTRUCTION,
                "input": create_input_text(row),
                "output": create_output_text(row)
            }
            f.write(json.dumps(record) + "\n")
def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    input_csv = BASE_DIR / "data/processed/telecom_fault_dataset.csv"
    

    output_dir = Path(BASE_DIR /"data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    print(f"Total Rows: {len(df)}")

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    test_df, eval_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    eval_df = eval_df.head(500)

    dataframe_to_jsonl(
        train_df,
        output_dir / "train.jsonl"
    )

    dataframe_to_jsonl(
        test_df,
        output_dir / "test.jsonl"
    )

    dataframe_to_jsonl(
        eval_df,
        output_dir / "eval_500.jsonl"
    )

    print("\nCreated Files:")
    print(f"Train: {len(train_df)} rows")
    print(f"Test: {len(test_df)} rows")
    print(f"Eval: {len(eval_df)} rows")

    print("\nSaved to:")
    print("data/processed/train.jsonl")
    print("data/processed/test.jsonl")
    print("data/processed/eval_500.jsonl")


if __name__ == "__main__":
    main()
