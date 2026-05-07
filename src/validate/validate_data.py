import great_expectations as ge
from great_expectations import expectations as E
import pandas as pd
from typing import Tuple, List
import re


def validate_data(data: pd.DataFrame) -> Tuple[bool, List, pd.DataFrame]:
    context = ge.get_context()

    # Prevent duplicate datasource creation
    # Data source
    try:
        data_source = context.data_sources.get("fraud_data")
    except LookupError:
        data_source = context.data_sources.add_pandas("fraud_data")

    # Data asset
    try:
        data_asset = data_source.get_asset("transactions")
    except LookupError:
        data_asset = data_source.add_dataframe_asset("transactions")

    batch_definition = data_asset.add_batch_definition_whole_dataframe("fraud_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": data})

    expectations = [
        # Required columns
        E.ExpectColumnToExist(column="Class"),
        E.ExpectColumnToExist(column="Amount"),

        # Null checks
        E.ExpectColumnValuesToNotBeNull(column="Class"),
        E.ExpectColumnValuesToNotBeNull(column="Amount"),

        # Valid target values
        E.ExpectColumnValuesToBeInSet(column="Class", value_set=[0, 1]),

        # Amount checks
        E.ExpectColumnValuesToBeBetween(column="Amount", min_value=0, max_value=100000),

        # Fraud rate sanity check (~0.17% expected)
        E.ExpectColumnMeanToBeBetween(
            column="Class",
            min_value=0.001,
            max_value=0.01
        ),
    ]

    # PCA feature checks (V1–V28)
    pca_columns = [
        col for col in data.columns
        if re.match(r"^V\d+$", col)
    ]

    if len(pca_columns) < 20:
        print(f"⚠️ Warning: Expected ~28 PCA columns, found {len(pca_columns)}")

    # Optional strict check (logs only)
    expected_pca = {f"V{i}" for i in range(1, 29)}
    missing_pca = expected_pca - set(pca_columns)
    if missing_pca:
        print(f" Missing PCA columns: {sorted(missing_pca)}")

    # Apply expectations dynamically to PCA columns
    for col in pca_columns:
        expectations.append(
            E.ExpectColumnValuesToNotBeNull(column=col)
        )

    # Run validations
    results = [batch.validate(exp) for exp in expectations]

    # Print each result
    all_passed = all(r["success"] for r in results)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['expectation_config']['type']}")
    print(f"\nValidation success: {all_passed}")

    # Collect failed expectations
    failed = [r for r in results if not r["success"]]

    required_cols = {"Class", "Amount"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(data.columns)}")

    # Filter out invalid rows manually
    valid_mask = (
            data["Class"].isin([0, 1]) &
            (data["Amount"] >= 0) &
            data["Amount"].notna() &
            data["Class"].notna()
    )

    cleaned_data = data.loc[valid_mask].copy()
    print(f"Removed {len(data) - len(cleaned_data)} invalid rows.")

    return all_passed, failed, cleaned_data




