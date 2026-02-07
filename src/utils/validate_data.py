import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    """
    print("🔍 Starting data validation...")
    
    # Track validation failures
    failed_expectations = []
    validation_passed = True
    checks_passed = 0
    total_checks = 0
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Required columns list
    required_cols = [
        "customerID", "gender", "Partner", "Dependents",
        "PhoneService", "InternetService", "Contract",
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    
    for col in required_cols:
        total_checks += 1
        if col not in df.columns:
            failed_expectations.append(f"Missing column: {col}")
            validation_passed = False
        else:
            checks_passed += 1
            # Check for null values in customer ID
            if col == "customerID":
                total_checks += 1
                if df[col].isnull().any():
                    failed_expectations.append(f"Null values in {col}")
                    validation_passed = False
                else:
                    checks_passed += 1
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Define valid values for categorical columns
    categorical_checks = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "InternetService": ["DSL", "Fiber optic", "No"]
    }
    
    for col, valid_values in categorical_checks.items():
        total_checks += 1
        if col in df.columns:
            invalid_count = (~df[col].isin(valid_values)).sum()
            if invalid_count > 0:
                failed_expectations.append(f"{col} has {invalid_count} invalid values")
                validation_passed = False
            else:
                checks_passed += 1
        else:
            checks_passed += 1  # Already caught in required columns check
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    numeric_checks = [
        ("tenure", 0, 120),
        ("MonthlyCharges", 0, 200),
        ("TotalCharges", 0, None)
    ]
    
    for col, min_val, max_val in numeric_checks:
        if col in df.columns:
            # Check min value
            total_checks += 1
            if (df[col] < min_val).any():
                failed_expectations.append(f"{col} has values below {min_val}")
                validation_passed = False
            else:
                checks_passed += 1
            
            # Check max value if specified
            if max_val is not None:
                total_checks += 1
                if (df[col] > max_val).any():
                    failed_expectations.append(f"{col} has values above {max_val}")
                    validation_passed = False
                else:
                    checks_passed += 1
    
    # === NULL VALUE CHECKS ===
    print("   🔍 Checking for missing values in critical columns...")
    
    critical_cols = ["tenure", "MonthlyCharges"]
    for col in critical_cols:
        total_checks += 1
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            failed_expectations.append(f"{col} has {null_count} null values")
            validation_passed = False
        else:
            checks_passed += 1
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # TotalCharges should generally be >= MonthlyCharges (95% threshold)
    total_checks += 1
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        inconsistent = (df["TotalCharges"] < df["MonthlyCharges"]).sum()
        inconsistent_pct = inconsistent / len(df)
        if inconsistent_pct > 0.05:  # More than 5% inconsistent
            failed_expectations.append(
                f"TotalCharges < MonthlyCharges in {inconsistent_pct*100:.1f}% of records"
            )
            validation_passed = False
        else:
            checks_passed += 1
    else:
        checks_passed += 1
    
    # === MINIMUM DATASET SIZE ===
    total_checks += 1
    if len(df) < 100:
        failed_expectations.append(f"Dataset too small: {len(df)} rows (minimum 100 required)")
        validation_passed = False
    else:
        checks_passed += 1
    
    # Print validation summary
    failed_checks = total_checks - checks_passed
    
    if validation_passed:
        print(f"✅ Data validation PASSED: {checks_passed}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations[:5]}")  # Show first 5 failures
    
    return validation_passed, failed_expectations
