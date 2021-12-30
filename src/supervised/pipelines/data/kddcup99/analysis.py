import pandas as pd
import sys
import io


def exploration(data: pd.DataFrame, targets: pd.DataFrame) -> str:
    """
    Explore Raw Data

    Args:
        data: The data.
        targets: The targets.
    """

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    report = ""
    # Data
    print("# Data")

    print("## Describe")
    print(data.describe().to_markdown() + "\n")
    print("\n\n")

    print("## Correlation")
    print(data.corr().to_markdown() + "\n")
    print("\n\n")

    print("## Coveriance")
    print(data.cov().to_markdown() + "\n")
    print("\n\n")   
    print("```")

    print("## Info")
    data.info()
    print("```")

    print("## NaNs")
    print(data.isna().sum().to_markdown())

    print("## Head")
    print(data.head().to_markdown() + "\n")
    print("\n\n")

    # Targets
    print("# Targets")

    print("## Describe")
    print(targets.describe().to_markdown() + "\n") # TODO why does to_markdown break?

    print("## Counts")
    print(targets.value_counts().to_markdown() + "\n")

    print("## NaNs")
    print(f"Total: {targets.isna().sum()}")
    #report += targets.corr().to_string() + "\n"
    #report += targets.cov().to_string() + "\n"
    #report += targets.info().to_string() + "\n"
    #report += targets.head().to_string() + "\n"
    #report += targets.tail().to_string() + "\n"
    
    report = new_stdout.getvalue()

    sys.stdout = old_stdout
    return report


