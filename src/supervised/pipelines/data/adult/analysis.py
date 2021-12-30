import pandas as pd
import sys
import io


def exploration(data: pd.DataFrame) -> str:
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    print(
        f"""
            ## Adult Data Exploration

            ## Correlation
            {data.corr().to_markdown()}

            ## Info
            ```
            {data.info()}
            ```

            ## Descriptive
            {data.describe().to_markdown()}
    
        """

    report = new_stdout.getvalue()
    sys.stdout = old_stdout
    return report