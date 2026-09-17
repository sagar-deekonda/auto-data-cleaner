from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np

import os
import uuid

from openai import OpenAI


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "cleaned_files"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx",
    "xls"
}


def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# OPENAI CLIENT
# ============================================================

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    client = OpenAI(api_key=api_key)
else:
    client = None


# ============================================================
# LOAD DATA
# ============================================================

def load_data(filepath):

    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "csv":

        return pd.read_csv(filepath)

    elif extension == "xlsx":

        return pd.read_excel(filepath)

    elif extension == "xls":

        return pd.read_excel(filepath)

    else:

        raise ValueError(
            "Unsupported file format."
        )


# ============================================================
# DATA CLEANING
# ============================================================

def clean_data(df):

    # --------------------------------------------------------
    # ORIGINAL SIZE
    # --------------------------------------------------------

    rows_before = len(df)

    columns_before = len(df.columns)


    # --------------------------------------------------------
    # CLEAN COLUMN NAMES FIRST
    # --------------------------------------------------------

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


    # --------------------------------------------------------
    # REPLACE COMMON EMPTY VALUES WITH NaN
    # --------------------------------------------------------

    missing_words = [
        "",
        " ",
        "na",
        "n/a",
        "null",
        "none",
        "unknown",
        "-"
    ]

    df = df.replace(
        missing_words,
        np.nan
    )


    # --------------------------------------------------------
    # DUPLICATES
    # --------------------------------------------------------

    duplicates_removed = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()


    # --------------------------------------------------------
    # NUMERIC TYPE CONVERSION
    # --------------------------------------------------------

    numeric_columns_converted = 0

    for col in df.columns:

        # Skip boolean columns
        if pd.api.types.is_bool_dtype(df[col]):
            continue

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        success_ratio = (
            converted.notna().mean()
        )

        if success_ratio > 0.90:

            if not pd.api.types.is_numeric_dtype(
                df[col]
            ):

                numeric_columns_converted += 1

            df[col] = converted


    # --------------------------------------------------------
    # DATETIME TYPE CONVERSION
    # --------------------------------------------------------

    date_columns_converted = 0

    for col in df.columns:

        # Don't try date conversion on numbers
        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            continue

        # Don't try date conversion on booleans
        if pd.api.types.is_bool_dtype(
            df[col]
        ):
            continue

        converted = pd.to_datetime(
            df[col],
            errors="coerce"
        )

        success_ratio = (
            converted.notna().mean()
        )

        if success_ratio > 0.90:

            if not pd.api.types.is_datetime64_any_dtype(
                df[col]
            ):

                date_columns_converted += 1

                df[col] = converted


    total_data_types_converted = (
        numeric_columns_converted
        + date_columns_converted
    )


    # --------------------------------------------------------
    # IDENTIFY COLUMNS
    # --------------------------------------------------------

    num_cols = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    cat_cols = (
        df.select_dtypes(
            include="object"
        )
        .columns
        .tolist()
    )


    # --------------------------------------------------------
    # MISSING VALUES BEFORE CLEANING
    # --------------------------------------------------------

    missing_before = int(
        df.isna().sum().sum()
    )


    # --------------------------------------------------------
    # NUMERIC MISSING VALUES
    # --------------------------------------------------------

    for num_col in num_cols:

        best_cat = None
        best_score = 0

        for cat_col in cat_cols:

            unique_ratio = (
                df[cat_col].nunique()
                / max(len(df), 1)
            )

            # Avoid high-cardinality columns
            if unique_ratio > 0.01:
                continue

            grouped = (
                df.groupby(cat_col)[num_col]
                .count()
            )

            if len(grouped) == 0:
                continue

            score = grouped.mean()

            if score > best_score:

                best_score = score

                best_cat = cat_col


        # Group-based median
        if best_cat is not None:

            df[num_col] = (
                df[num_col]
                .fillna(
                    df.groupby(best_cat)[num_col]
                    .transform("median")
                )
            )


        # Overall median fallback
        df[num_col] = (
            df[num_col]
            .fillna(
                df[num_col].median()
            )
        )


    # --------------------------------------------------------
    # CATEGORICAL MISSING VALUES
    # --------------------------------------------------------

    cat_cols = (
        df.select_dtypes(
            include="object"
        )
        .columns
        .tolist()
    )

    if len(cat_cols) > 0:

        df[cat_cols] = (
            df[cat_cols]
            .fillna("Unavailable")
        )


    # --------------------------------------------------------
    # TEXT CLEANING
    # --------------------------------------------------------

    cat_cols = (
        df.select_dtypes(
            include="object"
        )
        .columns
        .tolist()
    )

    for col in cat_cols:

        # Remove spaces
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

        unique_ratio = (
            df[col].nunique()
            / max(len(df), 1)
        )

        # Lowercase relatively low-cardinality
        # categorical columns
        if unique_ratio < 0.50:

            df[col] = (
                df[col]
                .str.lower()
            )


    # --------------------------------------------------------
    # OUTLIER DETECTION
    # --------------------------------------------------------
    # Important:
    # We DETECT outliers but do not automatically delete them.
    # --------------------------------------------------------

    num_cols = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    outliers_detected = 0

    for num_col in num_cols:

        series = df[num_col].dropna()

        if series.empty:
            continue

        Q1 = series.quantile(0.25)

        Q3 = series.quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR

        upper = Q3 + 1.5 * IQR

        outliers = df[
            (df[num_col] < lower)
            |
            (df[num_col] > upper)
        ]

        outliers_detected += len(
            outliers
        )


    # --------------------------------------------------------
    # MISSING VALUES AFTER CLEANING
    # --------------------------------------------------------

    missing_after = int(
        df.isna().sum().sum()
    )

    missing_values_handled = (
        missing_before
        - missing_after
    )


    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    report = {

        "duplicates": duplicates_removed,

        "missing_values":
            missing_values_handled,

        "data_types":
            total_data_types_converted,

        "outliers":
            int(outliers_detected),

        "rows_before":
            rows_before,

        "rows_after":
            len(df),

        "columns_before":
            columns_before,

        "columns_after":
            len(df.columns)
    }


    return df, report


# ============================================================
# CREATE AI DATA SUMMARY
# ============================================================

def create_ai_summary(df, report):

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    summary = {

        "rows": int(len(df)),

        "columns": int(len(df.columns)),

        "column_names":
            df.columns.tolist(),

        "numeric_columns":
            df.select_dtypes(
                include=np.number
            ).columns.tolist(),

        "categorical_columns":
            df.select_dtypes(
                include="object"
            ).columns.tolist(),

        "datetime_columns":
            df.select_dtypes(
                include=["datetime"]
            ).columns.tolist()
    }


    # --------------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------------

    numeric_summary = {}

    numeric_df = df.select_dtypes(
        include=np.number
    )


    # Limit to first 20 numeric columns
    for col in numeric_df.columns[:20]:

        series = numeric_df[col]

        numeric_summary[col] = {

            "count":
                int(series.count()),

            "mean":
                round(
                    float(series.mean()),
                    2
                ),

            "median":
                round(
                    float(series.median()),
                    2
                ),

            "min":
                round(
                    float(series.min()),
                    2
                ),

            "max":
                round(
                    float(series.max()),
                    2
                ),

            "sum":
                round(
                    float(series.sum()),
                    2
                )
        }


    # --------------------------------------------------------
    # CATEGORICAL SUMMARY
    # --------------------------------------------------------

    categorical_summary = {}

    categorical_df = df.select_dtypes(
        include="object"
    )


    # Limit number of columns
    for col in categorical_df.columns[:20]:

        values = (
            categorical_df[col]
            .value_counts()
            .head(8)
        )

        categorical_summary[col] = {
            str(key): int(value)
            for key, value in values.items()
        }


    # --------------------------------------------------------
    # MISSING VALUES BY COLUMN
    # --------------------------------------------------------

    missing_by_column = {}

    for col in df.columns:

        missing_count = int(
            df[col].isna().sum()
        )

        if missing_count > 0:

            missing_by_column[col] = (
                missing_count
            )


    # --------------------------------------------------------
    # FINAL AI SUMMARY
    # --------------------------------------------------------

    return {

        "dataset": summary,

        "cleaning_report": report,

        "numeric_summary":
            numeric_summary,

        "categorical_summary":
            categorical_summary,

        "missing_values_by_column":
            missing_by_column
    }


# ============================================================
# AI BUSINESS ANALYSIS
# ============================================================

def generate_ai_analysis(df, report):

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if client is None:

        return (
            "AI analysis is not available because "
            "OPENAI_API_KEY has not been configured.\n\n"
            "The Pandas cleaning process completed successfully."
        )


    # --------------------------------------------------------
    # CREATE COMPACT DATA SUMMARY
    # --------------------------------------------------------

    dataset_summary = create_ai_summary(
        df,
        report
    )


    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an experienced Data Analyst and Business Analyst.

Analyze the cleaned dataset information below.

DATASET INFORMATION:
{dataset_summary}

Your task is to produce useful, evidence-based business analysis.

IMPORTANT:

1. Use only information present in the dataset summary.
2. Do not invent business facts.
3. Do not assume that a column represents revenue, profit,
   customers, sales, products, employees, etc. unless the
   column actually indicates that.
4. If business growth cannot be calculated from the available
   data, explicitly say that it cannot be determined.
5. Recommendations must be connected to actual patterns.
6. Separate facts/observations from possible interpretations.
7. Do not make up percentages or metrics.
8. Identify limitations when the available information is not enough.

Return the analysis in this structure:

EXECUTIVE SUMMARY
Give a concise overview of the most important findings.

KEY DATA INSIGHTS
Explain the strongest patterns found in the dataset.

PERFORMANCE ANALYSIS
Analyze important numeric fields and their distributions.

CUSTOMER / PRODUCT / CATEGORY INSIGHTS
Discuss relevant categorical patterns when available.

PROBLEMS AND AREAS TO IMPROVE
Identify unusual patterns, weaknesses, data-quality concerns,
or areas that deserve attention.

BUSINESS RECOMMENDATIONS
Give practical recommendations based on the evidence.

GROWTH OPPORTUNITIES
Identify possible opportunities supported by the available data.
Do not claim actual business growth unless it can be measured.

MANAGEMENT ACTION PLAN
Provide practical next steps that a manager or analyst could take.

DATA LIMITATIONS
Explain what cannot be concluded from this dataset.

Use professional but simple business language.
"""


    # --------------------------------------------------------
    # CALL OPENAI
    # --------------------------------------------------------

    try:

        response = client.responses.create(

            model="gpt-5.6-luna",

            input=prompt
        )

        return response.output_text


    except Exception as e:

        return (
            "AI analysis could not be completed.\n\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD + CLEAN
# ============================================================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if "file" not in request.files:

        return render_template(
            "index.html",
            error="Please select a file."
        )


    file = request.files["file"]


    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select a file."
        )


    if not allowed_file(
        file.filename
    ):

        return render_template(
            "index.html",
            error=(
                "Only CSV, XLSX and XLS "
                "files are supported."
            )
        )


    try:

        # ----------------------------------------------------
        # SECURE FILE NAME
        # ----------------------------------------------------

        original_filename = secure_filename(
            file.filename
        )


        unique_filename = (
            str(uuid.uuid4())
            + "_"
            + original_filename
        )


        uploaded_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            unique_filename
        )


        # ----------------------------------------------------
        # SAVE FILE
        # ----------------------------------------------------

        file.save(
            uploaded_path
        )


        # ----------------------------------------------------
        # LOAD FILE
        # ----------------------------------------------------

        df = load_data(
            uploaded_path
        )


        # ----------------------------------------------------
        # CHECK EMPTY DATASET
        # ----------------------------------------------------

        if df.empty:

            return render_template(
                "index.html",
                error=(
                    "The uploaded file does not "
                    "contain any rows."
                )
            )


        # ----------------------------------------------------
        # CLEAN DATA
        # ----------------------------------------------------

        cleaned_df, report = clean_data(
            df
        )


        # ----------------------------------------------------
        # AI ANALYSIS
        # ----------------------------------------------------

        ai_analysis = generate_ai_analysis(
            cleaned_df,
            report
        )


        # ----------------------------------------------------
        # OUTPUT FILE NAME
        # ----------------------------------------------------

        base_name = os.path.splitext(
            original_filename
        )[0]


        output_filename = (
            "cleaned_"
            + base_name
            + ".csv"
        )


        output_path = os.path.join(
            app.config["OUTPUT_FOLDER"],
            output_filename
        )


        # ----------------------------------------------------
        # SAVE CLEANED DATA
        # ----------------------------------------------------

        cleaned_df.to_csv(
            output_path,
            index=False
        )


        # ----------------------------------------------------
        # CREATE PREVIEW TABLE
        # ----------------------------------------------------

        preview = (
            cleaned_df
            .head(20)
            .to_html(
                classes="data-table",
                index=False,
                border=0
            )
        )


        # ----------------------------------------------------
        # SEND EVERYTHING TO HTML
        # ----------------------------------------------------

        return render_template(

            "index.html",

            report=report,

            preview=preview,

            ai_analysis=ai_analysis,

            download_url=(
                "/download/"
                + output_filename
            )
        )


    except Exception as e:

        return render_template(

            "index.html",

            error=(
                "Error while processing "
                "the file: "
                + str(e)
            )
        )


# ============================================================
# DOWNLOAD CLEANED FILE
# ============================================================

@app.route(
    "/download/<filename>"
)
def download_file(filename):

    # Prevent unsafe path input
    safe_filename = secure_filename(
        filename
    )

    file_path = os.path.join(
        app.config["OUTPUT_FOLDER"],
        safe_filename
    )


    if not os.path.exists(
        file_path
    ):

        return (
            "File not found.",
            404
        )


    return send_file(
        file_path,
        as_attachment=True
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )