import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


class StudentPerformancePipeline:
    """
    End-to-End Academic Performance Analytics Engine.
    Processes raw student records, performs defensive data auditing,
    computes statistical metrics, generates visual dashboards, and
    synthesizes data-driven recommendations.
    """

    def __init__(self, filepath: str, output_dir: str = "analytics_outputs"):
        self.filepath = filepath
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.df = None
        self.score_cols = []
        self.audit_log = {}

        # Configure distinct visual theme
        sns.set_theme(style="whitegrid", context="notebook")
        plt.rcParams["figure.dpi"] = 140
        plt.rcParams["axes.titleweight"] = "bold"
        self.palette = ["#2B5B84", "#E07A5F", "#3D405B", "#81B29A", "#F2CC8F"]

    def _section_banner(self, step_num: int, title: str):
        print(f"\n{'=' * 75}")
        print(f"TASK {step_num:02d} | {title.upper()}")
        print(f"{'=' * 75}")

    # -------------------------------------------------------------------------
    # TASK 1: Load and Inspect Dataset
    # -------------------------------------------------------------------------
    def load_and_inspect(self):
        self._section_banner(1, "Dataset Ingestion & Structural Inspection")
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"Dataset '{self.filepath}' not found. Place it in the working directory."
            )

        self.df = pd.read_csv(self.filepath)

        print(f"Dataset Dimensions: {self.df.shape[0]} Rows x {self.df.shape[1]} Columns")
        print(f"Columns Detected  : {list(self.df.columns)}\n")

        print("--- First 5 Records ---")
        print(self.df.head(5).to_string())

        print("\n--- Last 5 Records ---")
        print(self.df.tail(5).to_string())

        print("\n--- Schema & Data Types ---")
        print(self.df.dtypes.to_frame(name="Data_Type"))

        print("\n--- Descriptive Statistical Summary ---")
        print(self.df.describe(include="all").T.to_string())

    # -------------------------------------------------------------------------
    # TASK 2: Missing Values and Duplicate Records
    # -------------------------------------------------------------------------
    def audit_data_quality(self):
        self._section_banner(2, "Data Quality Audit: Missing Values & Duplicates")

        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        missing_report = pd.DataFrame({
            "Missing_Count": missing_counts,
            "Missing_Percentage": missing_pct
        })
        print("--- Missing Values Report ---")
        print(missing_report.to_string())

        dup_ids = self.df["Student_ID"].duplicated().sum() if "Student_ID" in self.df.columns else 0
        exact_dups = self.df.duplicated().sum()

        print(f"\nDuplicate Student_ID Records : {dup_ids}")
        print(f"Exact Row Duplicates         : {exact_dups}")

        if dup_ids > 0:
            print("\nDuplicate Student_ID Entries Found (Retaining first occurrence):")
            print(self.df[self.df["Student_ID"].duplicated(keep=False)].to_string())
            self.df = self.df.drop_duplicates(subset=["Student_ID"], keep="first").reset_index(drop=True)

        self.audit_log["missing_total"] = int(missing_counts.sum())
        self.audit_log["duplicate_ids"] = int(dup_ids)

    # -------------------------------------------------------------------------
    # TASK 3: Handle Missing Attendance
    # -------------------------------------------------------------------------
    def impute_attendance(self):
        self._section_banner(3, "Missing Attendance Imputation (Median Strategy)")

        if "Attendance" not in self.df.columns:
            print("Warning: 'Attendance' column not found.")
            return

        missing_before = self.df["Attendance"].isnull().sum()
        median_att = self.df["Attendance"].median()
        mean_att = self.df["Attendance"].mean()

        print(f"Missing Attendance Records (Before) : {missing_before}")
        print(f"Calculated Median Attendance        : {median_att:.2f}% (Mean: {mean_att:.2f}%)")

        self.df["Attendance"] = self.df["Attendance"].fillna(median_att)
        missing_after = self.df["Attendance"].isnull().sum()

        print(f"Missing Attendance Records (After)  : {missing_after}")
        print(
            "\nJustification for Median Imputation:\n"
            "Educational attendance distributions are typically left-skewed due to a subset of "
            "chronically absent students. Unlike the arithmetic mean, the median is robust against "
            "extreme outliers and preserves the central tendency of typical student attendance."
        )

    # -------------------------------------------------------------------------
    # TASK 4: Validate Score Columns
    # -------------------------------------------------------------------------
    def validate_scores(self):
        self._section_banner(4, "Examination Score Validation [Range: 0 - 100]")

        # Dynamically identify examination score columns
        exclude_cols = {"Student_ID", "Attendance", "Study_Hours", "Daily_Study_Hours", "Age"}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.score_cols = [
            c for c in numeric_cols
            if c not in exclude_cols and ("mark" in c.lower() or "score" in c.lower() or c not in exclude_cols)
        ]
        # Filter strictly to score/mark columns if naming follows standard convention
        explicit_marks = [c for c in numeric_cols if "mark" in c.lower() or "score" in c.lower()]
        if explicit_marks:
            self.score_cols = explicit_marks

        print(f"Exam Score Columns Identified: {self.score_cols}")

        invalid_mask = pd.Series(False, index=self.df.index)
        for col in self.score_cols:
            col_invalid = (self.df[col] < 0) | (self.df[col] > 100)
            invalid_mask = invalid_mask | col_invalid

        invalid_records = self.df[invalid_mask]
        print(f"Total Invalid Score Records Found: {len(invalid_records)}")

        if not invalid_records.empty:
            print("\n--- Invalid Score Records ---")
            print(invalid_records.to_string())
            # Clip out-of-bounds scores to [0, 100] so downstream calculations remain valid
            for col in self.score_cols:
                self.df[col] = self.df[col].clip(lower=0, upper=100)
            print("\nNote: Out-of-range values have been clipped to [0, 100] for accurate downstream aggregation.")
        else:
            print("All examination scores fall strictly within the valid [0, 100] range.")

    # -------------------------------------------------------------------------
    # TASK 5: Calculate Student Performance
    # -------------------------------------------------------------------------
    def compute_student_metrics(self):
        self._section_banner(5, "Student-Wise Total & Average Marks Calculation")

        self.df["Total_Marks"] = self.df[self.score_cols].sum(axis=1)
        self.df["Average_Marks"] = self.df[self.score_cols].mean(axis=1).round(2)

        display_cols = ["Student_ID"] + (["Department"] if "Department" in self.df.columns else []) + \
                       self.score_cols + ["Total_Marks", "Average_Marks"]
        print("--- Sample Student Performance Table (First 10) ---")
        print(self.df[display_cols].head(10).to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK 6: Create Grade Classification
    # -------------------------------------------------------------------------
    def classify_grades(self):
        self._section_banner(6, "Academic Grade Classification")

        conditions = [
            self.df["Average_Marks"] >= 80,
            (self.df["Average_Marks"] >= 70) & (self.df["Average_Marks"] < 80),
            (self.df["Average_Marks"] >= 60) & (self.df["Average_Marks"] < 70),
            (self.df["Average_Marks"] >= 40) & (self.df["Average_Marks"] < 60),
            self.df["Average_Marks"] < 40
        ]
        choices = ["A", "B", "C", "D", "F"]
        self.df["Grade"] = np.select(conditions, choices, default="F")

        grade_order = ["A", "B", "C", "D", "F"]
        grade_counts = self.df["Grade"].value_counts().reindex(grade_order, fill_value=0)
        grade_summary = pd.DataFrame({
            "Student_Count": grade_counts,
            "Percentage (%)": (grade_counts / len(self.df) * 100).round(2)
        })
        print("--- Grade Distribution Summary ---")
        print(grade_summary.to_string())

    # -------------------------------------------------------------------------
    # TASK 7: Department-wise Performance
    # -------------------------------------------------------------------------
    def analyze_departments(self):
        self._section_banner(7, "Department-Wise Academic Performance")

        if "Department" not in self.df.columns:
            print("Warning: 'Department' column not found.")
            return

        dept_summary = (
            self.df.groupby("Department")
            .agg(
                Student_Count=("Student_ID", "count"),
                Mean_Average_Marks=("Average_Marks", "mean"),
                Median_Average_Marks=("Average_Marks", "median"),
                Std_Dev=("Average_Marks", "std"),
                Mean_Attendance=("Attendance", "mean")
            )
            .round(2)
            .sort_values(by="Mean_Average_Marks", ascending=False)
        )
        print(dept_summary.to_string())

        top_dept = dept_summary["Mean_Average_Marks"].idxmax()
        top_score = dept_summary.loc[top_dept, "Mean_Average_Marks"]
        low_dept = dept_summary["Mean_Average_Marks"].idxmin()
        low_score = dept_summary.loc[low_dept, "Mean_Average_Marks"]

        print(f"\nHighest Performing Department : {top_dept} (Avg: {top_score:.2f})")
        print(f"Lowest Performing Department  : {low_dept} (Avg: {low_score:.2f})")
        print(f"Inter-Departmental Spread     : {top_score - low_score:.2f} points")

        self.audit_log["top_dept"] = (top_dept, top_score)
        self.audit_log["low_dept"] = (low_dept, low_score)

    # -------------------------------------------------------------------------
    # TASK 8: Top 10 Academic Performers
    # -------------------------------------------------------------------------
    def extract_top_performers(self):
        self._section_banner(8, "Top 10 Academic Performers")

        top_10 = self.df.sort_values(by=["Average_Marks", "Total_Marks"], ascending=False).head(10)
        cols = [c for c in ["Student_ID", "Name", "Department", "Attendance", "Total_Marks", "Average_Marks", "Grade"] if c in self.df.columns]
        print(top_10[cols].to_string(index=False))

    # -------------------------------------------------------------------------
    # TASK 9: Students Requiring Academic Support
    # -------------------------------------------------------------------------
    def identify_at_risk_students(self):
        self._section_banner(9, "Students Requiring Academic Support (Average < 40%)")

        at_risk = self.df[self.df["Average_Marks"] < 40].sort_values(by="Average_Marks", ascending=True)
        req_cols = [c for c in ["Student_ID", "Department", "Attendance", "Average_Marks"] if c in self.df.columns]

        print(f"Total At-Risk Students Identified: {len(at_risk)} ({len(at_risk)/len(self.df)*100:.2f}% of cohort)\n")
        if not at_risk.empty:
            print(at_risk[req_cols].to_string(index=False))
        else:
            print("No students scored below 40%.")

        self.audit_log["at_risk_count"] = len(at_risk)

    # -------------------------------------------------------------------------
    # TASK 10: Attendance Band Analysis
    # -------------------------------------------------------------------------
    def analyze_attendance_bands(self):
        self._section_banner(10, "Attendance Band Performance Analysis")

        bins = [-np.inf, 59.999, 74.999, 89.999, 100.0]
        labels = ["Below 60%", "60–74%", "75–89%", "90–100%"]
        self.df["Attendance_Band"] = pd.cut(self.df["Attendance"], bins=bins, labels=labels)

        band_summary = (
            self.df.groupby("Attendance_Band", observed=False)
            .agg(
                Student_Count=("Student_ID", "count"),
                Average_Marks=("Average_Marks", "mean"),
                Median_Marks=("Average_Marks", "median"),
                Pass_Rate_Pct=("Average_Marks", lambda s: (s >= 40).mean() * 100)
            )
            .round(2)
        )
        print(band_summary.to_string())
        self.audit_log["att_bands"] = band_summary

    # -------------------------------------------------------------------------
    # TASK 11: Study Hours and Average Marks
    # -------------------------------------------------------------------------
    def analyze_study_hours(self):
        self._section_banner(11, "Daily Study Hours vs. Academic Performance")

        study_col = next((c for c in self.df.columns if "study" in c.lower()), None)
        if not study_col:
            print("Warning: Study hours column not found.")
            return
        self.study_col = study_col

        corr = self.df[study_col].corr(self.df["Average_Marks"])
        print(f"Study Hours Column Used          : '{study_col}'")
        print(f"Pearson Correlation Coefficient  : r = {corr:.4f}")

        if corr >= 0.7:
            strength = "Strong positive linear relationship"
        elif corr >= 0.4:
            strength = "Moderate positive linear relationship"
        elif corr >= 0.1:
            strength = "Weak positive linear relationship"
        else:
            strength = "Negligible or negative linear relationship"
        print(f"Statistical Interpretation       : {strength}")

        self.audit_log["study_corr"] = corr

    # -------------------------------------------------------------------------
    # TASK 12: Internet Access Comparison
    # -------------------------------------------------------------------------
    def analyze_internet_access(self):
        self._section_banner(12, "Internet Accessibility Impact Analysis")

        net_col = next((c for c in self.df.columns if "internet" in c.lower()), None)
        if not net_col:
            print("Warning: Internet access column not found.")
            return
        self.net_col = net_col

        net_summary = (
            self.df.groupby(net_col)
            .agg(
                Student_Count=("Student_ID", "count"),
                Mean_Average_Marks=("Average_Marks", "mean"),
                Median_Average_Marks=("Average_Marks", "median"),
                Mean_Study_Hours=(self.study_col, "mean") if hasattr(self, "study_col") else ("Average_Marks", "count")
            )
            .round(2)
        )
        print(net_summary.to_string())
        self.audit_log["net_summary"] = net_summary

    # -------------------------------------------------------------------------
    # TASK 13 & 14: NumPy Statistics & IQR Outlier Detection on Python_Marks
    # -------------------------------------------------------------------------
    def analyze_python_marks_numpy_iqr(self):
        self._section_banner(13, "NumPy Statistical Profile: Python_Marks")

        if "Python_Marks" not in self.df.columns:
            print("Warning: 'Python_Marks' column not found.")
            return

        py_scores = self.df["Python_Marks"].dropna().to_numpy()

        median_val = np.median(py_scores)
        std_val = np.std(py_scores, ddof=1)  # Sample standard deviation
        q1 = np.percentile(py_scores, 25)
        q3 = np.percentile(py_scores, 75)

        print(f"Median (50th Percentile) : {median_val:.2f}")
        print(f"Standard Deviation (s)   : {std_val:.2f}")
        print(f"First Quartile (Q1)      : {q1:.2f}")
        print(f"Third Quartile (Q3)      : {q3:.2f}")

        self._section_banner(14, "IQR-Based Outlier Detection: Python_Marks")
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = self.df[(self.df["Python_Marks"] < lower_bound) | (self.df["Python_Marks"] > upper_bound)]

        print(f"Interquartile Range (IQR) : {iqr:.2f}")
        print(f"Lower Outlier Fence       : {lower_bound:.2f}")
        print(f"Upper Outlier Fence       : {upper_bound:.2f}")
        print(f"Outliers Detected         : {len(outliers)}")

        if not outliers.empty:
            cols = [c for c in ["Student_ID", "Department", "Python_Marks", "Average_Marks"] if c in self.df.columns]
            print("\n--- Outlier Records ---")
            print(outliers[cols].to_string(index=False))

        self.audit_log["py_stats"] = {
            "median": median_val, "std": std_val, "q1": q1, "q3": q3,
            "iqr": iqr, "lower": lower_bound, "upper": upper_bound, "outliers": len(outliers)
        }

    # -------------------------------------------------------------------------
    # TASK 15: Academic Performance Visualizations
    # -------------------------------------------------------------------------
    def generate_visualizations(self):
        self._section_banner(15, "Generating Academic Analytics Charts")

        # 1. Grade Distribution
        plt.figure(figsize=(8, 5))
        grade_order = ["A", "B", "C", "D", "F"]
        ax = sns.countplot(data=self.df, x="Grade", order=grade_order, palette=self.palette)
        for p in ax.patches:
            ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha="center", va="bottom", fontsize=10, fontweight="bold")
        plt.title("Cohort Grade Distribution")
        plt.xlabel("Academic Grade")
        plt.ylabel("Number of Students")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "01_grade_distribution.png"))
        plt.close()

        # 2. Department-wise Average Marks
        if "Department" in self.df.columns:
            plt.figure(figsize=(9, 5))
            dept_order = self.df.groupby("Department")["Average_Marks"].mean().sort_values(ascending=False).index
            ax = sns.barplot(data=self.df, x="Department", y="Average_Marks", order=dept_order,
                             palette="viridis", errorbar="sd", capsize=0.1)
            plt.title("Department-Wise Average Marks (with ±1 SD)")
            plt.xlabel("Academic Department")
            plt.ylabel("Average Marks")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "02_department_performance.png"))
            plt.close()

        # 3. Attendance-wise Average Marks
        plt.figure(figsize=(8, 5))
        band_order = ["Below 60%", "60–74%", "75–89%", "90–100%"]
        sns.barplot(data=self.df, x="Attendance_Band", y="Average_Marks", order=band_order,
                    palette="Blues_d", errorbar=None)
        plt.title("Average Academic Marks Across Attendance Bands")
        plt.xlabel("Attendance Band")
        plt.ylabel("Mean Average Marks")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "03_attendance_band_marks.png"))
        plt.close()

        # 4. Study Hours vs Average Marks (Scatter + Regression Line)
        if hasattr(self, "study_col"):
            plt.figure(figsize=(8, 5))
            sns.regplot(data=self.df, x=self.study_col, y="Average_Marks",
                        scatter_kws={"alpha": 0.65, "color": "#2B5B84"},
                        line_kws={"color": "#E07A5F", "linewidth": 2})
            r_val = self.audit_log.get("study_corr", 0)
            plt.title(f"Daily Study Hours vs. Average Marks (r = {r_val:.2f})")
            plt.xlabel("Daily Study Hours")
            plt.ylabel("Average Marks")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "04_study_hours_vs_marks.png"))
            plt.close()

        # 5. Internet Access Comparison (Boxplot + Stripplot)
        if hasattr(self, "net_col"):
            plt.figure(figsize=(7, 5))
            sns.boxplot(data=self.df, x=self.net_col, y="Average_Marks", palette=["#81B29A", "#E07A5F"], width=0.45)
            sns.stripplot(data=self.df, x=self.net_col, y="Average_Marks", color="#3D405B", alpha=0.5, jitter=True)
            plt.title("Impact of Internet Accessibility on Average Marks")
            plt.xlabel("Internet Access")
            plt.ylabel("Average Marks")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "05_internet_access_comparison.png"))
            plt.close()

        # 6. Python Marks Distribution (Histogram + KDE + Boxplot Subplot)
        if "Python_Marks" in self.df.columns:
            fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.18, .82)}, figsize=(8, 5.5))
            sns.boxplot(x=self.df["Python_Marks"], ax=ax_box, color="#2B5B84")
            ax_box.set(xlabel="")
            ax_box.set_title("Python Marks Distribution & Outlier Profile")

            sns.histplot(self.df["Python_Marks"], kde=True, ax=ax_hist, color="#2B5B84", bins=15)
            ax_hist.axvline(self.audit_log["py_stats"]["median"], color="#E07A5F", linestyle="--", label="Median")
            ax_hist.set_xlabel("Python Marks")
            ax_hist.set_ylabel("Frequency")
            ax_hist.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "06_python_marks_distribution.png"))
            plt.close()

        print(f"All 6 analytical charts exported successfully to '{self.output_dir}/'.")

    # -------------------------------------------------------------------------
    # TASK 16: Data-Driven Academic Improvement Recommendations
    # -------------------------------------------------------------------------
    def generate_recommendations(self):
        self._section_banner(16, "Data-Driven Academic Improvement Recommendations")

        top_dept, top_score = self.audit_log.get("top_dept", ("Top Dept", 0))
        low_dept, low_score = self.audit_log.get("low_dept", ("Lowest Dept", 0))
        at_risk = self.audit_log.get("at_risk_count", 0)
        corr = self.audit_log.get("study_corr", 0)

        recommendations = [
            f"1. Early-Warning Intervention for At-Risk Cohort ({at_risk} Students): "
            f"Enroll all {at_risk} students scoring below 40% into mandatory weekly remedial bootcamps "
            f"and assign peer mentors before mid-semester evaluations.",

            f"2. Automated Attendance Threshold Alerts: "
            f"Data across attendance bands confirms a sharp academic drop-off below 75% attendance. "
            f"Implement automated advisor alerts when any student's attendance dips below 75%.",

            f"3. Cross-Departmental Pedagogy Transfer ({top_dept} to {low_dept}): "
            f"Audit the instructional frameworks, lab structures, and assessment rubrics used by "
            f"{top_dept} (avg: {top_score:.2f}) and adapt them to support {low_dept} (avg: {low_score:.2f}).",

            f"4. Structured Daily Study Habit Scaffolding (r = {corr:.2f}): "
            f"Given the observed correlation (r = {corr:.2f}) between daily study hours and marks, "
            f"introduce supervised after-hours study halls and weekly milestone trackers for students logging low study hours.",

            f"5. Bridging the Digital Divide for Offline Students: "
            f"Provide extended campus computer-lab hours, offline-accessible LMS study packs, and "
            f"subsidized mobile data vouchers for students lacking home internet access."
        ]

        for rec in recommendations:
            print(rec + "\n")

    def export_cleaned_dataset(self):
        out_csv = os.path.join(self.output_dir, "02_student_performance_cleaned.csv")
        self.df.to_csv(out_csv, index=False)
        print(f"Cleaned & enriched dataset saved to: '{out_csv}'")

    def run_all(self):
        self.load_and_inspect()
        self.audit_data_quality()
        self.impute_attendance()
        self.validate_scores()
        self.compute_student_metrics()
        self.classify_grades()
        self.analyze_departments()
        self.extract_top_performers()
        self.identify_at_risk_students()
        self.analyze_attendance_bands()
        self.analyze_study_hours()
        self.analyze_internet_access()
        self.analyze_python_marks_numpy_iqr()
        self.generate_visualizations()
        self.generate_recommendations()
        self.export_cleaned_dataset()


if __name__ == "__main__":
    pipeline = StudentPerformancePipeline(filepath="02_student_performance.csv")
    pipeline.run_all()