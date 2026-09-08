from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "sales_data.csv"
SUMMARY_PATH = BASE_DIR / "summary.md"


def generate_sales_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    order_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 366, size=200), unit="D"
    )
    ratings = rng.uniform(1, 5, size=200).round(2)
    missing_indices = rng.choice(200, size=10, replace=False)
    ratings[missing_indices] = np.nan

    data = pd.DataFrame(
        {
            "order_id": [f"ORD-{index:04d}" for index in range(1, 201)],
            "order_date": order_dates,
            "region": rng.choice(["North", "South", "East", "West"], size=200),
            "product_category": rng.choice(
                ["Electronics", "Clothing", "Home", "Sports"], size=200
            ),
            "units_sold": rng.integers(1, 51, size=200),
            "unit_price": rng.uniform(5, 500, size=200).round(2),
            "customer_rating": ratings,
        }
    )
    data.to_csv(CSV_PATH, index=False, date_format="%Y-%m-%d")
    return data


def write_summary(
    data: pd.DataFrame,
    region_summary: pd.DataFrame,
    category_summary: pd.DataFrame,
    top_orders: pd.DataFrame,
) -> None:
    region_lines = [
        f"| {index} | ${row.total_revenue:,.2f} | {row.average_rating:.2f} |"
        for index, row in region_summary.iterrows()
    ]
    category_lines = [
        f"| {index} | ${row.total_revenue:,.2f} | {row.average_rating:.2f} |"
        for index, row in category_summary.iterrows()
    ]
    order_lines = [
        f"| {row.order_id} | {row.order_date:%Y-%m-%d} | {row.region} | "
        f"{row.product_category} | ${row.total_revenue:,.2f} |"
        for row in top_orders.itertuples(index=False)
    ]

    report = f"""# Sales Analysis Summary

- **Orders analyzed:** {len(data):,}
- **Total revenue:** ${data["total_revenue"].sum():,.2f}
- **Average customer rating after median fill:** {data["customer_rating"].mean():.2f}
- **Missing ratings filled:** 10

## Revenue and Rating by Region

| Region | Total Revenue | Average Rating |
|---|---:|---:|
{chr(10).join(region_lines)}

## Revenue and Rating by Product Category

| Product Category | Total Revenue | Average Rating |
|---|---:|---:|
{chr(10).join(category_lines)}

## Top 5 Highest-Revenue Orders

| Order ID | Date | Region | Category | Total Revenue |
|---|---|---|---|---:|
{chr(10).join(order_lines)}

## Charts

- `revenue_by_region.png`
- `monthly_revenue_trend.png`
"""
    SUMMARY_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    if not CSV_PATH.exists():
        generate_sales_data()

    data = pd.read_csv(CSV_PATH, parse_dates=["order_date"])
    data["customer_rating"] = data["customer_rating"].fillna(
        data["customer_rating"].median()
    )
    data["total_revenue"] = data["units_sold"] * data["unit_price"]

    region_summary = (
        data.groupby("region")
        .agg(total_revenue=("total_revenue", "sum"), average_rating=("customer_rating", "mean"))
        .sort_values("total_revenue", ascending=False)
    )
    category_summary = (
        data.groupby("product_category")
        .agg(total_revenue=("total_revenue", "sum"), average_rating=("customer_rating", "mean"))
        .sort_values("total_revenue", ascending=False)
    )
    top_orders = data.nlargest(5, "total_revenue")[
        ["order_id", "order_date", "region", "product_category", "total_revenue"]
    ]

    plt.figure(figsize=(8, 5))
    region_summary["total_revenue"].plot(kind="bar", color="#2f6690")
    plt.title("Total Revenue by Region")
    plt.xlabel("Region")
    plt.ylabel("Revenue ($)")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "revenue_by_region.png", dpi=150)
    plt.close()

    monthly_revenue = data.set_index("order_date")["total_revenue"].resample("MS").sum()
    monthly_revenue = monthly_revenue.reindex(
        pd.date_range("2024-01-01", "2024-12-01", freq="MS"), fill_value=0
    )
    plt.figure(figsize=(9, 5))
    monthly_revenue.plot(marker="o", color="#c8553d")
    plt.title("Monthly Revenue Trend - 2024")
    plt.xlabel("Month")
    plt.ylabel("Revenue ($)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "monthly_revenue_trend.png", dpi=150)
    plt.close()

    write_summary(data, region_summary, category_summary, top_orders)
    print(f"Generated {CSV_PATH.name} with {len(data)} rows")
    print(f"Total revenue: ${data['total_revenue'].sum():,.2f}")
    print("Revenue by region:")
    print(region_summary["total_revenue"].round(2).to_string())
    print("Top 5 orders:")
    print(top_orders[["order_id", "total_revenue"]].to_string(index=False))
    print(f"Wrote {SUMMARY_PATH.name}, revenue_by_region.png, and monthly_revenue_trend.png")


if __name__ == "__main__":
    main()
