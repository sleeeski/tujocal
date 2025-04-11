import streamlit as st
import polars as pl
import plotly.express as px


def load_data(consumption_file):
    """Load and merge all necessary data files using Polars."""
    try:
        consumption_df = pl.read_csv(consumption_file)
        if consumption_df.shape[1] < 2:
            st.error("CSV-tiedostossa on oltava vähintään kaksi saraketta: aikaleima ja kulutus.")
            return None, None
        
        consumption_df = consumption_df.select([
            pl.col(consumption_df.columns[0]).alias("time_stamp"),
            pl.col(consumption_df.columns[1]).alias("consumption")
        ])
        
        price_df = pl.read_csv("electricity_prices.csv")
        reserve_df = pl.read_csv("reserve.csv").with_columns(
            pl.col("reserve_prices") * 0.1
        )
        
        merged_df = consumption_df.join(price_df, on="time_stamp", how="inner").join(
            reserve_df, on="time_stamp", how="inner"
        )
        
        merged_df = merged_df.with_columns([
            (pl.col("consumption") * pl.col("price")).alias("cost"),
            pl.col("time_stamp").str.to_datetime().alias("time_stamp")
        ])
        
        consumption_df = consumption_df.join(price_df, on="time_stamp", how="inner").with_columns([
            (pl.col("price")).alias("price"),
            pl.col("time_stamp").str.to_datetime().alias("time_stamp"),
            pl.col("consumption").cum_sum().alias("cumulative_consumption")
        ])
        
        return merged_df, consumption_df
    except FileNotFoundError:
        st.error("Hintatietotiedostoa 'electricity_prices.csv' tai 'reserve.csv' ei löytynyt.")
        return None, None
    except Exception as e:
        st.error(f"Virhe tiedoston lukemisessa: {e}")
        return None, None


def get_plotly_config():
    """Return standardized config for Plotly charts."""
    return {
        "scrollZoom": True,
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn", "zoomOut", "autoScale2d", "toImage"
        ]
    }


def display_consumption_metrics(consumption_df, merged_df):
    """Display key consumption metrics using Polars."""
    avg_hourly = consumption_df["consumption"].mean()
    avg_daily = consumption_df.group_by(pl.col("time_stamp").dt.date()).agg(
        pl.col("consumption").sum()
    )["consumption"].mean()
    avg_monthly = consumption_df.group_by(pl.col("time_stamp").dt.truncate("1mo")).agg(
        pl.col("consumption").sum()
    )["consumption"].mean()
    total = consumption_df["consumption"].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Keskimääräinen kulutus tunnissa", value=f"{avg_hourly:.0f} kWh")
    col2.metric(label="Keskimääräinen kulutus päivässä", value=f"{avg_daily:.0f} kWh")
    col3.metric(label="Keskimääräinen kulutus kuukaudessa", value=f"{avg_monthly:.0f} kWh")
    col4.metric(label="Kulutus yhteensä", value=f"{total:.0f} kWh")

    avg_hourly = merged_df["price"].mean()
    highest = merged_df["price"].max()

    st.subheader("Hintatiedot")
    col5, col6 = st.columns(2)
    col5.metric(label="Keskim. sähkönhinta tunneittain", value=f"{avg_hourly:.2f} snt/kWh")
    col6.metric(label="Korkein tuntihinta", value=f"{highest:.0f} snt/kWh")


def create_consumption_chart(consumption_df):
    """Create and display the consumption chart using Polars."""
    aggregation_period = st.selectbox(
        label="Valitse aikaväli kaavioon:", 
        options=["Päivä", "Viikko", "Kuukausi"], 
        index=0, 
        key="consume_period"
    )
    is_cumulative = st.toggle(label="Kumulatiivinen", key="consume_toggle")
    
    chart_name = "cumulative_consumption" if is_cumulative else "consumption"
    agg_type = pl.col(chart_name).last() if is_cumulative else pl.col(chart_name).sum()
    
    if aggregation_period == "Päivä":
        chart_data = consumption_df.group_by(pl.col("time_stamp").dt.date().alias("Päivä")).agg([
            agg_type.alias("Kulutus (kWh)"),
            pl.col("price").mean().alias("Keskim. sähkönhinta (snt/kWh)")
        ]).sort("Päivä")
        time_label = "päivittäin"
        x_col = "Päivä"
    elif aggregation_period == "Kuukausi":
        chart_data = consumption_df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("Kuukausi")).agg([
            agg_type.alias("Kulutus (kWh)"),
            pl.col("price").mean().alias("Keskim. sähkönhinta (snt/kWh)")
        ]).with_columns(
            pl.col("Kuukausi").dt.strftime("%Y-%m").alias("Kuukausi")
        ).sort("Kuukausi")
        time_label = "kuukausittain"
        x_col = "Kuukausi"
    else:
        chart_data = consumption_df.group_by([
            pl.col("time_stamp").dt.year().alias("year"),
            pl.col("time_stamp").dt.week().alias("week")
        ]).agg([
            agg_type.alias("Kulutus (kWh)"),
            pl.col("price").mean().alias("Keskim. sähkönhinta (snt/kWh)")
        ]).with_columns(
            (pl.col("year").cast(str) + "-W" + pl.col("week").cast(str).str.zfill(2)).alias("Viikko")
        ).sort("Viikko")
        time_label = "viikottain"
        x_col = "Viikko"
    
    y_max = chart_data["Kulutus (kWh)"].max() * 1.1
    
    chart_data_pd = chart_data.select([x_col, "Kulutus (kWh)", "Keskim. sähkönhinta (snt/kWh)"]).to_pandas()
    
    fig = px.line(
        chart_data_pd,
        x=x_col,
        y=["Kulutus (kWh)"],
        title=f"Kulutus esitettynä {time_label}",
        labels={
            x_col: "Aika",
            "value": "Kulutus (kWh)",
            "variable": "Tyyppi"
        }
    )
    fig.update_traces(mode="lines", hovertemplate="%{data.name}: %{y:.2f}<extra></extra>")
    fig.update_layout(
        xaxis_title="Aika",
        yaxis_title="Kulutus (kWh)",
        hovermode="x unified",
        dragmode="pan",
        height=500,
        yaxis=dict(range=[0, y_max], fixedrange=True),
        xaxis=dict(fixedrange=False),
        showlegend=False
    )

    fig1 = px.line(
        chart_data_pd,
        x=x_col,
        y=["Keskim. sähkönhinta (snt/kWh)"],
        labels={
            x_col: "Aika",
            "value": "Kulutus (kWh)",
            "variable": "Tyyppi"
        }
    )
    fig1.update_traces(mode="lines", hovertemplate="%{data.name}: %{y:.2f}<extra></extra>")
    fig1.update_layout(
        xaxis_title="Aika",
        yaxis_title="Keskim. sähkönhinta (snt/kWh)",
        hovermode="x unified",
        dragmode="pan",
        height=500,
        yaxis=dict(range=[0, chart_data_pd["Keskim. sähkönhinta (snt/kWh)"].max() * 1.1], fixedrange=True),
        xaxis=dict(fixedrange=False),
        showlegend=False
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, use_container_width=True, config=config, key="consume")
    st.plotly_chart(fig1, use_container_width=True, config=config)

def get_basic_battery_parameters():
    """Get basic battery parameters from user input."""
    col1, col2, col3 = st.columns(3)
    with col1:
        max_capacity = st.number_input(
            "Valitse akun kapasiteetti (kWh)", 
            min_value=0, max_value=1000, value=10, key="basic_capacity_tab"
        )
    with col2:
        max_charge_rate = st.number_input(
            "Valitse akun teho (kW)", 
            min_value=0, value=10, key="basic_charge_rate_tab"
        )
    with col3:
        battery_price = st.number_input(
            "Valitse akun hinta (€)", 
            min_value=0, value=10000, key="basic_price_tab"
        )
    return max_capacity, max_charge_rate, battery_price


def simulate_basic_battery(merged_df, max_capacity, max_charge_rate):
    """Simulate basic battery charging and discharging strategy using Polars."""
    df = merged_df.clone()
    price_limit = df["price"].mean() * 0.5
    initial_charge = 0.0
    
    df = df.with_columns([
        pl.lit(initial_charge).alias("battery_charge"),
        (pl.col("consumption") * pl.col("price")).alias("cost")
    ])
    
    def update_battery(df):
        current_charge = initial_charge
        charges = []
        costs = []
        for row in df.iter_rows(named=True):
            if row["price"] < price_limit and current_charge < max_capacity:
                charge_amount = min(max_charge_rate, max_capacity - current_charge)
                current_charge += charge_amount
                cost = row["cost"] + charge_amount * row["price"]
            elif current_charge > 0:
                discharge_amount = min(row["consumption"], current_charge)
                current_charge -= discharge_amount
                cost = row["cost"] - discharge_amount * row["price"]
            else:
                cost = row["cost"]
            charges.append(current_charge)
            costs.append(cost)
        return df.with_columns([
            pl.Series("battery_charge", charges),
            pl.Series("cost", costs)
        ])
    
    df = update_battery(df)
    
    df = df.with_columns(
        (pl.col("consumption") * pl.col("price")).alias("cost_without_battery")
    )
    df = df.with_columns([
        pl.col("cost_without_battery").cum_sum().alias("cumulative_cost_without_battery"),
        pl.col("cost").cum_sum().alias("cumulative_cost_with_battery")
    ])
    
    return df


def display_basic_battery_metrics(df):
    """Display cost metrics for basic battery simulation using Polars."""
    total_without_battery = df["cost_without_battery"].sum() * 0.01
    total_with_battery = df["cost"].sum() * 0.01
    total_savings = total_without_battery - total_with_battery
    
    time_span_days = (df["time_stamp"].max() - df["time_stamp"].min()).days + 1
    time_span_months = time_span_days / 30.44
    time_span_years = time_span_days / 365.25
    
    avg_savings_per_day = total_savings / time_span_days if time_span_days > 0 else 0
    avg_savings_per_month = total_savings / time_span_months if time_span_months > 0 else 0
    avg_savings_per_year = total_savings / time_span_years if time_span_years > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Kulut ilman akkua", value=f"{total_without_battery:.2f}€")
    col2.metric(label="Kulut akulla", value=f"{total_with_battery:.2f}€")
    col3.metric(label="Säästöt akusta", value=f"{total_savings:.2f}€")
    
    col4, col5, col6 = st.columns(3)
    col4.metric(label="Keskim. säästöt päivässä", value=f"{avg_savings_per_day:.2f}€")
    col5.metric(label="Keskim. säästöt kuukaudessa", value=f"{avg_savings_per_month:.2f}€")
    col6.metric(label="Keskim. säästöt vuodessa", value=f"{avg_savings_per_year:.2f}€")


def create_basic_battery_chart(df, battery_price):
    """Create and display the cost comparison chart for basic battery using Polars."""
    aggregation_period = st.selectbox(
        "Valitse aikaväli kaavioon:", 
        ["Päivä", "Viikko", "Kuukausi"], 
        index=0, key="basic_period_tab"
    )
    is_cumulative = st.toggle("Kumulatiivinen", key="basic_toggle_tab")
    show_price_line = st.toggle("Näytä akun hinta viivana", key="basic_price_line_toggle")
    
    no_battery_column = "cumulative_cost_without_battery" if is_cumulative else "cost_without_battery"
    with_battery_column = "cumulative_cost_with_battery" if is_cumulative else "cost"
    agg_type = pl.last if is_cumulative else pl.sum
    
    if aggregation_period == "Päivä":
        chart_data = df.group_by(pl.col("time_stamp").dt.date().alias("Päivä")).agg([
            agg_type(no_battery_column).alias("Hinta ilman akkua"),
            agg_type(with_battery_column).alias("Hinta akulla")
        ]).sort("Päivä")
        time_label = "päivittäin"
        x_col = "Päivä"
    elif aggregation_period == "Kuukausi":
        chart_data = df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("Kuukausi")).agg([
            agg_type(no_battery_column).alias("Hinta ilman akkua"),
            agg_type(with_battery_column).alias("Hinta akulla")
        ]).with_columns(
            pl.col("Kuukausi").dt.strftime("%Y-%m").alias("Kuukausi")
        ).sort("Kuukausi")
        time_label = "kuukausittain"
        x_col = "Kuukausi"
    else:
        chart_data = df.group_by([
            pl.col("time_stamp").dt.year().alias("year"),
            pl.col("time_stamp").dt.week().alias("week")
        ]).agg([
            agg_type(no_battery_column).alias("Hinta ilman akkua"),
            agg_type(with_battery_column).alias("Hinta akulla")
        ]).with_columns(
            (pl.col("year").cast(str) + "-W" + pl.col("week").cast(str).str.zfill(2)).alias("Viikko")
        ).sort("Viikko")
        time_label = "viikottain"
        x_col = "Viikko"
    
    chart_data = chart_data.with_columns(pl.col("*").exclude(x_col) * 0.01)
    y_max = max(chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla")]).max().to_numpy().max() * 1.1, battery_price * 1.1) if show_price_line else chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla")]).max().to_numpy().max() * 1.1
    y_min = chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla")]).min().to_numpy().min() * 1.1 if chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla")]).min().to_numpy().min() < 0 else 0
    
    chart_data_pd = chart_data.select([x_col, "Hinta ilman akkua", "Hinta akulla"]).to_pandas()
    
    fig = px.line(
        chart_data_pd,
        x=x_col,
        y=["Hinta ilman akkua", "Hinta akulla"],
        title=f"Kustannukset esitettynä {time_label}",
        labels={
            x_col: "Aika",
            "value": "Hinta (€)",
            "variable": "Tyyppi"
        }
    )
    fig.update_traces(mode="lines", hovertemplate="%{data.name}: %{y:.2f}€<extra></extra>")
    
    if show_price_line:
        fig.add_hline(
            y=battery_price,
            line_dash="dash",
            line_color="orange",
            annotation_text="Akun hinta",
            annotation_position="top left"
        )
    
    fig.update_layout(
        xaxis_title="Aika",
        yaxis_title="Hinta (€)",
        legend_title="Tyyppi",
        hovermode="x unified",
        dragmode="pan",
        height=500,
        yaxis=dict(range=[y_min, y_max], fixedrange=True),
        xaxis=dict(fixedrange=False)
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, use_container_width=True, config=config)


def get_reserve_battery_parameters(max_capacity):
    """Get reserve capacity for reserve simulation."""
    reserve_capacity = st.number_input(
        "Valitse reservikapasiteetti (kW)", 
        min_value=0, max_value=max_capacity, value=max_capacity, key="reserve_amount_tab"
    )
    return reserve_capacity


def simulate_reserve_battery(merged_df, max_capacity, max_charge_rate, reserve_capacity):
    """Simulate battery operation with reserve capacity using Polars."""
    df = merged_df.clone()
    price_limit = df["price"].mean() * 0.5
    initial_charge = 0.0
    
    df = df.with_columns([
        pl.lit(initial_charge).alias("battery_charge_reserve"),
        (pl.col("consumption") * pl.col("price")).alias("cost_for_reserve")
    ])
    
    def update_reserve_battery(df):
        current_charge = initial_charge
        charges = []
        costs = []
        for row in df.iter_rows(named=True):
            if row["price"] < price_limit and current_charge < max_capacity:
                charge_amount = min(max_charge_rate - reserve_capacity, max_capacity - current_charge)
                current_charge += charge_amount
                cost = row["cost_for_reserve"] + charge_amount * row["price"]
            elif current_charge > 0:
                discharge_amount = min(row["consumption"], current_charge)
                current_charge -= discharge_amount
                cost = row["cost_for_reserve"] - discharge_amount * row["price"]
            else:
                cost = row["cost_for_reserve"]
            
            cost -= reserve_capacity * row["reserve_prices"]
            
            charges.append(current_charge)
            costs.append(cost)
        return df.with_columns([
            pl.Series("battery_charge_reserve", charges),
            pl.Series("cost_for_reserve", costs)
        ])
    
    df = update_reserve_battery(df)
    
    df = df.with_columns(
        (pl.col("consumption") * pl.col("price")).alias("cost_without_battery")
    )
    df = df.with_columns([
        pl.col("cost_without_battery").cum_sum().alias("cumulative_cost_without_battery"),
        pl.col("cost_for_reserve").cum_sum().alias("cumulative_cost_with_battery_reserve")
    ])
    
    return df


def display_reserve_battery_metrics(df):
    """Display cost metrics for reserve battery simulation using Polars."""
    total_without_battery = df["cost_without_battery"].sum() * 0.01
    total_with_reserve = df["cost_for_reserve"].sum() * 0.01
    total_earnings = total_without_battery - total_with_reserve
    
    time_span_days = (df["time_stamp"].max() - df["time_stamp"].min()).days + 1
    time_span_months = time_span_days / 30.44
    time_span_years = time_span_days / 365.25
    
    avg_earnings_per_day = total_earnings / time_span_days if time_span_days > 0 else 0
    avg_earnings_per_month = total_earnings / time_span_months if time_span_months > 0 else 0
    avg_earnings_per_year = total_earnings / time_span_years if time_span_years > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Keskim. tuotto päivässä", value=f"{avg_earnings_per_day:.2f}€")
    col2.metric(label="Keskim. tuotto kuukaudessa", value=f"{avg_earnings_per_month:.2f}€")
    col3.metric(label="Keskim. tuotto vuodessa", value=f"{avg_earnings_per_year:.2f}€")
    st.metric(label="Kokonaisetu reservistä", value=f"{total_earnings:.2f}€")

def create_reserve_prices_chart(df):
    """Create and display the reservemarket prices using Polars"""
    aggregation_period = st.selectbox(
        "Valitse aikaväli kaavioon:",
        ["Päivä", "Viikko", "Kuukausi"],
        index=0, key="reserve_prices_period"
    )
    if aggregation_period == "Päivä":
        chart_data = df.group_by(pl.col("time_stamp").dt.date().alias("Päivä")).agg([
            pl.sum("reserve_prices").alias("FCR-N hinta")
        ]).sort("Päivä")
        time_label = "päivittäin"
        x_col = "Päivä"
    elif aggregation_period == "Kuukausi":
        chart_data = df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("Kuukausi")).agg([
            pl.sum("reserve_prices").alias("FCR-N hinta")
        ]).with_columns(
            pl.col("Kuukausi").dt.strftime("%Y-%m").alias("Kuukausi")
        ).sort("Kuukausi")
        time_label = "kuukausittain"
        x_col = "Kuukausi"
    else:
        chart_data = df.group_by([
            pl.col("time_stamp").dt.year().alias("year"),
            pl.col("time_stamp").dt.year().alias("week")
        ]).agg([
            pl.sum("reserve_price").alias("FCR-N hinta")
        ]).with_columns(
            (pl.col("year").cast(str) + "-W" + pl.col("week").cast(str).str.zfill(2)).alias("Viikko")
        ).sort("Viikko")
        time_label = "viikottain"
        x_col = "Viikko"
    
    y_max = chart_data.select([pl.col("FCR-N hinta")]).max().to_numpy().max() * 1.1
    y_min = 0.0
    chart_data_pd = chart_data.select([x_col, "FCR-N hinta"]).to_pandas()
    
    fig = px.line(
        chart_data_pd,
        x=x_col,
        y=["FCR-N hinta"],
        title=f"Hinnat esitettynä {time_label}",
        labels={
            x_col: "Aika",
            "value": "Hinta (EUR/MW)",
            "variable": "Tyyppi"
        }
    )
    fig.update_traces(mode="lines", hovertemplate="%{data.name}: %{y:.2f}€<extra></extra>")
    fig.update_layout(
        xaxis_title="Aika",
        yaxis_title="Hinta (EUR/MW)",
        legend_title="Tyyppi",
        hovermode="x unified",
        dragmode="pan",
        height=500,
        yaxis=dict(range=[y_min, y_max], fixedrange=True),
        xaxis=dict(fixedrange=False)
    )
    
    config = get_plotly_config()
    st.plotly_chart(fig, use_container_width=True, config=config)

def create_reserve_battery_chart(df, basic_battery_df, battery_price):
    """Create and display the cost comparison chart for reserve battery using Polars."""
    aggregation_period = st.selectbox(
        "Valitse aikaväli kaavioon:", 
        ["Päivä", "Viikko", "Kuukausi"], 
        index=0, key="reserve_period_tab"
    )
    is_cumulative = st.toggle("Kumulatiivinen", key="reserve_toggle_tab")
    show_price_line = st.toggle("Näytä akun hinta viivana", key="reserve_price_line_toggle")
    
    no_battery_column = "cumulative_cost_without_battery" if is_cumulative else "cost_without_battery"
    with_battery_column = "cumulative_cost_with_battery" if is_cumulative else "cost"
    reserve_column = "cumulative_cost_with_battery_reserve" if is_cumulative else "cost_for_reserve"
    agg_type = pl.last if is_cumulative else pl.sum
    
    if aggregation_period == "Päivä":
        reserve_data = df.group_by(pl.col("time_stamp").dt.date().alias("Päivä")).agg([
            agg_type(reserve_column).alias("Reservimarkkinoiden tuotto"),
            agg_type(no_battery_column).alias("Hinta ilman akkua")
        ]).sort("Päivä")
        battery_data = basic_battery_df.group_by(pl.col("time_stamp").dt.date().alias("Päivä")).agg(
            agg_type(with_battery_column).alias("Hinta akulla")
        ).sort("Päivä")
        chart_data = reserve_data.join(battery_data, on="Päivä", how="full")
        time_label = "päivittäin"
        x_col = "Päivä"
    elif aggregation_period == "Kuukausi":
        reserve_data = df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("Kuukausi")).agg([
            agg_type(reserve_column).alias("Reservimarkkinoiden tuotto"),
            agg_type(no_battery_column).alias("Hinta ilman akkua")
        ]).with_columns(
            pl.col("Kuukausi").dt.strftime("%Y-%m").alias("Kuukausi")
        ).sort("Kuukausi")
        battery_data = basic_battery_df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("Kuukausi")).agg(
            agg_type(with_battery_column).alias("Hinta akulla")
        ).with_columns(
            pl.col("Kuukausi").dt.strftime("%Y-%m").alias("Kuukausi")
        ).sort("Kuukausi")
        chart_data = reserve_data.join(battery_data, on="Kuukausi", how="full")
        time_label = "kuukausittain"
        x_col = "Kuukausi"
    else:
        reserve_data = df.group_by([
            pl.col("time_stamp").dt.year().alias("year"),
            pl.col("time_stamp").dt.week().alias("week")
        ]).agg([
            agg_type(reserve_column).alias("Reservimarkkinoiden tuotto"),
            agg_type(no_battery_column).alias("Hinta ilman akkua")
        ]).with_columns(
            (pl.col("year").cast(str) + "-W" + pl.col("week").cast(str).str.zfill(2)).alias("Viikko")
        ).sort("Viikko")
        battery_data = basic_battery_df.group_by([
            pl.col("time_stamp").dt.year().alias("year"),
            pl.col("time_stamp").dt.week().alias("week")
        ]).agg(
            agg_type(with_battery_column).alias("Hinta akulla")
        ).with_columns(
            (pl.col("year").cast(str) + "-W" + pl.col("week").cast(str).str.zfill(2)).alias("Viikko")
        ).sort("Viikko")
        chart_data = reserve_data.join(battery_data, on="Viikko", how="full")
        time_label = "viikottain"
        x_col = "Viikko"
    
    chart_data = chart_data.with_columns([
        ((pl.col("Hinta ilman akkua") - pl.col("Reservimarkkinoiden tuotto")) * 0.01).alias("Reservimarkkinoiden tuotto"),
        (pl.col("Hinta ilman akkua") * 0.01).alias("Hinta ilman akkua"),
        (pl.col("Hinta akulla") * 0.01).alias("Hinta akulla")
    ])
    
    y_max = max(chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla"), pl.col("Reservimarkkinoiden tuotto")]).max().to_numpy().max() * 1.1, battery_price * 1.1) if show_price_line else chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla"), pl.col("Reservimarkkinoiden tuotto")]).max().to_numpy().max() * 1.1
    y_min = chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla"), pl.col("Reservimarkkinoiden tuotto")]).min().to_numpy().min() * 1.1 if chart_data.select([pl.col("Hinta ilman akkua"), pl.col("Hinta akulla"), pl.col("Reservimarkkinoiden tuotto")]).min().to_numpy().min() < 0 else 0
    
    chart_data_pd = chart_data.select([x_col, "Hinta ilman akkua", "Hinta akulla", "Reservimarkkinoiden tuotto"]).to_pandas()
    
    fig = px.line(
        chart_data_pd,
        x=x_col,
        y=["Hinta ilman akkua", "Hinta akulla", "Reservimarkkinoiden tuotto"],
        title=f"Kustannusvertailu esitettynä {time_label}",
        labels={
            x_col: "Aika",
            "value": "Hinta (€)",
            "variable": "Tyyppi"
        }
    )
    fig.update_traces(mode="lines", hovertemplate="%{data.name}: %{y:.2f}€<extra></extra>")
    fig.data[0].line.color = "red"
    fig.data[1].line.color = "blue"
    fig.data[2].line.color = "green"
    
    if show_price_line:
        fig.add_hline(
            y=battery_price,
            line_dash="dash",
            line_color="orange",
            annotation_text="Akun hinta",
            annotation_position="top left"
        )
    
    fig.update_layout(
        xaxis_title="Aika",
        yaxis_title="Hinta (€)",
        legend_title="Tyyppi",
        hovermode="x unified",
        dragmode="pan",
        height=500,
        yaxis=dict(range=[y_min, y_max], fixedrange=True),
        xaxis=dict(fixedrange=False)
    )
    
    config = get_plotly_config()

    fig1 = px.bar(
        chart_data_pd,
        x=x_col,
        y=["Hinta ilman akkua", "Hinta akulla", "Reservimarkkinoiden tuotto"],
        barmode="group")
    
    fig1.update_traces(hovertemplate="%{data.name}: %{y:.2f}€<extra></extra>")

    fig1.update_layout(
        xaxis_title="Aika",
        yaxis_title="Hinta (€)",
        legend_title="Tyyppi",
        hovermode="x unified",
        dragmode="pan",
        yaxis=dict(range=[y_min, y_max], fixedrange=True),
        xaxis=dict(fixedrange=False)
    )
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig, use_container_width=True, config=config)
    with col2:
        st.plotly_chart(fig1, use_container_width=True, config=config)


def main():
    """Main application function with tabs."""
    st.set_page_config(layout="wide")
    st.header("Lähetä CSV-tiedosto kulutuksestasi")
    st.subheader("Tarkista, että tiedostossa on aikaleima ensimmäisessä sarakkeessa ja kulutus toisessa sarakkeessa.")
    consumption_file = st.file_uploader("Valitse CSV-tiedosto", type="csv")
    
    if consumption_file is not None:
        merged_df, consumption_df = load_data(consumption_file)
        
        if merged_df is not None and consumption_df is not None:
            tab1, tab2, tab3 = st.tabs(["Kulutustiedot", "Akun vaikutus", "Reservimarkkinat"])
            
            with tab1:
                st.header("Kulutustiedot")
                display_consumption_metrics(consumption_df, merged_df)
                create_consumption_chart(consumption_df)
            
            with tab2:
                st.header("Akun vaikutus kustannuksiin")
                st.subheader("Akun ominaisuudet")
                max_capacity, max_charge_rate, battery_price = get_basic_battery_parameters()
                basic_battery_df = simulate_basic_battery(merged_df, max_capacity, max_charge_rate)
                display_basic_battery_metrics(basic_battery_df)
                create_basic_battery_chart(basic_battery_df, battery_price)
            
            with tab3:
                st.header("Reservimarkkinat")
                st.subheader("Reservikapasiteetti")
                reserve_capacity = get_reserve_battery_parameters(max_capacity)
                reserve_battery_df = simulate_reserve_battery(merged_df, max_capacity, max_charge_rate, reserve_capacity)
                display_reserve_battery_metrics(reserve_battery_df)
                create_reserve_battery_chart(reserve_battery_df, basic_battery_df, battery_price)
                check = st.expander(label="Tarkista simulaation laskelmat taulukosta", expanded=False)
                with check:
                    st.dataframe(reserve_battery_df)
                st.subheader("Reservimarkkinan tuntihinnat")
                create_reserve_prices_chart(merged_df)


if __name__ == "__main__":
    main()
