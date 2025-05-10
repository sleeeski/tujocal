import streamlit as st
import polars as pl
import numpy as np
import plotly.express as px
from datetime import timedelta
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD

# Set page configuration
st.set_page_config(page_title="Battery Savings Simulator", layout="wide")

# Initialize session state for file uploads and calculations
if 'uploaded_file_tab1' not in st.session_state:
    st.session_state.uploaded_file_tab1 = None
if 'uploaded_file_tab2' not in st.session_state:
    st.session_state.uploaded_file_tab2 = None
if 'uploaded_file_tab3' not in st.session_state:
    st.session_state.uploaded_file_tab3 = None
if 'savings_results' not in st.session_state:
    st.session_state.savings_results = None
if 'earnings_results' not in st.session_state:
    st.session_state.earnings_results = None
if 'interval_tab2' not in st.session_state:
    st.session_state.interval_tab2 = "Päivä"
if 'interval_tab3' not in st.session_state:
    st.session_state.interval_tab3 = "Päivä"

def load_data(file):
    try:
        df = pl.read_csv(file)
        # Validate that CSV has at least three columns
        if len(df.columns) < 3:
            st.error("CSV must have at least three columns in order: timestamp, spot price, consumption")
            return None
        # Select columns by position and rename them
        df = df.select([
            pl.col(df.columns[0]).alias("time_stamp"),
            pl.col(df.columns[1]).alias("spot_price"),
            pl.col(df.columns[2]).alias("consumption")
        ])
        # Convert time_stamp to datetime with explicit format, allowing nulls for unparseable values
        df = df.with_columns(
            pl.col('time_stamp').str.to_datetime(format="%Y-%m-%d %H:%M", strict=False).alias("time_stamp")
        )
        # Check for null timestamps and provide feedback
        if df['time_stamp'].is_null().any():
            invalid_timestamps = df.filter(pl.col('time_stamp').is_null()).select('time_stamp').to_series().to_list()
            # Filter out rows with null timestamps
            df = df.filter(pl.col('time_stamp').is_not_null())
            if df.is_empty():
                st.error("No valid timestamps remain after filtering. Please ensure timestamps are in YYYY-MM-DD HH:MM format.")
                return None
        # Ensure spot_price and consumption are numeric and finite
        df = df.with_columns([
            pl.col('spot_price').cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col('consumption').cast(pl.Float64, strict=False).fill_null(0.0)
        ])
        # Check for non-finite values
        if df['spot_price'].is_nan().any() or df['spot_price'].is_infinite().any():
            st.error("Spot price column contains invalid (NaN or infinite) values")
            return None
        if df['consumption'].is_nan().any() or df['consumption'].is_infinite().any():
            st.error("Consumption column contains invalid (NaN or infinite) values")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def load_reserve_data(file):
    try:
        # Load CSV with semicolon delimiter, double-quoted values, and try parsing dates
        df = pl.read_csv(file, separator=';', try_parse_dates=True)
        if len(df.columns) < 3:
            st.error("CSV must have at least three columns: startTime, endTime, reserve price")
            return None
        
        # Select and rename columns
        df = df.select([
            pl.col('startTime').alias("time_stamp"),
            pl.col('Taajuusohjattu käyttöreservi, tuntimarkkinahinnat').alias("reserve_price")
        ])
        
        # Ensure time_stamp is datetime (Polars should parse YYYY-MM-DDTHH:MM:SS.000Z automatically)
        try:
            df = df.with_columns(
                pl.col('time_stamp').cast(pl.Datetime).alias("time_stamp")
            )
        except Exception as e:
            st.error(f"Error parsing timestamps (expected format: YYYY-MM-DDTHH:MM:SS.000Z): {e}")
            return None
        
        # Cast reserve_price to float
        df = df.with_columns(
            pl.col('reserve_price').cast(pl.Float64, strict=False).fill_null(0.0)
        )
        
        # Validate data
        if df['reserve_price'].is_nan().any() or df['reserve_price'].is_infinite().any():
            st.error("Reserve price column contains invalid (NaN or infinite) values")
            return None
        if df['time_stamp'].is_null().any():
            st.error("Some timestamps could not be parsed or are missing")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def calculate_stats(df):
    # Average hourly consumption
    avg_hourly = df['consumption'].mean()
    
    # Daily consumption
    daily = df.group_by(df['time_stamp'].dt.date()).agg(pl.col('consumption').sum())
    avg_daily = daily['consumption'].mean()
    
    # Monthly consumption
    monthly = df.group_by(df['time_stamp'].dt.truncate('1mo')).agg(pl.col('consumption').sum())
    avg_monthly = monthly['consumption'].mean()
    
    return avg_hourly, avg_daily, avg_monthly

def get_plotly_config():
    """Return standardized config for Plotly charts."""
    return {
        "scrollZoom": True,  # Enable zooming with scroll wheel
        "displayModeBar": True,
        "modeBarButtonsToRemove": [
            "zoom2d", "pan2d", "select2d", "lasso2d",
            "zoomIn", "zoomOut", "autoScale2d", "toImage"
        ]
    }

def prepare_graph_data(df, interval, data_column='consumption', cumulative=False):
    if interval == "Päivä":
        # Aggregate by day
        chart_data = df.group_by(pl.col("time_stamp").dt.date().alias("date")).agg(
            pl.col(data_column).sum().alias(data_column)
        ).sort("date")
        # Ensure all dates are present
        min_date = chart_data['date'].min()
        max_date = chart_data['date'].max()
        all_dates = pl.date_range(min_date, max_date, interval="1d", eager=True)
        chart_data = pl.DataFrame({"date": all_dates}).join(
            chart_data, on="date", how="left"
        ).fill_null(0)  # Fill missing data with 0
        x_col = "date"
        time_label = "päivittäin"
    else:  # Kuukausi
        # Aggregate by month
        chart_data = df.group_by(pl.col("time_stamp").dt.truncate("1mo").alias("date")).agg(
            pl.col(data_column).sum().alias(data_column)
        ).sort("date")
        # Ensure all months are present
        min_date = chart_data['date'].min()
        max_date = chart_data['date'].max()
        all_dates = pl.date_range(min_date, max_date, interval="1mo", eager=True)
        # Create a DataFrame with all months and their string representation
        all_dates_df = pl.DataFrame({"date": all_dates}).with_columns(
            pl.col("date").dt.strftime("%Y-%m").alias("date_str")
        )
        # Join with chart_data, using string-based join to avoid datetime issues
        chart_data = all_dates_df.join(
            chart_data.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("date_str")),
            on="date_str",
            how="left"
        ).select([
            pl.col("date"),
            pl.col("date_str"),
            pl.col(data_column).fill_null(0)
        ])
        x_col = "date_str"
        time_label = "kuukausittain"
    
    # Apply cumulative sum if requested
    if cumulative:
        chart_data = chart_data.with_columns(
            pl.col(data_column).cum_sum().alias(data_column)
        )
    
    return chart_data, x_col, time_label

def optimize_battery_savings(df, capacity, power):
    # Validate input data
    if df.is_empty():
        st.error("Input data is empty")
        return None, None
    
    # Convert Polars DataFrame to lists for PuLP
    prices = df['spot_price'].to_list()  # in snt/kWh (cents per kWh)
    consumption = df['consumption'].to_list()  # in kWh
    n = len(df)
    
    # Check for non-finite values in prices and consumption
    if any(not np.isfinite(p) for p in prices):
        st.error("Spot prices contain invalid values")
        return None, None
    if any(not np.isfinite(c) for c in consumption):
        st.error("Consumption values contain invalid values")
        return None, None
    
    # Create the LP problem
    model = LpProblem("Battery_Optimization", LpMinimize)
    
    # Variables
    # Battery state of charge at each time step (kWh)
    soc = [LpVariable(f"soc_{i}", 0, capacity) for i in range(n)]
    # Energy charged to battery (kWh)
    charge = [LpVariable(f"charge_{i}", 0, power) for i in range(n)]
    # Energy discharged from battery (kWh)
    discharge = [LpVariable(f"discharge_{i}", 0, power) for i in range(n)]
    
    # Objective: Minimize total cost
    # Cost = (consumption - discharge + charge) * price (in cents)
    model += lpSum([(consumption[i] - discharge[i] + charge[i]) * prices[i] for i in range(n)])
    
    # Constraints
    for i in range(n):
        if i == 0:
            # Initial state of charge
            model += soc[0] == charge[0] - discharge[0]
        else:
            # State of charge update
            model += soc[i] == soc[i-1] + charge[i] - discharge[i]
    
    # Solve the problem
    solver = PULP_CBC_CMD(msg=False)
    model.solve(solver)
    
    # Check if solution was found
    if model.status != 1:
        st.error("Optimization failed to find a solution")
        return None, None
    
    # Extract results
    net_consumption = []
    for i in range(n):
        net = consumption[i] - discharge[i].varValue + charge[i].varValue
        net_consumption.append(net)
    
    # Calculate costs (in cents)
    cost_with_battery = sum(net * price for net, price in zip(net_consumption, prices))
    cost_without_battery = sum(c * price for c, price in zip(consumption, prices))
    savings = cost_without_battery - cost_with_battery  # in cents
    
    # Convert total savings to euros for display
    savings_eur = savings / 100.0
    
    # Add savings to DataFrame (savings in euros per hour)
    df_with_savings = df.with_columns(
        pl.Series("net_consumption", net_consumption),
        pl.Series("savings", [((c - nc) * p) / 100.0 for c, nc, p in zip(consumption, net_consumption, prices)])
    )
    
    return savings_eur, df_with_savings

def calculate_periodic_metrics(df, total_amount):
    # Calculate the total duration of the data in hours
    time_diff = df['time_stamp'].max() - df['time_stamp'].min()
    total_hours = time_diff.total_seconds() / 3600 if time_diff is not None else len(df)
    
    # Handle case where data is less than an hour
    if total_hours < 1:
        total_hours = len(df)  # Assume each row is one hour if time stamps are missing or too close
    
    # Calculate amount rate (euros per hour)
    amount_per_hour = total_amount / total_hours if total_hours > 0 else 0
    
    # Scale to different time periods
    # Daily: amount_per_hour * 24 hours
    avg_daily = amount_per_hour * 24
    
    # Monthly: amount_per_hour * 24 hours * 30.42 days (average days per month)
    avg_monthly = amount_per_hour * 24 * 30.42
    
    # Yearly: amount_per_hour * 24 hours * 365 days
    avg_yearly = amount_per_hour * 24 * 365
    
    return avg_daily, avg_monthly, avg_yearly

def calculate_reserve_earnings(df, power):
    # Convert power from kW to MW
    power_mw = power / 1000.0
    
    # Calculate earnings per hour: power (MW) * reserve price (€/MW per hour)
    df = df.with_columns(
        pl.col('reserve_price').mul(power_mw).alias('earnings')
    )
    
    # Total earnings over the data period
    total_earnings = df['earnings'].sum()
    
    return total_earnings, df

# Create tabs
tab1, tab2, tab3 = st.tabs(["Kulutustiedot", "Akun vaikutus", "Reservimarkkinat"])

with tab1:
    st.header("Kulutustiedot")
    
    # File uploader
    uploaded_file = st.file_uploader("Lähetä CSV-tiedosto kulutustiedoista", type=["csv"], key="tab1_uploader")
    st.markdown("Varmista, että CSV-tiedostossa on sarakkeet: aika, spot-hinta ja kulutus erotettuina pilkuilla. VÄE:n antamat datat toimivat hyvin.")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_tab1 = uploaded_file
        df = load_data(uploaded_file)
        
        if df is not None:
            # Calculate statistics
            avg_hourly, avg_daily, avg_monthly = calculate_stats(df)
            
            # Display statistics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Keskim. kulutus tunnissa", f"{avg_hourly:.2f} kWh")
            with col2:
                st.metric("Keskim. kulutus päivässä", f"{avg_daily:.2f} kWh")
            with col3:
                st.metric("Keskim. kulutus kuukaudessa", f"{avg_monthly:.2f} kWh")
            
            # Dropdown for time interval
            st.subheader("Kulutus aikajaksolla")
            interval = st.selectbox("Valitse aikaväli kaavioon", ["Päivä", "Kuukausi"], index=0, key="tab1_interval")
            
            # Prepare and display Plotly graph
            chart_data, x_col, time_label = prepare_graph_data(df, interval, data_column='consumption')
            chart_data_pd = chart_data.select([x_col, "consumption"]).to_pandas()
            y_max = chart_data["consumption"].max() * 1.1 if chart_data["consumption"].max() > 0 else 1.0
            
            fig = px.line(
                chart_data_pd,
                x=x_col,
                y=["consumption"],
                title=f"Kulutus {time_label}",
                labels={
                    x_col: "Aika",
                    "value": "Kulutus (kWh)",
                    "variable": "Type"
                }
            )
            fig.update_traces(mode="lines", hovertemplate="Kulutus: %{y:.2f} kWh<extra></extra>")
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
            
            config = get_plotly_config()
            st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.info("Lähetä CSV-tiedosto tarkastellaksesi kulutustietoja.")

with tab2:
    st.header("Akun vaikutus kustannuksiin")
    
    # File uploader
    uploaded_file = st.file_uploader("Lähetä CSV-tiedosto kulutustiedoista", type=["csv"], key="tab2_uploader")
    st.markdown("Varmista, että CSV-tiedostossa on sarakkeet: aika, spot-hinta ja kulutus erotettuina pilkuilla. VÄE:n antamat datat toimivat hyvin.")
    
    if uploaded_file is not None:
        st.session_state.uploaded_file_tab2 = uploaded_file
        df = load_data(uploaded_file)
        
        if df is not None:
            # Battery parameters input
            st.subheader("Akun ominaisuudet")
            col1, col2, col3 = st.columns(3)
            with col1:
                capacity = st.number_input("Valitse akun kapasiteetti (kWh)", min_value=0.0, value=10.0, step=1.0)
            with col2:
                power = st.number_input("Valitse akun teho (kW)", min_value=0.0, value=5.0, step=1.0)
            with col3:
                price = st.number_input("Valitse akun hinta (€)", min_value=0.0, value=10000.0, step=100.0)
            
            if st.button("Laske säästöt"):
                # Optimize battery savings
                total_savings, df_with_savings = optimize_battery_savings(df, capacity, power)
                # Store results in session state
                st.session_state.savings_results = {
                    'total_savings': total_savings,
                    'df_with_savings': df_with_savings
                }
            
            # Display results if available
            if st.session_state.savings_results is not None:
                total_savings = st.session_state.savings_results['total_savings']
                df_with_savings = st.session_state.savings_results['df_with_savings']
                
                if total_savings is not None and df_with_savings is not None:
                    # Calculate savings statistics
                    avg_daily_savings, avg_monthly_savings, avg_yearly_savings = calculate_periodic_metrics(df_with_savings, total_savings)
                    
                    # Calculate payback period
                    payback_period = price / avg_yearly_savings if avg_yearly_savings > 0 else float('inf')
                    
                    # Display statistics in columns
                    st.subheader("Säästöt")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Keskim. säästöt päivässä", f"{avg_daily_savings:.2f} €")
                    with col2:
                        st.metric("Keskim. säästöt kuukaudessa", f"{avg_monthly_savings:.2f} €")
                    with col3:
                        st.metric("Keskim. säästöt vuodessa", f"{avg_yearly_savings:.2f} €")
                    with col4:
                        st.metric("Kokonaissäästöt", f"{total_savings:.2f} €")
                    
                    # Display payback period with subheader
                    st.subheader("Takaisinmaksuaika")
                    if payback_period != float('inf'):
                        st.markdown(f"Takaisinmaksu aika akullesi kulutusoptimoinnilla on **{payback_period:.2f} vuotta**.")
                    else:
                        st.markdown("Takaisinmaksu aika akullesi kulutusoptimoinnilla on **N/A** (ei säästöja).")
                    
                    # Dropdown for time interval
                    st.subheader("Säästöt aikajaksolla")
                    interval = st.selectbox(
                        "Valitse aikaväli kaavioon",
                        ["Päivä", "Kuukausi"],
                        index=0 if st.session_state.interval_tab2 == "Päivä" else 1,
                        key="tab2_interval"
                    )
                    st.session_state.interval_tab2 = interval
                    
                    # Checkbox for cumulative graph
                    cumulative = st.checkbox("Näytä kumulatiiviset säästöt", key="tab2_cumulative")
                    
                    # Prepare and display Plotly graph
                    chart_data, x_col, time_label = prepare_graph_data(
                        df_with_savings, interval, data_column='savings', cumulative=cumulative
                    )
                    chart_data_pd = chart_data.select([x_col, "savings"]).to_pandas()
                    y_max = chart_data["savings"].max() * 1.1 if chart_data["savings"].max() > 0 else 1.0
                    
                    title = f"Kumulatiiviset Säästöt {time_label}" if cumulative else f"Säästöt {time_label}"
                    y_label = "Kumulatiiviset Säästöt (€)" if cumulative else "Säästöt (€)"
                    hover_template = "Kumulatiiviset Säästöt: %{y:.2f} €<extra></extra>" if cumulative else "Säästöt: %{y:.2f} €<extra></extra>"
                    
                    fig = px.line(
                        chart_data_pd,
                        x=x_col,
                        y=["savings"],
                        title=title,
                        labels={
                            x_col: "Aika",
                            "value": y_label,
                            "variable": "Type"
                        }
                    )
                    fig.update_traces(mode="lines", hovertemplate=hover_template)
                    fig.update_layout(
                        xaxis_title="Aika",
                        yaxis_title=y_label,
                        hovermode="x unified",
                        dragmode="pan",
                        height=500,
                        yaxis=dict(range=[0, y_max], fixedrange=True),
                        xaxis=dict(fixedrange=False),
                        showlegend=False
                    )
                    
                    config = get_plotly_config()
                    st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.info("Lähetä CSV-tiedosto tarkastellaksesi säästötietoja.")

with tab3:
    st.header("Reservituottolaskuri")
    
    # File uploader
    uploaded_file = st.file_uploader("Lähetä CSV-tiedosto reservihinnoista (€/MW)", type=["csv"], key="tab3_uploader")
    st.markdown("Varmista, että CSV-tiedosto on samassa muodossa kuin Fingridin avoimen datan sivuston tiedostot. CSV-tiedosto kannattaa olla ladattu juuri Fingridin sivustolta.")
    if uploaded_file is not None:
        st.session_state.uploaded_file_tab3 = uploaded_file
        df = load_reserve_data(uploaded_file)
        
        if df is not None:
            # Battery parameters input
            st.subheader("Akun ominaisuudet")
            col1, col2 = st.columns(2)
            with col1:
                power = st.number_input("Valitse akun teho (kW)", min_value=0.0, value=5.0, step=1.0, key="tab3_power")
            with col2:
                price = st.number_input("Valitse akun hinta (€)", min_value=0.0, value=10000.0, step=100.0, key="tab3_price")
            
            if st.button("Laske reservituotot", key="tab3_calculate"):
                # Calculate reserve earnings
                total_earnings, df_with_earnings = calculate_reserve_earnings(df, power)
                # Store results in session state
                st.session_state.earnings_results = {
                    'total_earnings': total_earnings,
                    'df_with_earnings': df_with_earnings
                }
            
            # Display results if available
            if st.session_state.earnings_results is not None:
                total_earnings = st.session_state.earnings_results['total_earnings']
                df_with_earnings = st.session_state.earnings_results['df_with_earnings']
                
                if total_earnings is not None and df_with_earnings is not None:
                    # Calculate earnings statistics
                    avg_daily_earnings, avg_monthly_earnings, avg_yearly_earnings = calculate_periodic_metrics(df_with_earnings, total_earnings)
                    
                    # Calculate payback period
                    payback_period = price / avg_yearly_earnings if avg_yearly_earnings > 0 else float('inf')
                    
                    # Display statistics in columns
                    st.subheader("Tuottotilastot")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Keskim. reservituotot päivässä", f"{avg_daily_earnings:.2f} €")
                    with col2:
                        st.metric("Keskim. reservituotot kuukaudessa", f"{avg_monthly_earnings:.2f} €")
                    with col3:
                        st.metric("Keskim. reservituotot vuodessa", f"{avg_yearly_earnings:.2f} €")
                    
                    # Display payback period with subheader
                    st.subheader("Takaisinmaksuaika")
                    if payback_period != float('inf'):
                        st.markdown(f"Takaisinmaksuaika akullesi reservimarkkinoilla on **{payback_period:.2f} vuotta**.")
                    else:
                        st.markdown("Takaisinmaksuaika akullesi reservimarkkinoilla on **N/A** (ei tuottoja).")
                    
                    # Dropdown for time interval
                    st.subheader("Tuotot aikajaksolla")
                    interval = st.selectbox(
                        "Valitse aikaväli kaavioon",
                        ["Päivä", "Kuukausi"],
                        index=0 if st.session_state.interval_tab3 == "Päivä" else 1,
                        key="tab3_interval"
                    )
                    st.session_state.interval_tab3 = interval
                    
                    # Checkbox for cumulative graph
                    cumulative = st.checkbox("Näytä kumulatiiviset tuotot", key="tab3_cumulative")
                    
                    # Prepare and display Plotly graph
                    chart_data, x_col, time_label = prepare_graph_data(
                        df_with_earnings, interval, data_column='earnings', cumulative=cumulative
                    )
                    chart_data_pd = chart_data.select([x_col, "earnings"]).to_pandas()
                    y_max = chart_data["earnings"].max() * 1.1 if chart_data["earnings"].max() > 0 else 1.0
                    
                    title = f"Kumulatiiviset Tuotot {time_label}" if cumulative else f"Tuotot {time_label}"
                    y_label = "Kumulatiiviset Tuotot (€)" if cumulative else "Tuotot (€)"
                    hover_template = "Kumulatiiviset Tuotot: %{y:.2f} €<extra></extra>" if cumulative else "Tuotot: %{y:.2f} €<extra></extra>"
                    
                    fig = px.line(
                        chart_data_pd,
                        x=x_col,
                        y=["earnings"],
                        title=title,
                        labels={
                            x_col: "Aika",
                            "value": y_label,
                            "variable": "Type"
                        }
                    )
                    fig.update_traces(mode="lines", hovertemplate=hover_template)
                    fig.update_layout(
                        xaxis_title="Aika",
                        yaxis_title=y_label,
                        hovermode="x unified",
                        dragmode="pan",
                        height=500,
                        yaxis=dict(range=[0, y_max], fixedrange=True),
                        xaxis=dict(fixedrange=False),
                        showlegend=False
                    )
                    
                    config = get_plotly_config()
                    st.plotly_chart(fig, use_container_width=True, config=config)
    else:
        st.info("Lähetä CSV-tiedosto tarkastellaksesi reservituottoja.")
