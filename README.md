import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bars

# Manual inputs
provided_hashrate = 100 * 10**12  # Provided hashrate in TH/s converted to H/s
power_consumption = 3000  # Power consumption in Watts
risk_free_rate = 0.03  # Risk-free rate
block_reward = 3.125  # Block reward in BTC
mining_pool_fee = 0.02  # Mining pool fee as a percentage
transaction_fee_reward = 0.08  # Transaction fee reward as a percentage

# Helper function to calculate the NPV of future net incomes from mining, considering BTC price, electricity price, hashrate, and exchange rate.
def calculate_exercise_value(btc_price_path, electricity_price_path, hashrate_path, exchange_rate_path, start_day, end_day):
    days = np.arange(start_day, end_day)
    discount_factors = (1 + risk_free_rate) ** ((days - start_day) / 365)

    # Precompute constants
    constant_factor = 24 * 60 * block_reward * transaction_fee_reward * (1 - mining_pool_fee) / 10
    revenue_per_day_btc = constant_factor / hashrate_path[days]

    revenue_per_day_eur = revenue_per_day_btc * btc_price_path[days] * exchange_rate_path[days]
    electricity_costs_per_day_eur = power_consumption * electricity_price_path[days] * 24 / 1000

    net_income_per_day_eur = (revenue_per_day_eur - electricity_costs_per_day_eur) / discount_factors
    total_net_income = np.sum(net_income_per_day_eur)

    return total_net_income

# Step 1: BTC Price Simulation
def simulate_btc_prices():
    btc_data = yf.download('BTC-USD', start='2020-01-01', end='2024-05-01')['Close']
    returns = btc_data.pct_change().dropna()
    num_simulations = 10000
    num_days = 609  # Forecast for 2 years
    bootstrap_returns = np.random.choice(returns, size=(num_simulations, num_days), replace=True)
    simulated_price_paths = np.zeros_like(bootstrap_returns)
    simulated_price_paths[:, 0] = btc_data.iloc[-1]

    for t in range(1, num_days):
        simulated_price_paths[:, t] = simulated_price_paths[:, t - 1] * (1 + bootstrap_returns[:, t])

    return simulated_price_paths

# Step 2: Electricity Price Forecast
def forecast_electricity_prices():
    data = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/MFW/Electricity prices - MFW.xlsx')
    data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')
    data.set_index('DATE', inplace=True)
    data_daily = data.resample('D').interpolate('linear')
    log_returns = np.log(data_daily / data_daily.shift(1)).dropna()
    mean_reversion_speeds = log_returns.apply(lambda x: -sm.OLS(x.diff().dropna(), sm.add_constant(x.shift(1).dropna())).fit().params[1])

    def monte_carlo_simulation_with_trend(start_price, daily_returns, num_days, num_simulations, mean_reversion_level, mean_reversion_speed, volatility_scale=3.0, mean_reversion_scale=0.4):
        simulated_prices = np.zeros((num_simulations, num_days))
        simulated_prices[:, 0] = start_price
        volatility = np.std(daily_returns) * volatility_scale

        for t in range(1, num_days):
            random_shocks = np.random.normal(loc=0, scale=volatility, size=num_simulations)
            trend_component = mean_reversion_scale * mean_reversion_speed * (mean_reversion_level - simulated_prices[:, t-1])
            simulated_prices[:, t] = simulated_prices[:, t-1] * (1 + random_shocks) + trend_component

        return simulated_prices

    mean_reversion_level = data_daily.mean().values[0]
    simulated_electricity_paths = monte_carlo_simulation_with_trend(
        data_daily.iloc[-1].values[0], log_returns.values.flatten(), 609, 10000, mean_reversion_level, mean_reversion_speeds.values[0]
    )

    return simulated_electricity_paths

# Step 3: Hashrate Forecast
def forecast_hashrate():
    data2 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/MFW/BitcoinVisuals.com_chart_Hash_per_s.csv')
    data2 = data2[['DateTime', 'Hash Rate']]
    data2.dropna(subset=['Hash Rate'], inplace=True)
    data2['DateTime'] = data2['DateTime'].str.split(' ').str[0]
    data2['DateTime'] = pd.to_datetime(data2['DateTime'], format='%m/%d/%Y')
    data2.set_index('DateTime', inplace=True)
    three_years_ago = data2.index.max() - pd.DateOffset(years=3)
    historical_data = data2[data2.index >= three_years_ago].copy()
    historical_data['Smoothed Hash Rate'] = historical_data['Hash Rate'].rolling(window=7).mean()

    model = ExponentialSmoothing(historical_data['Hash Rate'], trend='mul', seasonal='mul', seasonal_periods=365)
    model_fit = model.fit()
    forecast_days = 609  # From 01-May-2024 to 31-Dec-2025
    forecast = model_fit.forecast(steps=forecast_days)
    return forecast

# Step 4: Exchange Rate Interpolation
def interpolate_exchange_rate():
    data = {
        'Tenor': ['SPOT', '1W', '2W', '1M', '2M', '3M', '4M', '5M', '6M', '9M', '12M', '18M'],
        'Bid': ['0.9224', '0.9222', '0.9219', '0.9212', '0.9198', '0.9185', '0.9172', '0.9158', '0.9145', '0.9103', '0.9064', '0.8990']
    }

    df = pd.DataFrame(data)
    df['Bid'] = df['Bid'].astype(float)
    current_date = datetime(2024, 5, 1)

    tenor_to_days = {
        'SPOT': 0,
        '1W': 7,
        '2W': 14,
        '1M': 30,
        '2M': 60,
        '3M': 90,
        '4M': 120,
        '5M': 150,
        '6M': 180,
        '9M': 270,
        '12M': 360,
        '18M': 540
    }

    df['Days'] = df['Tenor'].map(tenor_to_days)
    df['Date'] = df['Days'].apply(lambda x: current_date + timedelta(days=x))

    end_date = datetime(2025, 12, 31)
    num_days = (end_date - current_date).days
    interpolated_dates = [current_date + timedelta(days=i) for i in range(num_days)]

    bid_interp = interp1d(df['Days'], df['Bid'], kind='linear', fill_value='extrapolate')
    interpolated_days = np.arange(0, num_days)
    interpolated_bid = bid_interp(interpolated_days)

    return interpolated_bid

# Step 5: Calculate NPV of Net Incomes for each path
def calculate_all_exercise_values(simulated_btc_prices, simulated_electricity_prices, hashrate_forecast, exchange_rate_forecast):
    num_simulations, num_days = simulated_btc_prices.shape
    all_exercise_values = np.zeros((num_simulations, num_days))
    for i in tqdm(range(num_simulations), desc="Calculating NPV for all paths", unit="simulation"):
        for t in range(num_days):
            all_exercise_values[i, t] = calculate_exercise_value(
                simulated_btc_prices[i], simulated_electricity_prices[i], hashrate_forecast, exchange_rate_forecast, t, num_days
            )
    return all_exercise_values

# Step 6: Implement the Longstaff and Schwartz algorithm
def longstaff_schwartz_valuation(all_exercise_values):
    num_simulations, num_days = all_exercise_values.shape
    option_values = np.zeros((num_simulations, num_days))

    # Ensure non-negative exercise values
    exercise_values = np.maximum(all_exercise_values, 0)

    # Backward induction
    exercise_points = []
    for t in tqdm(range(num_days - 1, -1, -1), desc="Backward Induction"):
        if t == num_days - 1:
            option_values[:, t] = exercise_values[:, t]
        else:
            continuation_values = option_values[:, t + 1] * np.exp(-risk_free_rate / 365)  # Discount factor
            exercise_flags = exercise_values[:, t] > continuation_values

            X = np.column_stack((np.ones(num_simulations), exercise_values[:, t]))
            Y = continuation_values

            model = np.linalg.lstsq(X, Y, rcond=None)[0]
            regression_values = X @ model

            option_values[:, t] = np.where(exercise_flags, exercise_values[:, t], regression_values)
            if np.any(exercise_flags):
                exercise_points.append((t, exercise_flags))

    # Ensure non-negative option values
    option_values = np.maximum(option_values, 0)

    # Calculate how often the option would be exercised
    exercised_paths = np.any(exercise_values > option_values, axis=1)
    exercise_count = np.sum(exercised_paths)
    non_exercise_count = num_simulations - exercise_count
    print(f"The option would be exercised in {exercise_count} out of {num_simulations} simulations.")
    print(f"The option would not be exercised in {non_exercise_count} out of {num_simulations} simulations.")

    return option_values[:, 0].mean(), exercised_paths, exercise_points

# Run the simulations and valuation
simulated_btc_prices = simulate_btc_prices()
simulated_electricity_prices = forecast_electricity_prices()
hashrate_forecast = forecast_hashrate()
exchange_rate_forecast = interpolate_exchange_rate()

print(f"Shape of simulated BTC prices: {simulated_btc_prices.shape}")
print(f"Shape of simulated electricity prices: {simulated_electricity_prices.shape}")

all_exercise_values = calculate_all_exercise_values(simulated_btc_prices, simulated_electricity_prices, hashrate_forecast, exchange_rate_forecast)
option_value, exercised_paths, exercise_points = longstaff_schwartz_valuation(all_exercise_values)
print(f"The value of the American call option is: {option_value}")

# Identify paths that led to early exercise
exercised_indices = np.where(exercised_paths)[0]

# Calculate net income for each path that led to early exercise
net_incomes = []
for i in tqdm(exercised_indices, desc="Calculating Net Incomes"):
    path_net_income = []
    for t in range(simulated_btc_prices.shape[1]):
        net_income = calculate_exercise_value(simulated_btc_prices[i], simulated_electricity_prices[i], hashrate_forecast, exchange_rate_forecast, t, simulated_btc_prices.shape[1])
        path_net_income.append(net_income)
    net_incomes.append(path_net_income)

# Plot net income for paths that led to early exercise
plt.figure(figsize=(14, 7))
for path_net_income in net_incomes:
    plt.plot(path_net_income, linestyle='-', alpha=0.7)
plt.title('Net Income Paths for Early Exercise Scenarios')
plt.xlabel('Days')
plt.ylabel('Net Income (EUR)')
plt.show()
