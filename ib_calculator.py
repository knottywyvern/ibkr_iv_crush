"""
DISCLAIMER: 

This software is provided solely for educational and research purposes. 
It is not intended to provide investment advice, and no investment recommendations are made herein. 
The developers are not financial advisors and accept no responsibility for any financial decisions or losses resulting from the use of this software. 
Always consult a professional financial advisor before making any investment decisions.
"""

import FreeSimpleGUI as sg
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import threading
from ib_insync import IB, Contract, Option, Stock, util
import pandas as pd
import math
import asyncio
import nest_asyncio
import os
import yfinance as yf
from curl_cffi import requests as curl_requests
import utils.styles as styles

# Create a session for yfinance to use
session = curl_requests.Session(impersonate="chrome")

# Apply nest_asyncio to allow for nested event loops (required for GUI + asyncio)
nest_asyncio.apply()

# Apply the Bloomberg-like theme
styles.setup_bloomberg_theme()

# Connect to Interactive Brokers TWS/Gateway
async def connect_to_ib(host='127.0.0.1', port=7497, client_id=1):
    """Connect to Interactive Brokers TWS or Gateway"""
    ib = IB()
    try:
        # Set timeout for IB operations
        ib.RequestTimeout = 10  # 10 second timeout for all requests
        await ib.connectAsync(host, port, clientId=client_id)
        # Ensure we're using the IB event loop for all operations
        await asyncio.sleep(0.1)  # Small sleep to initialize connection properly
        return ib
    except Exception as e:
        raise ConnectionError(f"Failed to connect to IB: {str(e)}")

def filter_dates(dates):
    """Filter option expiration dates to those 45+ days out.
    If no dates meet criteria, returns all available dates instead of raising an error."""
    today = datetime.today().date()
    cutoff_date = today + timedelta(days=45)
    
    sorted_dates = sorted(dates)
    sorted_date_objs = [datetime.strptime(date, "%Y%m%d").date() for date in sorted_dates]

    arr = []
    for i, date in enumerate(sorted_date_objs):
        if date >= cutoff_date:
            arr = sorted_dates[:i+1]
            break
    
    if arr:
        today_str = today.strftime("%Y%m%d")
        if arr[0] == today_str and len(arr) > 1:
            return arr[1:]
        return arr

    # Return all dates if no date meets the criteria instead of raising an error
    return sorted_dates

def yang_zhang(price_data, window=30, trading_periods=252, return_last_only=True):
    log_ho = (price_data['high'] / price_data['open']).apply(np.log)
    log_lo = (price_data['low'] / price_data['open']).apply(np.log)
    log_co = (price_data['close'] / price_data['open']).apply(np.log)
    
    log_oc = (price_data['open'] / price_data['close'].shift(1)).apply(np.log)
    log_oc_sq = log_oc**2
    
    log_cc = (price_data['close'] / price_data['close'].shift(1)).apply(np.log)
    log_cc_sq = log_cc**2
    
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
    
    close_vol = log_cc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    open_vol = log_oc_sq.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    window_rs = rs.rolling(
        window=window,
        center=False
    ).sum() * (1.0 / (window - 1.0))

    k = 0.34 / (1.34 + ((window + 1) / (window - 1)))
    result = (open_vol + k * close_vol + (1 - k) * window_rs).apply(np.sqrt) * np.sqrt(trading_periods)

    if return_last_only:
        return result.iloc[-1]
    else:
        return result.dropna()

def build_term_structure(days, ivs):
    days = np.array(days)
    ivs = np.array(ivs)

    sort_idx = days.argsort()
    days = days[sort_idx]
    ivs = ivs[sort_idx]

    spline = interp1d(days, ivs, kind='linear', fill_value="extrapolate")

    def term_spline(dte):
        if dte < days[0]:  
            return ivs[0]
        elif dte > days[-1]:
            return ivs[-1]
        else:  
            return float(spline(dte))

    return term_spline

def get_stock_contract(symbol):
    """Create a Stock contract object"""
    contract = Stock(symbol, 'SMART', 'USD')
    return contract

async def get_current_price(ib, symbol):
    """Get current price for a stock"""
    contract = get_stock_contract(symbol)
    try:
        await ib.qualifyContractsAsync(contract)
        
        ticker = ib.reqMktData(contract)
        await asyncio.sleep(1)  # Give IB time to return the data
        
        if not math.isnan(ticker.marketPrice()):
            return ticker.marketPrice()
        elif not math.isnan(ticker.last):
            return ticker.last
        else:
            # Try to get the close price from historical data
            bars = await ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            if bars and len(bars) > 0:
                return bars[-1].close
            raise ValueError(f"Unable to get price for {symbol}")
    except Exception as e:
        raise ValueError(f"Error fetching current price: {str(e)}")

async def get_historical_data(ib, symbol, duration='3 M'):
    """Get historical price data for a stock"""
    contract = get_stock_contract(symbol)
    try:
        await ib.qualifyContractsAsync(contract)
        
        bars = await ib.reqHistoricalDataAsync(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        
        if not bars:
            raise ValueError(f"No historical data available for {symbol}")
        
        df = util.df(bars)
        df.columns = [col.lower() for col in df.columns]  # Ensure lowercase column names
        return df
    except Exception as e:
        raise ValueError(f"Error fetching historical data: {str(e)}")

# Function to get average volume from Yahoo Finance
async def get_average_volume(symbol):
    """Get average volume for a stock from Yahoo Finance"""
    try:
        # Create a ticker object for the symbol
        stock = yf.Ticker(symbol, session=session)
        
        # Get 3-month history
        price_history = stock.history(period='3mo')
        
        # Calculate 30-day rolling average volume
        avg_volume = price_history['Volume'].rolling(30).mean().dropna().iloc[-1]
        
        return avg_volume
    except Exception as e:
        raise ValueError(f"Error fetching volume data from Yahoo Finance: {str(e)}")

async def get_option_expiration_dates(ib, symbol):
    """Get available option expiration dates within 45 DTE to reduce processing time"""
    try:
        # Create and qualify the contract
        contract = get_stock_contract(symbol)
        await ib.qualifyContractsAsync(contract)
        
        # Request option parameters
        expiry_dates = await ib.reqSecDefOptParamsAsync(
            contract.symbol, 
            '', 
            contract.secType, 
            contract.conId
        )
        
        if not expiry_dates:
            raise ValueError(f"No options data returned for {symbol}")
        
        # Extract unique expiration dates
        all_dates = set()
        for params in expiry_dates:
            for date in params.expirations:
                # Convert to format YYYYMMDD
                all_dates.add(date)
        
        if not all_dates:
            raise ValueError(f"No expiration dates found for {symbol}")
        
        # Filter dates to only include 45 DTE or less
        filtered_dates = []
        today = datetime.today().date()
        max_date = today + timedelta(days=45)
        
        for date_str in all_dates:
            try:
                date_obj = datetime.strptime(date_str, "%Y%m%d").date()
                if today <= date_obj <= max_date:
                    filtered_dates.append(date_str)
            except ValueError:
                continue  # Skip invalid date formats
        
        if not filtered_dates:
            raise ValueError(f"No suitable expiration dates found within 45 DTE for {symbol}")
        
        return sorted(filtered_dates)
    except Exception as e:
        raise ValueError(f"Error fetching option expiration dates: {str(e)}")

async def get_options_chain(ib, symbol, expiry_date, underlying_price):
    """Get options chain for a specific expiration date"""
    try:
        contract = get_stock_contract(symbol)
        await ib.qualifyContractsAsync(contract)
        
        # Get option chain parameters
        chains = await ib.reqSecDefOptParamsAsync(
            contract.symbol, 
            '', 
            contract.secType, 
            contract.conId
        )
        
        if not chains:
            raise ValueError(f"No options chain available for {symbol}")
        
        # Find closest strikes to the current price
        strikes = []
        exchange = None
        for chain in chains:
            if chain.expirations and expiry_date in chain.expirations:
                strikes = [strike for strike in chain.strikes 
                          if 0.8 * underlying_price <= strike <= 1.2 * underlying_price]
                exchange = chain.exchange
                break
        
        if not strikes or not exchange:
            raise ValueError(f"No suitable strikes found for {symbol} with expiry {expiry_date}")
        
        # Find the ATM strike
        atm_strike = min(strikes, key=lambda x: abs(x - underlying_price))
        
        # Create call and put option contracts
        call = Option(symbol, expiry_date, atm_strike, 'C', exchange)
        put = Option(symbol, expiry_date, atm_strike, 'P', exchange)
        
        # Qualify contracts separately and handle any errors
        try:
            await ib.qualifyContractsAsync(call)
        except Exception as e:
            raise ValueError(f"Failed to qualify call option: {str(e)}")
            
        try:
            await ib.qualifyContractsAsync(put)
        except Exception as e:
            raise ValueError(f"Failed to qualify put option: {str(e)}")
        
        # Request option prices using reqMktData
        call_ticker = ib.reqMktData(call, '106', False, False)  # '106' includes implied volatility
        put_ticker = ib.reqMktData(put, '106', False, False)
        
        # Wait for data to arrive
        await asyncio.sleep(2)
        
        # Get option prices
        call_price = None
        put_price = None
        
        # Try to get mid prices first
        if not math.isnan(call_ticker.bid) and not math.isnan(call_ticker.ask):
            call_price = (call_ticker.bid + call_ticker.ask) / 2
        elif not math.isnan(call_ticker.last):
            call_price = call_ticker.last
            
        if not math.isnan(put_ticker.bid) and not math.isnan(put_ticker.ask):
            put_price = (put_ticker.bid + put_ticker.ask) / 2
        elif not math.isnan(put_ticker.last):
            put_price = put_ticker.last
        
        # Calculate IV from straddle price
        call_iv = None
        put_iv = None
        
        # First try to get IV directly from the IB data if available
        try:
            if hasattr(call_ticker, 'modelGreeks') and call_ticker.modelGreeks and hasattr(call_ticker.modelGreeks, 'implVol'):
                call_iv = call_ticker.modelGreeks.implVol
            
            if hasattr(put_ticker, 'modelGreeks') and put_ticker.modelGreeks and hasattr(put_ticker.modelGreeks, 'implVol'):
                put_iv = put_ticker.modelGreeks.implVol
        except Exception:
            pass
        
        # If direct IV isn't available, calculate from straddle price
        if (call_iv is None or put_iv is None) and call_price is not None and put_price is not None:
            # Calculate days to expiration
            exp_date_obj = datetime.strptime(expiry_date, "%Y%m%d").date()
            days_to_expiry = (exp_date_obj - datetime.today().date()).days
            years_to_expiry = days_to_expiry / 365.0
            
            if years_to_expiry > 0:
                # Calculate IV from ATM straddle price
                straddle_price = call_price + put_price
                
                # Simplified IV calculation based on straddle price
                # The straddle price divided by the stock price approximates one standard deviation move
                # IV ≈ (Straddle Price / Stock Price) / sqrt(time to expiry in years)
                straddle_iv = (straddle_price / underlying_price) / math.sqrt(years_to_expiry)
                
                # Apply reasonable bounds to the IV
                straddle_iv = min(max(straddle_iv, 0.01), 2.0)  # Clamp between 1% and 200%
                
                if call_iv is None:
                    call_iv = straddle_iv
                
                if put_iv is None:
                    put_iv = straddle_iv
        
        # Create call dataframe
        call_data = {
            'strike': atm_strike,
            'bid': call_ticker.bid if not math.isnan(call_ticker.bid) else None,
            'ask': call_ticker.ask if not math.isnan(call_ticker.ask) else None,
            'impliedVolatility': call_iv
        }
        
        # Create put dataframe
        put_data = {
            'strike': atm_strike,
            'bid': put_ticker.bid if not math.isnan(put_ticker.bid) else None,
            'ask': put_ticker.ask if not math.isnan(put_ticker.ask) else None,
            'impliedVolatility': put_iv
        }
        
        # Cancel market data to avoid hitting limits
        ib.cancelMktData(call)
        ib.cancelMktData(put)
        
        # Check that we have valid data
        if call_data['impliedVolatility'] is None and put_data['impliedVolatility'] is None:
            raise ValueError("Could not retrieve or calculate implied volatility data")
            
        return pd.DataFrame([call_data]), pd.DataFrame([put_data])
    except Exception as e:
        raise ValueError(f"Error fetching options chain: {str(e)}")

async def compute_recommendation(ib, ticker):
    try:
        ticker = ticker.strip().upper()
        if not ticker:
            return "No stock symbol provided."
        
        try:
            # Check if options are available for this stock
            exp_dates = await get_option_expiration_dates(ib, ticker)
            if not exp_dates:
                raise KeyError("No options found")
        except Exception as e:
            return f"Error: No options found for stock symbol '{ticker}': {str(e)}"
        
        try:
            exp_dates = filter_dates(exp_dates)
        except Exception as e:
            return f"Error: Not enough option data: {str(e)}"
        
        try:
            underlying_price = await get_current_price(ib, ticker)
            if underlying_price is None:
                raise ValueError("No market price found.")
        except Exception as e:
            return f"Error: Unable to retrieve underlying stock price: {str(e)}"
        
        atm_iv = {}
        straddle = None 
        i = 0
        
        for exp_date in exp_dates:
            try:
                calls, puts = await get_options_chain(ib, ticker, exp_date, underlying_price)
                
                if calls.empty or puts.empty:
                    continue
                
                call_iv = calls.loc[0, 'impliedVolatility']
                put_iv = puts.loc[0, 'impliedVolatility']
                
                if call_iv is None or put_iv is None:
                    continue
                
                atm_iv_value = (call_iv + put_iv) / 2.0
                atm_iv[exp_date] = atm_iv_value
                
                if i == 0:
                    call_bid = calls.loc[0, 'bid']
                    call_ask = calls.loc[0, 'ask']
                    put_bid = puts.loc[0, 'bid']
                    put_ask = puts.loc[0, 'ask']
                    
                    if call_bid is not None and call_ask is not None and not math.isnan(call_bid) and not math.isnan(call_ask):
                        call_mid = (call_bid + call_ask) / 2.0
                    else:
                        call_mid = None
                    
                    if put_bid is not None and put_ask is not None and not math.isnan(put_bid) and not math.isnan(put_ask):
                        put_mid = (put_bid + put_ask) / 2.0
                    else:
                        put_mid = None
                    
                    if call_mid is not None and put_mid is not None:
                        straddle = (call_mid + put_mid)
                
                i += 1
            except Exception as e:
                print(f"Error processing expiry {exp_date}: {str(e)}")
                continue
        
        if not atm_iv:
            return "Error: Could not determine ATM IV for any expiration dates."
        
        today = datetime.today().date()
        dtes = []
        ivs = []
        
        for exp_date, iv in atm_iv.items():
            exp_date_obj = datetime.strptime(exp_date, "%Y%m%d").date()
            days_to_expiry = (exp_date_obj - today).days
            dtes.append(days_to_expiry)
            ivs.append(iv)
        
        term_spline = build_term_structure(dtes, ivs)
        
        ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])
        
        price_history = await get_historical_data(ib, ticker)
        iv30_rv30 = term_spline(30) / yang_zhang(price_history)
        
        # Get average volume from Yahoo Finance instead of IB
        avg_volume = await get_average_volume(ticker)
        
        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None
        
        return {
            'avg_volume': avg_volume >= 1500000, 
            'iv30_rv30': iv30_rv30 >= 1.25, 
            'ts_slope_0_45': ts_slope_0_45 <= -0.00406, 
            'expected_move': expected_move
        }
    except Exception as e:
        raise Exception(f'Error occurred processing: {str(e)}')

class AsyncApp:
    def __init__(self):
        self.ib = None
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
    
    async def init_connection(self, host, port, client_id):
        self.ib = await connect_to_ib(host, port, client_id)
        return self.ib is not None and self.ib.isConnected()
    
    async def fetch_option_data_for_date(self, symbol, exp_date, underlying_price, status_callback=None):
        """Helper method to fetch option data for a single expiration date"""
        try:
            if status_callback:
                status_callback(f"Processing expiry {exp_date}...")
            
            calls, puts = await get_options_chain(self.ib, symbol, exp_date, underlying_price)
            
            if calls.empty or puts.empty:
                return (exp_date, None, None, None)
            
            call_iv = calls.loc[0, 'impliedVolatility']
            put_iv = puts.loc[0, 'impliedVolatility']
            
            if call_iv is None or put_iv is None:
                return (exp_date, None, None, None)
            
            atm_iv_value = (call_iv + put_iv) / 2.0
            
            # For first expiration, calculate straddle price
            call_mid = None
            put_mid = None
            
            call_bid = calls.loc[0, 'bid']
            call_ask = calls.loc[0, 'ask']
            put_bid = puts.loc[0, 'bid']
            put_ask = puts.loc[0, 'ask']
            
            if call_bid is not None and call_ask is not None and not math.isnan(call_bid) and not math.isnan(call_ask):
                call_mid = (call_bid + call_ask) / 2.0
            
            if put_bid is not None and put_ask is not None and not math.isnan(put_bid) and not math.isnan(put_ask):
                put_mid = (put_bid + put_ask) / 2.0
            
            straddle = None
            if call_mid is not None and put_mid is not None:
                straddle = (call_mid + put_mid)
            
            return (exp_date, atm_iv_value, straddle, True)
            
        except Exception as e:
            print(f"Error processing expiry {exp_date}: {str(e)}")
            return (exp_date, None, None, None)

    async def analyze_stock(self, symbol, status_callback=None, progress_callback=None):
        """Analyze a stock with progress updates"""
        if not self.ib or not self.ib.isConnected():
            return {'error': 'Not connected to IB'}
        
        result = {}
        
        try:
            # Update progress to 10%
            if progress_callback:
                progress_callback(10)
            if status_callback:
                status_callback("Checking if options are available...")
            
            # Check if options are available for this stock
            try:
                if status_callback:
                    status_callback("Requesting option chain data from IB...")
                exp_dates = await get_option_expiration_dates(self.ib, symbol)
                if not exp_dates:
                    return {'error': "No options found for this stock"}
                if status_callback:
                    status_callback(f"Found {len(exp_dates)} option expiration dates within 45 DTE")
            except Exception as e:
                return {'error': f"No options found: {str(e)}"}
            
            # Update progress to 20%
            if progress_callback:
                progress_callback(20)
            if status_callback:
                status_callback("Continuing with analysis...")
            
            # No need to filter dates here since we're already filtered to 45 DTE in get_option_expiration_dates
            # Try to use farthest date available within the 45 DTE window
            try:
                # Check if we have at least one date
                if not exp_dates:
                    return {'error': "No suitable expiration dates found"}
                
                # Sort dates for consistency
                exp_dates = sorted(exp_dates)
                
            except Exception as e:
                return {'error': f"Error processing expiration dates: {str(e)}"}
            
            # Update progress to 30%
            if progress_callback:
                progress_callback(30)
            if status_callback:
                status_callback("Getting current stock price...")
            
            # Get current price
            try:
                underlying_price = await get_current_price(self.ib, symbol)
                if underlying_price is None:
                    return {'error': "No market price found"}
            except Exception as e:
                return {'error': f"Unable to retrieve stock price: {str(e)}"}
            
            # Update progress to 40%
            if progress_callback:
                progress_callback(40)
            if status_callback:
                status_callback("Processing option chains in parallel...")
            
            # Process option chains in parallel with concurrency limit
            max_date_concurrency = 3  # Maximum number of dates to process concurrently
            atm_iv = {}
            straddle = None
            
            # Process in batches to control concurrency
            for i in range(0, len(exp_dates), max_date_concurrency):
                batch = exp_dates[i:i+max_date_concurrency]
                tasks = []
                
                for exp_date in batch:
                    task = self.fetch_option_data_for_date(symbol, exp_date, underlying_price, status_callback)
                    tasks.append(task)
                
                # Run batch of expiration dates in parallel
                batch_results = await asyncio.gather(*tasks)
                
                # Process results
                for exp_date, atm_iv_value, exp_straddle, success in batch_results:
                    if success and atm_iv_value is not None:
                        atm_iv[exp_date] = atm_iv_value
                        # Use straddle from first expiration only
                        if exp_date == exp_dates[0] and exp_straddle is not None:
                            straddle = exp_straddle
                
                # Update progress incrementally
                if progress_callback:
                    current_progress = 40 + int(((i + len(batch)) / len(exp_dates)) * 30)
                    progress_callback(min(70, current_progress))  # Cap at 70%
            
            if not atm_iv:
                return {'error': "Could not determine ATM IV for any expiration dates"}
            
            # Update progress to 70%
            if progress_callback:
                progress_callback(70)
            if status_callback:
                status_callback("Building volatility term structure...")
            
            # Calculate remaining metrics
            today = datetime.today().date()
            dtes = []
            ivs = []
            
            for exp_date, iv in atm_iv.items():
                exp_date_obj = datetime.strptime(exp_date, "%Y%m%d").date()
                days_to_expiry = (exp_date_obj - today).days
                dtes.append(days_to_expiry)
                ivs.append(iv)
            
            term_spline = build_term_structure(dtes, ivs)
            ts_slope_0_45 = (term_spline(45) - term_spline(dtes[0])) / (45-dtes[0])
            
            # Update progress to 80%
            if progress_callback:
                progress_callback(80)
            if status_callback:
                status_callback("Getting price history...")
            
            price_history = await get_historical_data(self.ib, symbol)
            
            # Update progress to 85%
            if progress_callback:
                progress_callback(85)
            if status_callback:
                status_callback("Getting volume data from Yahoo Finance...")
            
            # Get average volume from Yahoo Finance instead of IB
            avg_volume = await get_average_volume(symbol)
            
            # Update progress to 90%
            if progress_callback:
                progress_callback(90)
            if status_callback:
                status_callback("Calculating volatility metrics...")
            
            # Calculate raw values
            rv30 = yang_zhang(price_history)
            iv30 = term_spline(30)
            iv30_rv30_ratio = iv30 / rv30 if rv30 > 0 else 0
            expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None
            
            # Update progress to 100%
            if progress_callback:
                progress_callback(100)
            if status_callback:
                status_callback("Analysis complete!")
            
            # Return both raw values and pass/fail status
            result = {
                'avg_volume': avg_volume >= 1500000, 
                'iv30_rv30': iv30_rv30_ratio >= 1.25, 
                'ts_slope_0_45': ts_slope_0_45 <= -0.00406, 
                'expected_move': expected_move,
                # Raw values
                'avg_volume_raw': avg_volume,
                'iv30_rv30_raw': iv30_rv30_ratio,
                'iv30_raw': iv30,
                'rv30_raw': rv30,
                'ts_slope_0_45_raw': ts_slope_0_45,
                'current_price': underlying_price
            }
        except Exception as e:
            result = {'error': f"Error occurred: {str(e)}"}
        
        return result
    
    async def analyze_stocks_batch(self, symbols, max_concurrent=5, status_callbacks=None, progress_callbacks=None):
        """Process multiple stocks concurrently with a limit on concurrency"""
        if not self.ib or not self.ib.isConnected():
            return [{'ticker': symbol, 'error': 'Not connected to IB'} for symbol in symbols]
        
        results = []
        
        # Process in batches to control concurrency
        for i in range(0, len(symbols), max_concurrent):
            batch = symbols[i:i+max_concurrent]
            batch_tasks = []
            
            for j, symbol in enumerate(batch):
                idx = i + j
                status_callback = status_callbacks[idx] if status_callbacks and idx < len(status_callbacks) else None
                progress_callback = progress_callbacks[idx] if progress_callbacks and idx < len(progress_callbacks) else None
                
                task = self.analyze_stock(symbol, status_callback, progress_callback)
                batch_tasks.append(task)
            
            # Wait for all tasks in batch to complete
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                symbol = batch[j]
                
                if isinstance(result, Exception):
                    results.append({
                        'ticker': symbol,
                        'error': str(result),
                        'avg_volume_raw': None,
                        'iv30_rv30_raw': None,
                        'ts_slope_0_45_raw': None,
                        'expected_move': None,
                        'current_price': None
                    })
                elif 'error' in result:
                    results.append({
                        'ticker': symbol,
                        'error': result['error'],
                        'avg_volume_raw': None,
                        'iv30_rv30_raw': None,
                        'ts_slope_0_45_raw': None,
                        'expected_move': None,
                        'current_price': None
                    })
                else:
                    # Include raw metrics and booleans without status categorization
                    results.append({
                        'ticker': symbol,
                        'avg_volume': result['avg_volume'],
                        'iv30_rv30': result['iv30_rv30'],
                        'ts_slope_0_45': result['ts_slope_0_45'],
                        'avg_volume_raw': result['avg_volume_raw'],
                        'iv30_rv30_raw': result['iv30_rv30_raw'],
                        'ts_slope_0_45_raw': result['ts_slope_0_45_raw'],
                        'expected_move': result['expected_move'],
                        'current_price': result['current_price']
                    })
        
        return results
    
    def run_async(self, coro):
        """Run a coroutine in the event loop"""
        return asyncio.run_coroutine_threadsafe(coro, self.event_loop).result()
    
    async def disconnect(self):
        """Disconnect from IB"""
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
    
    def shutdown(self):
        """Shutdown the app and clean up resources"""
        self.run_async(self.disconnect())
        self.event_loop.stop()

def main_gui():
    # Initialize AsyncApp with its own event loop
    app = AsyncApp()
    
    # Start the event loop in a separate thread
    def run_event_loop():
        asyncio.set_event_loop(app.event_loop)
        app.event_loop.run_forever()
    
    loop_thread = threading.Thread(target=run_event_loop, daemon=True)
    loop_thread.start()
    
    # Connection settings layout with Bloomberg-like theme
    connection_layout = [
        [styles.bold_label("IB Connection Settings")],
        [styles.label_text("Host:"), styles.input_field("127.0.0.1", key="host")],
        [styles.label_text("Port:"), styles.input_field("7497", key="port"), 
         sg.Text("(TWS=7497, Gateway=4001)", font=styles.SMALL_FONT)],
        [styles.label_text("Client ID:"), styles.input_field("1", key="client_id")],
        [styles.primary_button("Connect")],
        [styles.status_text((40, 1), key="connection_status")]
    ]
    
    connection_window = sg.Window("IB Connection", connection_layout, **styles.window_params())
    
    while True:
        event, values = connection_window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            app.shutdown()
            break
        
        if event == "Connect":
            host = values.get("host", "127.0.0.1")
            port = int(values.get("port", 7497))
            client_id = int(values.get("client_id", 1))
            
            try:
                connected = app.run_async(app.init_connection(host, port, client_id))
                if connected:
                    connection_window["connection_status"].update("Connected to IB", text_color=styles.GREEN)
                    break
                else:
                    connection_window["connection_status"].update("Failed to connect", text_color=styles.RED)
            except Exception as e:
                connection_window["connection_status"].update(f"Connection failed: {str(e)}", text_color=styles.RED)
    
    if app.ib and app.ib.isConnected():
        connection_window.close()
        
        main_layout = [
            [styles.label_text("Enter Stock Symbols (comma-separated):"), 
             sg.Input(key="stocks", size=(40, 1), focus=True, background_color=styles.DARK_GRAY)],
            [styles.label_text("Concurrency Level:"),
             sg.Slider(range=(1, 10), default_value=3, orientation='h', size=(15, 15), key="concurrency")],
            [styles.primary_button("Submit", bind_return_key=True), 
             styles.secondary_button("Exit")],
            [styles.status_text(key="status")]
        ]
        
        window = sg.Window("Earnings Position Checker (IB)", main_layout, **styles.window_params())
        
        while True:
            event, values = window.read()
            if event in (sg.WINDOW_CLOSED, "Exit"):
                break
            
            if event == "Submit":
                window["status"].update("Processing...", text_color=styles.LIGHT_BLUE)
                stocks_input = values.get("stocks", "")
                
                if not stocks_input.strip():
                    window["status"].update("Please enter at least one stock symbol", text_color=styles.RED)
                    continue
                
                # Parse comma-separated tickers
                tickers = [ticker.strip().upper() for ticker in stocks_input.split(',') if ticker.strip()]
                
                if not tickers:
                    window["status"].update("Please enter valid stock symbols", text_color=styles.RED)
                    continue
                
                # Get concurrency level
                max_concurrent = int(values.get("concurrency", 3))
                
                # Create progress and status tracking for each ticker
                progress_callbacks = []
                status_callbacks = []
                progress_bars = []
                
                # Create a multi-progress window
                progress_layout = [
                    [styles.bold_label("Processing tickers in parallel...")],
                ]
                
                for ticker in tickers:
                    progress_layout.append([
                        styles.label_text(f"{ticker}:"),
                        styles.create_progress_bar(key=f'progress_{ticker}'),
                        sg.Text("", key=f'status_{ticker}', size=(30, 1))
                    ])
                
                progress_layout.append([styles.secondary_button("Cancel")])
                
                # Create the progress window
                progress_window = sg.Window("Processing", progress_layout, modal=True, **styles.window_params())
                
                # Setup callbacks for each ticker
                for ticker in tickers:
                    progress_bar = progress_window[f'progress_{ticker}']
                    progress_bars.append(progress_bar)
                    
                    # Create closure for each ticker's callbacks
                    def make_progress_callback(ticker_symbol):
                        def progress_callback(value):
                            progress_window[f'progress_{ticker_symbol}'].update(current_count=value)
                            progress_window.refresh()
                        return progress_callback
                    
                    def make_status_callback(ticker_symbol):
                        def status_callback(text):
                            progress_window[f'status_{ticker_symbol}'].update(text)
                            progress_window.refresh()
                        return status_callback
                    
                    progress_callbacks.append(make_progress_callback(ticker))
                    status_callbacks.append(make_status_callback(ticker))
                
                results = []
                cancelled = False
                
                # Define the function to process tickers in batch before creating the thread
                def process_tickers_batch(app, tickers, max_concurrent, progress_callbacks, status_callbacks):
                    nonlocal results, cancelled
                    try:
                        batch_results = app.run_async(
                            app.analyze_stocks_batch(tickers, max_concurrent, status_callbacks, progress_callbacks)
                        )
                        if not cancelled:
                            results = batch_results
                    except Exception as e:
                        if not cancelled:
                            results = [{'ticker': t, 'error': str(e)} for t in tickers]
                
                # Start processing in background
                processing_thread = threading.Thread(
                    target=lambda: process_tickers_batch(
                        app, tickers, max_concurrent, progress_callbacks, status_callbacks
                    ),
                    daemon=True
                )
                processing_thread.start()
                
                # Monitor the progress
                while processing_thread.is_alive():
                    event_progress, _ = progress_window.read(timeout=100)
                    
                    if event_progress == "Cancel" or event_progress == sg.WINDOW_CLOSED:
                        cancelled = True
                        # Stop processing by disconnecting and reconnecting
                        app.run_async(app.disconnect())
                        app.run_async(app.init_connection(
                            values.get("host", "127.0.0.1"),
                            int(values.get("port", 7497)),
                            int(values.get("client_id", 1))
                        ))
                        break
                
                # Close the progress window
                progress_window.close()
                
                if cancelled:
                    window["status"].update("Operation cancelled", text_color=styles.RED)
                    continue
                
                # Automatically save to output.csv
                if results:
                    try:
                        df = pd.DataFrame(results)
                        output_file = 'output.csv'
                        df.to_csv(output_file, index=False)
                        window["status"].update(f"Results saved to {output_file}", text_color=styles.GREEN)
                    except Exception as e:
                        window["status"].update(f"Error saving file: {str(e)}", text_color=styles.RED)
                
                # Display results if option selected
                if results:
                    # Create a table display with Bloomberg-like colors
                    table_headers = ['Ticker', 'Price', 'Volume (30d)', 'IV/RV Ratio', 'Term Structure', 'Expected Move']
                    table_data = []
                    
                    for result in results:
                        if 'error' in result:
                            row = [
                                result['ticker'],
                                'Error',
                                'Error',
                                'Error',
                                'Error',
                                'Error'
                            ]
                        else:
                            row = [
                                result['ticker'],
                                f"${result['current_price']:.2f}" if result['current_price'] else 'N/A',
                                f"{result['avg_volume_raw']:,.0f}" if result['avg_volume_raw'] else 'N/A',
                                f"{result['iv30_rv30_raw']:.2f}" if result['iv30_rv30_raw'] else 'N/A',
                                f"{result['ts_slope_0_45_raw']:.6f}" if result['ts_slope_0_45_raw'] else 'N/A',
                                result['expected_move'] if result['expected_move'] else 'N/A'
                            ]
                        table_data.append(row)
                    
                    # Color settings for the results window
                    results_layout = [
                        [styles.title_text("Analysis Results")],
                        [sg.Table(
                            **styles.table_params(
                                table_data, 
                                table_headers, 
                                min(25, len(results))
                            ),
                            col_widths=[10, 10, 15, 12, 15, 15],
                            key='-TABLE-',
                            enable_events=True,
                            enable_click_events=True  # Enable clicking on header to sort
                        )],
                        [styles.secondary_button("Close")]
                    ]
                    
                    results_window = sg.Window("Results", results_layout, resizable=True, 
                                             size=(800, 600), **styles.window_params())
                    
                    # This will be used to track the sort state and direction
                    sort_col_idx = 0  # Default sort by first column (Ticker)
                    sort_reversed = False  # Default sort ascending
                    
                    # Initial sort
                    table_data = sorted(table_data, key=lambda x: x[sort_col_idx])
                    results_window['-TABLE-'].update(values=table_data)
                    
                    while True:
                        event_result, values_result = results_window.read()
                        
                        if event_result == "Close" or event_result == sg.WINDOW_CLOSED:
                            break
                        
                        # Handle table header click for sorting
                        if isinstance(event_result, tuple) and event_result[0] == '-TABLE-' and event_result[2][0] == -1:
                            # Header was clicked
                            col_num = event_result[2][1]
                            
                            # Determine sort order
                            if col_num == sort_col_idx:
                                # Same column clicked, reverse the order
                                sort_reversed = not sort_reversed
                            else:
                                # New column clicked, sort ascending
                                sort_col_idx = col_num
                                sort_reversed = False
                            
                            # Sort the table data
                            try:
                                table_data = sorted(table_data, key=lambda x: (x[sort_col_idx] == 'N/A', x[sort_col_idx] == 'Error', x[sort_col_idx]), reverse=sort_reversed)
                            except:
                                # If there's an error (e.g., comparing strings with numbers),
                                # sort by string representation
                                table_data = sorted(table_data, key=lambda x: str(x[sort_col_idx]), reverse=sort_reversed)
                            
                            # Update the table
                            results_window['-TABLE-'].update(values=table_data)
                    
                    results_window.close()
                
                window["status"].update(f"Processed {len(results)} ticker(s). Results saved to output.csv", text_color=styles.GREEN)
                
        window.close()
    
    app.shutdown()

def gui():
    main_gui()

if __name__ == "__main__":
    gui() 