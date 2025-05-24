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
import logging
import sys
import platform
import time
import concurrent.futures

# Import the working connection function from ib_connections
from ib_connections import connect_to_tws, disconnect_from_tws

# Import the direct connection method
import ibkr_direct_connect

# Create a session for yfinance to use
session = curl_requests.Session(impersonate="chrome")

# Apply nest_asyncio to allow for nested event loops (required for GUI + asyncio)
nest_asyncio.apply()

# Apply the Bloomberg-like theme
styles.setup_bloomberg_theme()

# Setup logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"ib_calculator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = logging.getLogger("ib_calculator")
logger.setLevel(logging.INFO)  # Default to INFO level

# File handler for all logs
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler for debug level or higher
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
console_handler.setLevel(logging.WARNING)  # By default only show warnings and above
logger.addHandler(console_handler)

# Enable debug logging
def set_debug_mode(enable=True):
    logger.setLevel(logging.DEBUG if enable else logging.INFO)
    if enable:
        logger.debug("Debug logging enabled")

# Connect to Interactive Brokers TWS/Gateway - specifically for macOS
async def connect_to_ib(host='127.0.0.1', port=7497, client_id=1):
    """Connect to Interactive Brokers TWS or Gateway"""
    logger.info(f"Connecting to IB at {host}:{port} with client ID {client_id}")
    
    # On macOS, override to use 7496 which is open
    if platform.system() == "Darwin":  # macOS
        mac_port = 7496  # Hard-code to 7496 which we know is open
        logger.info(f"macOS detected - overriding port to {mac_port} which is known to be open")
        
        try:
            # Create a new IB instance
            ib = IB()
            # Set unlimited timeout
            ib.RequestTimeout = None
            
            # Apply the patch first
            util.patchAsyncio()
            logger.info(f"Applied asyncio patch")
            
            # Connect directly using the simplest approach
            logger.info(f"Directly connecting to {host}:{mac_port}")
            ib.connect(host, mac_port, clientId=client_id)
            
            if ib.isConnected():
                logger.info(f"Successfully connected to TWS on macOS using port {mac_port}")
                logger.info(f"Server version: {ib.client.serverVersion()}")
                return ib
            else:
                logger.error(f"Direct connection to {mac_port} failed - not connected")
                return None
                
        except Exception as e:
            error_msg = f"macOS connection failed: {str(e)}"
            logger.error(error_msg)
            logger.exception("macOS connection error details:")
            raise ConnectionError(error_msg)
    
    # Regular non-macOS connection path
    ib = IB()
    try:
        logger.debug(f"System info: {platform.system()} {platform.release()}")
        
        # Set timeout for IB operations
        ib.RequestTimeout = 10  # 10 second timeout for all requests
        logger.debug("Set request timeout to 10 seconds")
        
        # Log more detailed connection attempt
        logger.debug(f"Attempting connectAsync({host}, {port}, clientId={client_id})")
        await ib.connectAsync(host, port, clientId=client_id)
        
        # Ensure we're using the IB event loop for all operations
        await asyncio.sleep(0.1)  # Small sleep to initialize connection properly
        
        if ib.isConnected():
            logger.info("Successfully connected to IB")
            return ib
        else:
            logger.error("IB connection failed - not connected after attempt")
            return None
    except Exception as e:
        error_msg = f"Failed to connect to IB: {str(e)}"
        logger.error(error_msg)
        logger.exception("Detailed connection error:")
        raise ConnectionError(error_msg)

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
        
        # Determine the strike increment by looking at the spacing between strikes
        if len(strikes) >= 2:
            # Sort strikes and calculate the differences
            sorted_strikes = sorted(strikes)
            diffs = [sorted_strikes[i+1] - sorted_strikes[i] for i in range(len(sorted_strikes)-1)]
            
            # Find the most common difference (this is the likely increment)
            from collections import Counter
            if diffs:
                common_diffs = Counter(diffs).most_common()
                strike_increment = common_diffs[0][0]
            else:
                strike_increment = 1.0  # Default to 1.0 if can't determine
                
            # Round the underlying price to the nearest valid strike increment
            rounded_price = round(underlying_price / strike_increment) * strike_increment
            
            # Find the closest valid strike to the rounded price
            valid_strikes = [s for s in strikes if abs(s % strike_increment) < 0.01 or abs(s % strike_increment - strike_increment) < 0.01]
            
            # If no valid strikes found, use all strikes
            if not valid_strikes:
                valid_strikes = strikes
                
            atm_strike = min(valid_strikes, key=lambda x: abs(x - rounded_price))
        else:
            # If only one strike is available, use it
            atm_strike = strikes[0]
        
        # Create call and put option contracts
        call = Option(symbol, expiry_date, atm_strike, 'C', exchange)
        put = Option(symbol, expiry_date, atm_strike, 'P', exchange)
        
        # Qualify contracts separately and handle any errors
        try:
            await ib.qualifyContractsAsync(call)
        except Exception as e:
            # If we get an error, try another strike from the list
            if len(strikes) > 1:
                # Sort strikes by distance from the underlying price
                sorted_valid_strikes = sorted(valid_strikes if 'valid_strikes' in locals() else strikes, 
                                             key=lambda x: abs(x - underlying_price))
                
                # Try different strikes until we find one that works
                for backup_strike in sorted_valid_strikes[1:]:
                    try:
                        call = Option(symbol, expiry_date, backup_strike, 'C', exchange)
                        await ib.qualifyContractsAsync(call)
                        atm_strike = backup_strike  # Update the strike since this one works
                        break
                    except:
                        continue
                else:
                    # If all attempts with valid strikes fail, try only whole number strikes
                    whole_strikes = [s for s in strikes if s.is_integer()]
                    if whole_strikes:
                        closest_whole = min(whole_strikes, key=lambda x: abs(x - underlying_price))
                        try:
                            call = Option(symbol, expiry_date, closest_whole, 'C', exchange)
                            await ib.qualifyContractsAsync(call)
                            atm_strike = closest_whole
                        except:
                            raise ValueError(f"Failed to qualify call option after multiple attempts: {str(e)}")
                    else:
                        raise ValueError(f"Failed to qualify call option after multiple attempts: {str(e)}")
            else:
                raise ValueError(f"Failed to qualify call option: {str(e)}")
            
        try:
            # Update put strike to match the call strike that worked
            put = Option(symbol, expiry_date, atm_strike, 'P', exchange)
            await ib.qualifyContractsAsync(put)
        except Exception as e:
            # If put fails, try whole number strikes as a last resort
            whole_strikes = [s for s in strikes if s.is_integer()]
            if whole_strikes:
                closest_whole = min(whole_strikes, key=lambda x: abs(x - underlying_price))
                try:
                    put = Option(symbol, expiry_date, closest_whole, 'P', exchange)
                    await ib.qualifyContractsAsync(put)
                    atm_strike = closest_whole  # Also update the call with this working strike
                    call = Option(symbol, expiry_date, atm_strike, 'C', exchange)
                    await ib.qualifyContractsAsync(call)
                except:
                    raise ValueError(f"Failed to qualify put option: {str(e)}")
            else:
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
        else:
            print(f"Warning: No valid price data for {symbol} call option, strike {atm_strike}, expiry {expiry_date}")
            
        if not math.isnan(put_ticker.bid) and not math.isnan(put_ticker.ask):
            put_price = (put_ticker.bid + put_ticker.ask) / 2
        elif not math.isnan(put_ticker.last):
            put_price = put_ticker.last
        else:
            print(f"Warning: No valid price data for {symbol} put option, strike {atm_strike}, expiry {expiry_date}")
        
        # Calculate IV from straddle price
        call_iv = None
        put_iv = None
        call_vega = None
        put_vega = None
        
        # First try to get IV directly from the IB data if available
        try:
            # Get modelGreeks data
            if hasattr(call_ticker, 'modelGreeks') and call_ticker.modelGreeks:
                if hasattr(call_ticker.modelGreeks, 'implVol'):
                    call_iv = call_ticker.modelGreeks.implVol
                if hasattr(call_ticker.modelGreeks, 'vega'):
                    call_vega = call_ticker.modelGreeks.vega
                    # Convert to a more meaningful value (as IB returns a small decimal)
                    if call_vega is not None:
                        call_vega = call_vega * 100  # Convert to vega per 1% change in IV
            
            if hasattr(put_ticker, 'modelGreeks') and put_ticker.modelGreeks:
                if hasattr(put_ticker.modelGreeks, 'implVol'):
                    put_iv = put_ticker.modelGreeks.implVol
                if hasattr(put_ticker.modelGreeks, 'vega'):
                    put_vega = put_ticker.modelGreeks.vega
                    # Convert to a more meaningful value (as IB returns a small decimal)
                    if put_vega is not None:
                        put_vega = put_vega * 100  # Convert to vega per 1% change in IV
                        
            # If we couldn't get vega from modelGreeks, calculate it manually
            if call_vega is None or put_vega is None:
                # Calculate days to expiration
                exp_date_obj = datetime.strptime(expiry_date, "%Y%m%d").date()
                days_to_expiry = (exp_date_obj - datetime.today().date()).days
                years_to_expiry = days_to_expiry / 365.0
                
                # Get ATM IV
                atm_iv = (call_iv + put_iv) / 2.0 if call_iv is not None and put_iv is not None else None  # Return None instead of default 0.3
                
                # Approximate vega using Black-Scholes formula for ATM options
                # Vega ≈ S * sqrt(T) * phi(0) / 100
                # where phi(0) is the standard normal PDF at 0, which is approximately 0.4
                if call_vega is None and years_to_expiry > 0 and atm_iv is not None:
                    call_vega = underlying_price * math.sqrt(years_to_expiry) * 0.4 * atm_iv
                
                if put_vega is None and years_to_expiry > 0 and atm_iv is not None:
                    put_vega = underlying_price * math.sqrt(years_to_expiry) * 0.4 * atm_iv
                
                # Special case for 0 DTE - estimate vega from price instead when it's very close to expiration
                if (call_vega is None or put_vega is None) and days_to_expiry <= 1:
                    logger.debug(f"Using alternative Vega estimation for near-expiry options ({days_to_expiry} DTE)")
                    
                    # For near-expiry options, we can estimate vega using a small IV change simulation
                    # This assumes a small artificial IV change and observes the theoretical price impact
                    if atm_iv is None and call_price is not None and put_price is not None:
                        # Rough IV estimate using price and simple model for very short-dated options
                        # We're using the straddle price as approximation of expected move
                        straddle_price = call_price + put_price
                        # For extremely short-dated options, IV can be approximated as:
                        # IV ≈ straddle_price / underlying_price / sqrt(1/365)
                        one_day_factor = math.sqrt(1/365.0)
                        atm_iv = (straddle_price / underlying_price) / one_day_factor
                    
                    # Use a small IV change to estimate vega if we have IV
                    if atm_iv is not None:
                        # For 0 DTE, use a small time factor instead of 0
                        min_time_factor = 1/365.0  # 1 day as fraction of year
                        
                        # Calculate approximate vega for 0-1 DTE
                        # Scale by time remaining in day (use at least 1 hour of time value)
                        hours_remaining = max(4, 24 - datetime.now().hour)
                        day_fraction = hours_remaining / 24.0
                        time_factor = min_time_factor * day_fraction
                        
                        # Calculate vega for short-dated options (smaller than normal but not zero)
                        small_vega = underlying_price * math.sqrt(time_factor) * 0.2 * atm_iv  # Use 0.2 factor for short-dated
                        
                        if call_vega is None:
                            call_vega = small_vega
                        
                        if put_vega is None:
                            put_vega = small_vega
                        
                        logger.debug(f"Estimated Vega for {expiry_date} ({days_to_expiry} DTE): {small_vega}")
        except Exception as e:
            logger.error(f"Error calculating Greeks: {str(e)}")
            logger.exception("Details:")
        
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
                
                if call_iv is None:
                    call_iv = straddle_iv
                
                if put_iv is None:
                    put_iv = straddle_iv
        
        # Create call dataframe
        call_data = {
            'strike': atm_strike,
            'bid': call_ticker.bid if not math.isnan(call_ticker.bid) else None,
            'ask': call_ticker.ask if not math.isnan(call_ticker.ask) else None,
            'impliedVolatility': call_iv,
            'vega': call_vega
        }
        
        # Create put dataframe
        put_data = {
            'strike': atm_strike,
            'bid': put_ticker.bid if not math.isnan(put_ticker.bid) else None,
            'ask': put_ticker.ask if not math.isnan(put_ticker.ask) else None,
            'impliedVolatility': put_iv,
            'vega': put_vega
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
                    logger.warning(f"Missing IV data for {ticker} at expiry {exp_date}")
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
        
        # Calculate RV and handle potential errors
        rv30 = yang_zhang(price_history)
        if rv30 <= 0 or math.isnan(rv30):
            return {
                'error': "Invalid realized volatility calculation (zero or negative value)",
                'avg_vega': None
            }
            
        iv30 = term_spline(30)
        if iv30 <= 0 or math.isnan(iv30):
            return {
                'error': "Invalid implied volatility calculation (zero or negative value)",
                'avg_vega': None
            }
            
        iv30_rv30_ratio = iv30 / rv30
        expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None
        
        # Fallback calculation for expected move if straddle is None but we have IV
        if expected_move is None and iv30 is not None:
            # Use 30-day IV to estimate the expected move
            # Expected move ≈ Stock Price * IV * sqrt(time in years)
            # For a standard 1-month expected move, use sqrt(30/365)
            time_factor = math.sqrt(30/365)
            estimated_straddle = underlying_price * iv30 * time_factor
            expected_move = str(round(estimated_straddle / underlying_price * 100, 2)) + "%"
        
        # Get average volume from Yahoo Finance instead of IB
        avg_volume = await get_average_volume(ticker)
        
        return {
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
            'current_price': underlying_price,
            'avg_vega': None
        }
    except Exception as e:
        raise Exception(f'Error occurred processing: {str(e)}')

class AsyncApp:
    def __init__(self):
        self.ib = None
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
    
    async def init_connection(self, host, port, client_id):
        logger.info(f"Initializing connection to {host}:{port}")
        self.ib = await connect_to_ib(host, port, client_id)
        connected = self.ib is not None and self.ib.isConnected()
        if connected:
            logger.info("Connection successful")
        else:
            logger.error("Connection initialization failed")
        return connected
    
    async def fetch_option_data_for_date(self, symbol, exp_date, underlying_price, status_callback=None):
        """Helper method to fetch option data for a single expiration date"""
        try:
            if status_callback:
                status_callback(f"Processing expiry {exp_date}...")
            
            calls, puts = await get_options_chain(self.ib, symbol, exp_date, underlying_price)
            
            if calls.empty or puts.empty:
                return (exp_date, None, None, None, None)
            
            call_iv = calls.loc[0, 'impliedVolatility']
            put_iv = puts.loc[0, 'impliedVolatility']
            
            # Get vega values
            call_vega = calls.loc[0, 'vega']
            put_vega = puts.loc[0, 'vega']
            
            # Calculate average straddle vega
            straddle_vega = None
            # If either call or put vega is available, use it (or average if both available)
            if call_vega is not None and put_vega is not None:
                straddle_vega = (call_vega + put_vega) / 2.0
            elif call_vega is not None:
                straddle_vega = call_vega
            elif put_vega is not None:
                straddle_vega = put_vega
            
            if call_iv is None or put_iv is None:
                return (exp_date, None, None, None, straddle_vega)
            
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
            # Fallback: If we can't get straddle from bid/ask but we have IV, estimate the straddle price
            elif atm_iv_value is not None:
                # Parse expiration date to calculate days to expiry
                exp_date_obj = datetime.strptime(exp_date, "%Y%m%d").date()
                days_to_expiry = (exp_date_obj - datetime.today().date()).days
                years_to_expiry = days_to_expiry / 365.0
                
                # Approximate ATM straddle price using IV and time to expiry
                # Formula: Stock Price * IV * sqrt(time to expiry)
                if years_to_expiry > 0:
                    straddle = underlying_price * atm_iv_value * math.sqrt(years_to_expiry)
            
            return (exp_date, atm_iv_value, straddle, True, straddle_vega)
            
        except Exception as e:
            print(f"Error processing expiry {exp_date}: {str(e)}")
            return (exp_date, None, None, None, None)

    async def analyze_stock(self, symbol, status_callback=None, progress_callback=None):
        """Analyze a stock with progress updates"""
        if not self.ib or not self.ib.isConnected():
            return {
                'error': 'Not connected to IB',
                'avg_vega': None
            }
        
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
                    return {
                        'error': "No options found for this stock",
                        'avg_vega': None
                    }
                if status_callback:
                    status_callback(f"Found {len(exp_dates)} option expiration dates within 45 DTE")
            except Exception as e:
                return {
                    'error': f"No options found: {str(e)}",
                    'avg_vega': None
                }
            
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
                    return {
                        'error': "No suitable expiration dates found",
                        'avg_vega': None
                    }
                
                # Sort dates for consistency
                exp_dates = sorted(exp_dates)
                
            except Exception as e:
                return {
                    'error': f"Error processing expiration dates: {str(e)}",
                    'avg_vega': None
                }
            
            # Update progress to 30%
            if progress_callback:
                progress_callback(30)
            if status_callback:
                status_callback("Getting current stock price...")
            
            # Get current price
            try:
                underlying_price = await get_current_price(self.ib, symbol)
                if underlying_price is None:
                    return {
                        'error': "No market price found",
                        'avg_vega': None
                    }
            except Exception as e:
                return {
                    'error': f"Unable to retrieve stock price: {str(e)}",
                    'avg_vega': None
                }
            
            # Update progress to 40%
            if progress_callback:
                progress_callback(40)
            if status_callback:
                status_callback("Processing option chains in parallel...")
            
            # Process option chains in parallel with concurrency limit
            max_date_concurrency = 3  # Maximum number of dates to process concurrently
            atm_iv = {}
            straddle = None
            straddle_vega = None
            
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
                for exp_date, atm_iv_value, exp_straddle, success, exp_vega in batch_results:
                    if success and atm_iv_value is not None:
                        atm_iv[exp_date] = atm_iv_value
                        # Use straddle and vega from first expiration only
                        if exp_date == exp_dates[0]:
                            if exp_straddle is not None:
                                straddle = exp_straddle
                            if exp_vega is not None:
                                straddle_vega = exp_vega
                
                # Update progress incrementally
                if progress_callback:
                    current_progress = 40 + int(((i + len(batch)) / len(exp_dates)) * 30)
                    progress_callback(min(70, current_progress))  # Cap at 70%
            
            if not atm_iv:
                return {
                    'error': "Could not determine ATM IV for any expiration dates",
                    'avg_vega': None
                }
            
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
            if rv30 <= 0 or math.isnan(rv30):
                return {
                    'error': "Invalid realized volatility calculation (zero or negative value)",
                    'avg_vega': None
                }
                
            iv30 = term_spline(30)
            if iv30 <= 0 or math.isnan(iv30):
                return {
                    'error': "Invalid implied volatility calculation (zero or negative value)",
                    'avg_vega': None
                }
                
            iv30_rv30_ratio = iv30 / rv30
            expected_move = str(round(straddle / underlying_price * 100, 2)) + "%" if straddle else None
            
            # Fallback calculation for expected move if straddle is None but we have IV
            if expected_move is None and iv30 is not None:
                # Use 30-day IV to estimate the expected move
                # Expected move ≈ Stock Price * IV * sqrt(time in years)
                # For a standard 1-month expected move, use sqrt(30/365)
                time_factor = math.sqrt(30/365)
                estimated_straddle = underlying_price * iv30 * time_factor
                expected_move = str(round(estimated_straddle / underlying_price * 100, 2)) + "%"
            
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
                'current_price': underlying_price,
                'avg_vega': straddle_vega
            }
        except Exception as e:
            result = {
                'error': f"Error occurred: {str(e)}",
                'avg_vega': None
            }
        
        return result
    
    async def analyze_stocks_batch(self, symbols, max_concurrent=5, status_callbacks=None, progress_callbacks=None):
        """Process multiple stocks concurrently with a limit on concurrency"""
        if not self.ib or not self.ib.isConnected():
            return [{
                'ticker': symbol, 
                'error': 'Not connected to IB',
                'avg_volume_raw': None,
                'iv30_rv30_raw': None,
                'ts_slope_0_45_raw': None,
                'expected_move': None,
                'current_price': None,
                'avg_vega': None
            } for symbol in symbols]
        
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
                        'current_price': None,
                        'avg_vega': None
                    })
                elif 'error' in result:
                    results.append({
                        'ticker': symbol,
                        'error': result['error'],
                        'avg_volume_raw': None,
                        'iv30_rv30_raw': None,
                        'ts_slope_0_45_raw': None,
                        'expected_move': None,
                        'current_price': None,
                        'avg_vega': None
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
                        'current_price': result['current_price'],
                        'avg_vega': result['avg_vega']
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
    
    # Log the application startup
    logger.info("Application started")
    logger.info(f"System: {platform.system()} {platform.release()}")
    
    # Connection settings layout with Bloomberg-like theme
    connection_layout = [
        [styles.bold_label("IB Connection Settings")],
        [styles.label_text("Host:"), styles.input_field("127.0.0.1", key="host")],
        [styles.label_text("Port:"), styles.input_field("7497", key="port"), 
         sg.Text("(TWS=7497, Gateway=4001)", font=styles.SMALL_FONT)],
        [styles.label_text("Client ID:"), styles.input_field("1", key="client_id")],
        [sg.Checkbox("Debug Mode", key="debug_mode", enable_events=True, 
                    background_color=styles.BLACK, text_color=styles.LIGHT_BLUE)],
        [styles.primary_button("Connect")],
        [styles.status_text((80, 2), key="connection_status")],  # Increased width and height for status
    ]
    
    # Increase the window width to accommodate longer error messages
    connection_window = sg.Window("IB Connection", connection_layout, size=(600, 250), **styles.window_params())
    
    # Show log file location on startup
    connection_window["connection_status"].update(
        f"Log file location: {log_file}",
        text_color=styles.LIGHT_BLUE
    )
    
    while True:
        event, values = connection_window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            logger.info("Connection window closed")
            app.shutdown()
            break
        
        if event == "debug_mode":
            debug_enabled = values.get("debug_mode", False)
            set_debug_mode(debug_enabled)
            connection_window["connection_status"].update(
                f"Debug mode {('enabled' if debug_enabled else 'disabled')} - Logs saved to {log_file}", 
                text_color=styles.LIGHT_BLUE
            )
        
        if event == "Connect":
            host = values.get("host", "127.0.0.1")
            port = int(values.get("port", 7497))
            client_id = int(values.get("client_id", 1))
            
            logger.info(f"Connect button pressed - Host: {host}, Port: {port}, Client ID: {client_id}")
            connection_window["connection_status"].update("Connecting to IB...", text_color=styles.AMBER)
            connection_window.refresh()
            
            try:
                connected = app.run_async(app.init_connection(host, port, client_id))
                if connected:
                    logger.info("Connection successful")
                    connection_window["connection_status"].update("Connected to IB", text_color=styles.GREEN)
                    break
                else:
                    logger.error("Connection failed - IB returned not connected")
                    mac_tip = ""
                    if platform.system() == "Darwin":  # macOS specific help
                        mac_tip = "\nOn Mac: Check TWS/Gateway settings > API > Enable ActiveX and Socket Clients"
                    
                    connection_window["connection_status"].update(
                        f"Failed to connect - Check TWS/Gateway is running and accepting connections{mac_tip}", 
                        text_color=styles.RED
                    )
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Connection exception: {error_msg}")
                
                # Add Mac-specific troubleshooting tip
                mac_tip = ""
                if platform.system() == "Darwin":  # macOS specific help
                    mac_tip = "\nOn Mac: Check TWS/Gateway settings > API > Enable ActiveX and Socket Clients"
                
                connection_window["connection_status"].update(
                    f"Connection failed: {error_msg}{mac_tip}\nSee log file for details: {log_file}", 
                    text_color=styles.RED
                )
    
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
                
                # Save to output.csv
                if results:
                    try:
                        df = pd.DataFrame(results)
                        output_file = 'output.csv'
                        df.to_csv(output_file, index=False)
                        window["status"].update(f"Results saved to {output_file}", text_color=styles.GREEN)
                        
                        # Print results to command line for easy copying
                        print("\n===== ANALYSIS RESULTS =====")
                        print("Ticker | Price | Volume (30d) | IV/RV Ratio | Term Structure | Expected Move | Avg Vega")
                        print("-" * 100)
                        for result in results:
                            ticker = result['ticker']
                            if 'error' in result:
                                print(f"{ticker} | Error | Error | Error | Error | Error | Error")
                            else:
                                price = f"${result.get('current_price', 0):.2f}" if result.get('current_price') else 'N/A'
                                volume = f"{result.get('avg_volume_raw', 0):,.0f}" if result.get('avg_volume_raw') else 'N/A'
                                iv_rv = f"{result.get('iv30_rv30_raw', 0):.2f}" if result.get('iv30_rv30_raw') else 'N/A'
                                ts_slope = f"{result.get('ts_slope_0_45_raw', 0):.6f}" if result.get('ts_slope_0_45_raw') else 'N/A'
                                exp_move = result.get('expected_move', 'N/A')
                                vega = f"{result.get('avg_vega', 0):.2f}" if result.get('avg_vega') is not None else 'N/A'
                                print(f"{ticker} | {price} | {volume} | {iv_rv} | {ts_slope} | {exp_move} | {vega}")
                        print("===== END OF RESULTS =====\n")
                    except Exception as e:
                        window["status"].update(f"Error saving file: {str(e)}", text_color=styles.RED)
                
                # Display results if option selected
                if results:
                    # Create a table display with Bloomberg-like colors
                    table_headers = ['Ticker', 'Price', 'Volume (30d)', 'IV/RV Ratio', 'Term Structure', 'Expected Move', 'Avg Vega']
                    table_data = []
                    
                    for result in results:
                        if 'error' in result:
                            row = [
                                result['ticker'],
                                'Error',
                                'Error',
                                'Error',
                                'Error',
                                'Error',
                                'Error'
                            ]
                        else:
                            row = [
                                result['ticker'],
                                f"${result.get('current_price', None):.2f}" if result.get('current_price') else 'N/A',
                                f"{result.get('avg_volume_raw', None):,.0f}" if result.get('avg_volume_raw') else 'N/A',
                                f"{result.get('iv30_rv30_raw', None):.2f}" if result.get('iv30_rv30_raw') else 'N/A',
                                f"{result.get('ts_slope_0_45_raw', None):.6f}" if result.get('ts_slope_0_45_raw') else 'N/A',
                                result.get('expected_move', 'N/A'),
                                f"{result.get('avg_vega', None):.2f}" if result.get('avg_vega') is not None else 'N/A'
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
                            col_widths=[10, 10, 15, 10, 12, 12, 10],
                            key='-TABLE-',
                            enable_events=True,
                            enable_click_events=True  # Enable clicking on header to sort
                        )],
                        [styles.secondary_button("Close")]
                    ]
                    
                    results_window = sg.Window("Results", results_layout, resizable=True, 
                                             size=(900, 600), **styles.window_params())
                    
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