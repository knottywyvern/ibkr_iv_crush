# IB IV Crush Calculator
## Overview
This application analyzes stock options data from Interactive Brokers to identify potential trading opportunities based on volatility metrics. It evaluates stocks using three key criteria:
- High trading volume (average daily volume)
- High implied volatility relative to realized volatility (IV/RV ratio)
- Negative term structure slope
## Requirements
- Python 3.8 or higher
- Interactive Brokers TWS (Trader Workstation) or IB Gateway
- TWS/Gateway configured to accept API connections
## Installation
### 1. Clone or download this repository
```
git clone https://github.com/knottywyvern/iv_crush_ibkr.git
```
### 2. Create a virtual environment
#### Windows
```
python -m venv venv
venv\Scripts\activate
```
#### macOS/Linux
```
python3 -m venv venv
source venv/bin/activate
```
### 3. Install required packages
```
pip install -r requirements.txt
```
## Setup Interactive Brokers Connection
1. Launch TWS (Trader Workstation) or IB Gateway
2. Configure API settings:
   - In TWS: File > Global Configuration > API > Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set "Socket port" (default: 7497 for TWS, 4001 for Gateway)
   - Allow connections from localhost (127.0.0.1)
## Running the Application
1. Start TWS or IB Gateway and login
2. Ensure API connections are enabled
3. Run the script:
```
python ib_calculator.py
```
## Usage
1. In the connection window, enter:
   - Host: 127.0.0.1 (default for local connections)
   - Port: 7497 (for TWS) or 4001 (for Gateway)
   - Client ID: Any unique number (default: 1)
   - Click "Connect"
2. After connecting:
   - Enter stock symbols separated by commas (e.g., "AAPL, MSFT, GOOGL")
   - Set concurrency level (how many stocks to analyze simultaneously)
   - Click "Submit"
3. View results:
   - Results will appear in a table showing key metrics for each stock
   - Results are automatically saved to output.csv
## Output Metrics
- **Price**: Current stock price
- **Volume (30d)**: 30-day average trading volume
- **IV/RV Ratio**: Implied volatility to realized volatility ratio
- **Term Structure**: Slope of the volatility term structure
- **Expected Move**: Expected price movement based on options pricing
## Troubleshooting
- Ensure TWS/Gateway is running and logged in
- Verify API connections are enabled in TWS/Gateway
- Check that the port numbers match your TWS/Gateway configuration
- Confirm you have the required Python packages installed
## Disclaimer
This tool is for educational purposes only. It's not financial advice, and the developers accept no responsibility for investment decisions or losses. 