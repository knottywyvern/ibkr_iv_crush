"""
Simple module for direct IBKR connection that works on macOS
"""
from ib_insync import *
import sys

def connect_direct(host='127.0.0.1', port=7496, clientId=1):
    """Connect directly to TWS using approach that works on macOS"""
    # Create a new IB instance
    ib = IB()
    
    try:
        # Apply the patch first (important!)
        util.patchAsyncio()
        # Wait a moment
        util.sleep(1)
        
        # Direct connection
        ib.connect(host, port, clientId)
        print(f"Successfully connected to TWS at {host}:{port}")
        print(f"Server version: {ib.client.serverVersion()}")
        return ib
    except Exception as e:
        print(f"Failed to connect to TWS: {e}")
        return None

# For testing this module directly
if __name__ == "__main__":
    # Test the connection
    ib = connect_direct()
    
    if ib and ib.isConnected():
        print("Connectivity test successful")
        # Disconnect
        ib.disconnect()
        print("Disconnected from TWS")
    else:
        print("Connection test failed")
        sys.exit(1) 